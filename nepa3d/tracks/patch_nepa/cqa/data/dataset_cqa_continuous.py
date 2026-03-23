from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ASK_DISTANCE, ASK_NORMAL, CQA_VOCAB_VERSION
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import (
    QUERY_ORDER_MODES,
    TASK_REGISTRY,
    _choice,
    _normalize_query_src_filter,
    _ordered_xyz_perm,
    _resolve_query_order,
    _select_optional_bank,
)


def _np_to_torch_f32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _np_to_torch_i64(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


def _apply_query_order_continuous(
    *,
    qry_xyz: np.ndarray,
    target_vec: np.ndarray,
    target_mask: np.ndarray,
    qry_src_code: np.ndarray,
    rng: Any,
    query_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if int(qry_xyz.shape[0]) <= 1:
        return qry_xyz, target_vec, target_mask, qry_src_code
    if query_order == "sampled":
        return qry_xyz, target_vec, target_mask, qry_src_code
    if query_order == "shuffled":
        perm = rng.permutation(int(qry_xyz.shape[0])).astype(np.int64)
    elif query_order == "ordered_xyz":
        perm = _ordered_xyz_perm(qry_xyz)
    else:
        raise KeyError(f"unknown query_order={query_order}")
    return qry_xyz[perm], target_vec[perm], target_mask[perm], qry_src_code[perm]


class V2PrimitiveCQAContinuousDataset(Dataset):
    """Continuous typed-answer dataset for `mesh_normal` and `udf_distance`.

    The output target is a shared 3D vector with a mask:
      - `udf_distance`: `[dist, 0, 0]`, mask `[1, 0, 0]`
      - `mesh_normal` : `[nx, ny, nz]`, mask `[1, 1, 1]`
    """

    def __init__(
        self,
        paths: Sequence[str],
        *,
        task_name: str,
        context_source: str = "surf",
        n_ctx: int = 2048,
        n_qry: int = 64,
        seed: int = 0,
        mode: str = "train",
        query_src_filter: Any = None,
        query_dist_min: float | None = None,
        query_dist_max: float | None = None,
        query_order: str | None = None,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        if task_name not in {"mesh_normal", "udf_distance"}:
            raise KeyError(f"continuous dataset only supports mesh_normal/udf_distance, got {task_name}")
        self.task = TASK_REGISTRY[task_name]
        self.context_source = str(context_source)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)
        self.query_src_filter = _normalize_query_src_filter(query_src_filter)
        self.query_dist_min = None if query_dist_min is None else float(query_dist_min)
        self.query_dist_max = None if query_dist_max is None else float(query_dist_max)
        self.query_order = _resolve_query_order(query_order, mode=self.mode)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + idx)
        with np.load(path, allow_pickle=False) as npz:
            if self.context_source == "surf":
                ctx_all = np.asarray(npz["surf_xyz"], dtype=np.float32)
                ctx_bank_idx = None
            elif self.context_source == "pc_bank":
                ctx_all, ctx_bank_idx = _select_optional_bank(
                    np.asarray(npz["pc_ctx_bank_xyz"], dtype=np.float32),
                    rng,
                    self.mode,
                    fallback_seed=self.seed + idx,
                )
            else:
                raise ValueError(f"unknown context_source={self.context_source}")

            ctx_idx = _choice(int(ctx_all.shape[0]), self.n_ctx, rng)
            ctx_xyz = np.asarray(ctx_all[ctx_idx], dtype=np.float32)

            qry_all = np.asarray(npz[self.task.query_xyz_key], dtype=np.float32)
            q_pool = np.arange(int(qry_all.shape[0]), dtype=np.int64)
            qry_src_code_all = None
            if self.task.query_xyz_key == "udf_qry_xyz":
                qry_src_code_all = np.asarray(npz["udf_qry_src_code"], dtype=np.int64).reshape(-1)
                if self.query_src_filter is not None:
                    q_pool = q_pool[np.isin(qry_src_code_all, np.asarray(self.query_src_filter, dtype=np.int64))]
                if self.query_dist_min is not None or self.query_dist_max is not None:
                    dist = np.asarray(npz["udf_qry_dist"], dtype=np.float32).reshape(-1)
                    keep = np.ones((int(dist.shape[0]),), dtype=np.bool_)
                    if self.query_dist_min is not None:
                        keep &= dist >= float(self.query_dist_min)
                    if self.query_dist_max is not None:
                        keep &= dist <= float(self.query_dist_max)
                    q_pool = q_pool[keep[q_pool]]
            if int(q_pool.shape[0]) <= 0:
                raise RuntimeError(
                    f"empty query pool after filtering: task={self.task.name} path={path} "
                    f"query_src_filter={self.query_src_filter} query_dist_min={self.query_dist_min} "
                    f"query_dist_max={self.query_dist_max}"
                )
            if self.n_qry >= int(q_pool.shape[0]):
                q_idx = q_pool
            else:
                take = _choice(int(q_pool.shape[0]), self.n_qry, rng)
                q_idx = q_pool[take]
            qry_xyz = np.asarray(qry_all[q_idx], dtype=np.float32)

            if qry_src_code_all is None:
                qry_src_code = np.full((int(q_idx.shape[0]),), -1, dtype=np.int64)
            else:
                qry_src_code = np.asarray(qry_src_code_all[q_idx], dtype=np.int64)

            if self.task.query_type == ASK_DISTANCE:
                dist = np.asarray(npz["udf_qry_dist"], dtype=np.float32).reshape(-1)[q_idx]
                target_vec = np.zeros((int(q_idx.shape[0]), 3), dtype=np.float32)
                target_vec[:, 0] = dist
                target_mask = np.zeros_like(target_vec)
                target_mask[:, 0] = 1.0
            elif self.task.query_type == ASK_NORMAL:
                normal = np.asarray(npz["mesh_surf_n"], dtype=np.float32)[q_idx]
                normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
                target_vec = normal.astype(np.float32, copy=False)
                target_mask = np.ones_like(target_vec, dtype=np.float32)
            else:
                raise AssertionError("unreachable")

            qry_xyz, target_vec, target_mask, qry_src_code = _apply_query_order_continuous(
                qry_xyz=qry_xyz,
                target_vec=target_vec,
                target_mask=target_mask,
                qry_src_code=qry_src_code,
                rng=rng,
                query_order=self.query_order,
            )

            path_obj = Path(path)
            return {
                "ctx_xyz": _np_to_torch_f32(ctx_xyz),
                "qry_xyz": _np_to_torch_f32(qry_xyz),
                "qry_type": torch.full((int(qry_xyz.shape[0]),), int(self.task.query_type), dtype=torch.long),
                "target_vec": _np_to_torch_f32(target_vec),
                "target_mask": _np_to_torch_f32(target_mask),
                "qry_src_code": _np_to_torch_i64(qry_src_code),
                "task_name": self.task.name,
                "context_source": self.context_source,
                "cache_split": path_obj.parent.parent.name,
                "synset": path_obj.parent.name,
                "path": path,
                "context_bank_idx": None if ctx_bank_idx is None else int(ctx_bank_idx),
                "query_src_filter": None if self.query_src_filter is None else list(self.query_src_filter),
                "query_dist_min": self.query_dist_min,
                "query_dist_max": self.query_dist_max,
                "query_order": self.query_order,
                "vocab_version": CQA_VOCAB_VERSION,
            }


def cqa_continuous_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ctx_xyz"] = torch.stack([b["ctx_xyz"] for b in batch], dim=0)
    out["qry_xyz"] = torch.stack([b["qry_xyz"] for b in batch], dim=0)
    out["qry_type"] = torch.stack([b["qry_type"] for b in batch], dim=0)
    out["target_vec"] = torch.stack([b["target_vec"] for b in batch], dim=0)
    out["target_mask"] = torch.stack([b["target_mask"] for b in batch], dim=0)
    out["qry_src_code"] = torch.stack([b["qry_src_code"] for b in batch], dim=0)
    out["task_name"] = [b["task_name"] for b in batch]
    out["context_source"] = [b["context_source"] for b in batch]
    out["cache_split"] = [b["cache_split"] for b in batch]
    out["synset"] = [b["synset"] for b in batch]
    out["path"] = [b["path"] for b in batch]
    out["context_bank_idx"] = [b.get("context_bank_idx") for b in batch]
    out["query_src_filter"] = [b.get("query_src_filter") for b in batch]
    out["query_dist_min"] = [b.get("query_dist_min") for b in batch]
    out["query_dist_max"] = [b.get("query_dist_max") for b in batch]
    out["query_order"] = [b.get("query_order") for b in batch]
    out["vocab_version"] = str(batch[0].get("vocab_version", CQA_VOCAB_VERSION))
    return out
