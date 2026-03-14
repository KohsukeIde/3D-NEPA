from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cqa_codec import (
    ASK_CLEARANCE,
    ASK_CURVATURE,
    ASK_DISTANCE,
    ASK_NORMAL,
    ASK_THICKNESS,
    ASK_VISIBILITY,
    ANSWER_VOCAB_SIZE,
    CQA_VOCAB_VERSION,
    encode_answers_from_fields,
)


def _choice(n: int, k: int, rng: Any) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def _select_optional_bank(arr: np.ndarray, rng: Any, mode: str, *, fallback_seed: int = 0):
    arr = np.asarray(arr)
    if arr.ndim < 3:
        return arr, None
    nb = int(arr.shape[0])
    if mode == "train":
        b = int(rng.randint(0, nb))
    else:
        b = int(fallback_seed % max(nb, 1))
    return np.asarray(arr[b]), b


def _np_to_torch_f32(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _np_to_torch_i64(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.int64))


@dataclass(frozen=True)
class CQATaskSpec:
    name: str
    query_type: int
    query_xyz_key: str
    field_keys: Dict[str, str]


TASK_REGISTRY: Dict[str, CQATaskSpec] = {
    # Surface-aligned tasks always use surf_xyz as carrier.
    "mesh_normal": CQATaskSpec(
        name="mesh_normal",
        query_type=ASK_NORMAL,
        query_xyz_key="surf_xyz",
        field_keys={"normal": "mesh_surf_n"},
    ),
    "mesh_visibility": CQATaskSpec(
        name="mesh_visibility",
        query_type=ASK_VISIBILITY,
        query_xyz_key="surf_xyz",
        field_keys={"visibility": "mesh_surf_vis_sig"},
    ),
    "mesh_curvature": CQATaskSpec(
        name="mesh_curvature",
        query_type=ASK_CURVATURE,
        query_xyz_key="surf_xyz",
        field_keys={"curvature": "mesh_surf_curv"},
    ),
    "udf_thickness": CQATaskSpec(
        name="udf_thickness",
        query_type=ASK_THICKNESS,
        query_xyz_key="surf_xyz",
        field_keys={"thickness": "udf_surf_thickness"},
    ),
    "udf_clearance": CQATaskSpec(
        name="udf_clearance",
        query_type=ASK_CLEARANCE,
        query_xyz_key="surf_xyz",
        field_keys={"clearance": "udf_surf_clear_front"},
    ),
    # Explicit off-surface query task.
    "udf_distance": CQATaskSpec(
        name="udf_distance",
        query_type=ASK_DISTANCE,
        query_xyz_key="udf_qry_xyz",
        field_keys={"distance": "udf_qry_dist"},
    ),
}


class V2PrimitiveCQADataset(Dataset):
    """Explicit-query CQA samples backed by v2 world-package caches."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        task_name: str,
        context_source: str = "surf",  # surf | pc_bank
        n_ctx: int = 2048,
        n_qry: int = 64,
        seed: int = 0,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        if task_name not in TASK_REGISTRY:
            raise KeyError(f"unknown task_name={task_name}")
        self.task = TASK_REGISTRY[task_name]
        self.context_source = str(context_source)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)

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
            q_idx = _choice(int(qry_all.shape[0]), self.n_qry, rng)
            qry_xyz = np.asarray(qry_all[q_idx], dtype=np.float32)

            fields: Dict[str, np.ndarray] = {}
            for alias, key in self.task.field_keys.items():
                arr = np.asarray(npz[key], dtype=np.float32)
                fields[alias] = arr[q_idx]
            answer_code = encode_answers_from_fields(self.task.query_type, fields)

            if self.mode == "train":
                perm = rng.permutation(int(qry_xyz.shape[0])).astype(np.int64)
                qry_xyz = qry_xyz[perm]
                answer_code = answer_code[perm]

            path_obj = Path(path)
            out: Dict[str, Any] = {
                "ctx_xyz": _np_to_torch_f32(ctx_xyz),
                "qry_xyz": _np_to_torch_f32(qry_xyz),
                "qry_type": torch.full((int(qry_xyz.shape[0]),), int(self.task.query_type), dtype=torch.long),
                "answer_code": _np_to_torch_i64(answer_code),
                "task_name": self.task.name,
                "context_source": self.context_source,
                "cache_split": path_obj.parent.parent.name,
                "synset": path_obj.parent.name,
                "path": path,
                "context_bank_idx": None if ctx_bank_idx is None else int(ctx_bank_idx),
                "answer_vocab_size": int(ANSWER_VOCAB_SIZE),
                "vocab_version": CQA_VOCAB_VERSION,
            }
            return out


def cqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ctx_xyz"] = torch.stack([b["ctx_xyz"] for b in batch], dim=0)
    out["qry_xyz"] = torch.stack([b["qry_xyz"] for b in batch], dim=0)
    out["qry_type"] = torch.stack([b["qry_type"] for b in batch], dim=0)
    out["answer_code"] = torch.stack([b["answer_code"] for b in batch], dim=0)
    out["task_name"] = [b["task_name"] for b in batch]
    out["context_source"] = [b["context_source"] for b in batch]
    out["cache_split"] = [b["cache_split"] for b in batch]
    out["synset"] = [b["synset"] for b in batch]
    out["path"] = [b["path"] for b in batch]
    out["context_bank_idx"] = [b.get("context_bank_idx") for b in batch]
    out["answer_vocab_size"] = int(batch[0].get("answer_vocab_size", ANSWER_VOCAB_SIZE))
    out["vocab_version"] = str(batch[0].get("vocab_version", CQA_VOCAB_VERSION))
    return out
