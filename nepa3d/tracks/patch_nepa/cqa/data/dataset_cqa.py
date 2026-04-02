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
    answer_vocab_size,
    encode_answers_from_fields,
    query_name_to_id,
)

QUERY_ORDER_MODES = ("sampled", "shuffled", "ordered_xyz")


def _choice(n: int, k: int, rng: Any) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def _choice_with_replacement_if_needed(n: int, k: int, rng: Any) -> np.ndarray:
    if int(n) <= 0:
        raise ValueError("cannot sample from an empty pool")
    if int(k) <= int(n):
        return _choice(int(n), int(k), rng)
    base = np.arange(int(n), dtype=np.int64)
    extra = rng.choice(int(n), size=int(k) - int(n), replace=True).astype(np.int64)
    return np.concatenate([base, extra], axis=0)


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


def _resolve_query_order(query_order: Any, *, mode: str) -> str:
    text = str(query_order or "").strip().lower()
    if not text:
        return "shuffled" if str(mode) == "train" else "sampled"
    if text not in QUERY_ORDER_MODES:
        raise KeyError(f"unknown query_order={query_order}")
    return text


def _ordered_xyz_perm(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    return np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0])).astype(np.int64, copy=False)


def _apply_query_order(
    *,
    qry_xyz: np.ndarray,
    answer_code: np.ndarray,
    qry_src_code: np.ndarray,
    rng: Any,
    query_order: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(qry_xyz.shape[0]) <= 1:
        return qry_xyz, answer_code, qry_src_code
    if query_order == "sampled":
        return qry_xyz, answer_code, qry_src_code
    if query_order == "shuffled":
        perm = rng.permutation(int(qry_xyz.shape[0])).astype(np.int64)
    elif query_order == "ordered_xyz":
        perm = _ordered_xyz_perm(qry_xyz)
    else:
        raise KeyError(f"unknown query_order={query_order}")
    return qry_xyz[perm], answer_code[perm], qry_src_code[perm]


_QUERY_SRC_NAME_TO_CODE = {
    "uniform": 0,
    "near": 1,
    "near_surface": 1,
}

QUERY_SRC_CODE_TO_NAME = {
    -1: "na",
    0: "uniform",
    1: "near_surface",
}


def _normalize_query_src_filter(v: Any) -> tuple[int, ...] | None:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        items = list(v)
    else:
        text = str(v).strip()
        if not text:
            return None
        items = [x.strip() for x in text.split(",") if x.strip()]
    codes: list[int] = []
    for item in items:
        if isinstance(item, (int, np.integer)):
            codes.append(int(item))
            continue
        key = str(item).strip().lower()
        if key not in _QUERY_SRC_NAME_TO_CODE:
            raise KeyError(f"unknown query_src_filter item={item}")
        codes.append(int(_QUERY_SRC_NAME_TO_CODE[key]))
    if not codes:
        return None
    return tuple(sorted(set(codes)))


@dataclass(frozen=True)
class CQATaskSpec:
    name: str
    query_name: str
    query_type: int
    query_xyz_key: str
    field_keys: Dict[str, str]
    encode_mode: str | None = None
    query_pool_mode: str | None = None


TASK_REGISTRY: Dict[str, CQATaskSpec] = {
    # Surface-aligned tasks always use surf_xyz as carrier.
    "mesh_normal": CQATaskSpec(
        name="mesh_normal",
        query_name="mesh_normal",
        query_type=ASK_NORMAL,
        query_xyz_key="surf_xyz",
        field_keys={"normal": "mesh_surf_n"},
    ),
    "mesh_normal_unsigned": CQATaskSpec(
        name="mesh_normal_unsigned",
        query_name="mesh_normal_unsigned",
        query_type=ASK_NORMAL,
        query_xyz_key="surf_xyz",
        field_keys={"normal": "mesh_surf_n"},
        encode_mode="normal_unsigned",
    ),
    "mesh_visibility": CQATaskSpec(
        name="mesh_visibility",
        query_name="mesh_visibility",
        query_type=ASK_VISIBILITY,
        query_xyz_key="surf_xyz",
        field_keys={"visibility": "mesh_surf_vis_sig"},
    ),
    "mesh_curvature": CQATaskSpec(
        name="mesh_curvature",
        query_name="mesh_curvature",
        query_type=ASK_CURVATURE,
        query_xyz_key="surf_xyz",
        field_keys={"curvature": "mesh_surf_curv"},
    ),
    "mesh_viscount": CQATaskSpec(
        name="mesh_viscount",
        query_name="mesh_visibility",
        query_type=ASK_VISIBILITY,
        query_xyz_key="surf_xyz",
        field_keys={"viscount": "mesh_surf_viscount"},
        encode_mode="mesh_viscount",
    ),
    "mesh_ao": CQATaskSpec(
        name="mesh_ao",
        query_name="mesh_ao",
        query_type=ASK_VISIBILITY,
        query_xyz_key="surf_xyz",
        field_keys={"ao": "mesh_surf_ao"},
        encode_mode="mesh_ao",
    ),
    "mesh_ao_hq": CQATaskSpec(
        name="mesh_ao_hq",
        query_name="mesh_ao",
        query_type=ASK_VISIBILITY,
        query_xyz_key="surf_xyz",
        field_keys={"ao": "mesh_surf_ao_hq"},
        encode_mode="mesh_ao",
    ),
    "udf_thickness": CQATaskSpec(
        name="udf_thickness",
        query_name="udf_thickness",
        query_type=ASK_THICKNESS,
        query_xyz_key="surf_xyz",
        field_keys={"thickness": "udf_surf_thickness"},
    ),
    "udf_thickness_valid_qbin": CQATaskSpec(
        name="udf_thickness_valid_qbin",
        query_name="udf_thickness_valid_qbin",
        query_type=ASK_THICKNESS,
        query_xyz_key="surf_xyz",
        field_keys={"thickness": "udf_surf_thickness"},
        encode_mode="udf_thickness_valid_qbin",
        query_pool_mode="udf_thickness_valid_qbin",
    ),
    "udf_clearance": CQATaskSpec(
        name="udf_clearance",
        query_name="udf_clearance",
        query_type=ASK_CLEARANCE,
        query_xyz_key="surf_xyz",
        # Current task semantics are front-clearance only.
        field_keys={"clearance": "udf_surf_clear_front"},
    ),
    # Explicit off-surface query task.
    "udf_distance": CQATaskSpec(
        name="udf_distance",
        query_name="udf_distance",
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
        query_src_filter: Any = None,
        query_dist_min: float | None = None,
        query_dist_max: float | None = None,
        query_order: str | None = None,
        codec_version: str = CQA_VOCAB_VERSION,
        task_id: int = 0,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        if task_name not in TASK_REGISTRY:
            raise KeyError(f"unknown task_name={task_name}")
        self.task = TASK_REGISTRY[task_name]
        self.codec_version = str(codec_version or CQA_VOCAB_VERSION)
        qname_to_id = query_name_to_id(self.codec_version)
        if self.task.query_name not in qname_to_id:
            raise KeyError(
                f"task={task_name} query_name={self.task.query_name} is not supported "
                f"under codec_version={self.codec_version}"
            )
        self.query_type = int(qname_to_id[self.task.query_name])
        self.context_source = str(context_source)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)
        self.query_src_filter = _normalize_query_src_filter(query_src_filter)
        self.query_dist_min = None if query_dist_min is None else float(query_dist_min)
        self.query_dist_max = None if query_dist_max is None else float(query_dist_max)
        self.query_order = _resolve_query_order(query_order, mode=self.mode)
        self.task_id = int(task_id)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + idx)
        with np.load(path, allow_pickle=False) as npz:
            ctx_xyz, ctx_bank_idx = _sample_context_xyz(
                npz=npz,
                context_source=self.context_source,
                n_ctx=self.n_ctx,
                rng=rng,
                mode=self.mode,
                fallback_seed=self.seed + idx,
            )
            return _build_single_task_sample(
                npz=npz,
                path=path,
                task=self.task,
                query_type=int(self.query_type),
                context_source=self.context_source,
                ctx_xyz=ctx_xyz,
                ctx_bank_idx=ctx_bank_idx,
                n_qry=self.n_qry,
                rng=rng,
                query_src_filter=self.query_src_filter,
                query_dist_min=self.query_dist_min,
                query_dist_max=self.query_dist_max,
                query_order=self.query_order,
                codec_version=self.codec_version,
                task_id=int(self.task_id),
                shape_id=int(idx),
            )


def _sample_context_xyz(
    *,
    npz: Any,
    context_source: str,
    n_ctx: int,
    rng: Any,
    mode: str,
    fallback_seed: int,
) -> tuple[np.ndarray, int | None]:
    if context_source == "surf":
        ctx_all = np.asarray(npz["surf_xyz"], dtype=np.float32)
        ctx_bank_idx = None
    elif context_source == "pc_bank":
        ctx_all, ctx_bank_idx = _select_optional_bank(
            np.asarray(npz["pc_ctx_bank_xyz"], dtype=np.float32),
            rng,
            mode,
            fallback_seed=fallback_seed,
        )
    else:
        raise ValueError(f"unknown context_source={context_source}")
    ctx_idx = _choice(int(ctx_all.shape[0]), int(n_ctx), rng)
    return np.asarray(ctx_all[ctx_idx], dtype=np.float32), None if ctx_bank_idx is None else int(ctx_bank_idx)


def _build_single_task_sample(
    *,
    npz: Any,
    path: str,
    task: CQATaskSpec,
    query_type: int,
    context_source: str,
    ctx_xyz: np.ndarray,
    ctx_bank_idx: int | None,
    n_qry: int,
    rng: Any,
    query_src_filter: tuple[int, ...] | None,
    query_dist_min: float | None,
    query_dist_max: float | None,
    query_order: str,
    codec_version: str,
    task_id: int = 0,
    shape_id: int = -1,
) -> Dict[str, Any]:
    qry_all = np.asarray(npz[task.query_xyz_key], dtype=np.float32)
    q_pool = np.arange(int(qry_all.shape[0]), dtype=np.int64)
    qry_src_code_all = None
    if task.query_xyz_key == "udf_qry_xyz":
        qry_src_code_all = np.asarray(npz["udf_qry_src_code"], dtype=np.int64).reshape(-1)
        if query_src_filter is not None:
            q_pool = q_pool[np.isin(qry_src_code_all, np.asarray(query_src_filter, dtype=np.int64))]
        if query_dist_min is not None or query_dist_max is not None:
            dist = np.asarray(npz["udf_qry_dist"], dtype=np.float32).reshape(-1)
            keep = np.ones((int(dist.shape[0]),), dtype=np.bool_)
            if query_dist_min is not None:
                keep &= dist >= float(query_dist_min)
            if query_dist_max is not None:
                keep &= dist <= float(query_dist_max)
            q_pool = q_pool[keep[q_pool]]
    if task.query_pool_mode == "udf_thickness_valid_qbin":
        thick = np.asarray(npz["udf_surf_thickness"], dtype=np.float32).reshape(-1)
        hit_out = np.asarray(npz["udf_surf_hit_out"], dtype=np.float32).reshape(-1)
        t_in = np.asarray(npz["udf_surf_t_in"], dtype=np.float32).reshape(-1)
        t_out = np.asarray(npz["udf_surf_t_out"], dtype=np.float32).reshape(-1)
        eps = np.float32(1e-4)
        max_t = np.float32(1.999)
        keep = (
            (hit_out > np.float32(0.5))
            & (thick > eps)
            & (t_in > eps)
            & (t_out > eps)
            & (t_in < max_t)
            & (t_out < max_t)
        )
        q_pool = q_pool[keep[q_pool]]
        if int(q_pool.shape[0]) <= 0:
            relaxed = (
                (hit_out > np.float32(0.5))
                & (thick > eps)
                & (t_in > eps)
                & (t_out > eps)
            )
            q_pool = np.arange(int(qry_all.shape[0]), dtype=np.int64)[relaxed]
    if int(q_pool.shape[0]) <= 0:
        raise RuntimeError(
            f"empty query pool after filtering: task={task.name} path={path} "
            f"query_src_filter={query_src_filter} query_dist_min={query_dist_min} "
            f"query_dist_max={query_dist_max}"
        )
    if int(n_qry) > 0:
        if int(q_pool.shape[0]) >= int(n_qry):
            take = _choice(int(q_pool.shape[0]), int(n_qry), rng)
        else:
            take = _choice_with_replacement_if_needed(int(q_pool.shape[0]), int(n_qry), rng)
        q_idx = q_pool[take]
    else:
        q_idx = q_pool
    qry_xyz = np.asarray(qry_all[q_idx], dtype=np.float32)
    if qry_src_code_all is None:
        qry_src_code = np.full((int(q_idx.shape[0]),), -1, dtype=np.int64)
    else:
        qry_src_code = np.asarray(qry_src_code_all[q_idx], dtype=np.int64)

    fields: Dict[str, np.ndarray] = {}
    for alias, key in task.field_keys.items():
        arr = np.asarray(npz[key], dtype=np.float32)
        fields[alias] = arr[q_idx]
    answer_code = encode_answers_from_fields(
        task.query_name,
        fields,
        codec_version=str(codec_version),
        encode_mode=task.encode_mode,
    )
    qry_xyz, answer_code, qry_src_code = _apply_query_order(
        qry_xyz=qry_xyz,
        answer_code=answer_code,
        qry_src_code=qry_src_code,
        rng=rng,
        query_order=query_order,
    )

    path_obj = Path(path)
    return {
        "ctx_xyz": _np_to_torch_f32(ctx_xyz),
        "qry_xyz": _np_to_torch_f32(qry_xyz),
        "qry_type": torch.full((int(qry_xyz.shape[0]),), int(query_type), dtype=torch.long),
        "answer_code": _np_to_torch_i64(answer_code),
        "qry_src_code": _np_to_torch_i64(qry_src_code),
        "task_name": task.name,
        "context_source": context_source,
        "cache_split": path_obj.parent.parent.name,
        "synset": path_obj.parent.name,
        "path": path,
        "context_bank_idx": ctx_bank_idx,
        "query_src_filter": None if query_src_filter is None else list(query_src_filter),
        "query_dist_min": query_dist_min,
        "query_dist_max": query_dist_max,
        "query_order": query_order,
        "task_id": int(task_id),
        "shape_id": int(shape_id),
        "answer_vocab_size": int(answer_vocab_size(str(codec_version))),
        "vocab_version": str(codec_version),
    }


@dataclass(frozen=True)
class PackedCQATaskSpec:
    task_name: str
    context_source: str = "surf"
    query_src_filter: tuple[int, ...] | None = None
    query_dist_min: float | None = None
    query_dist_max: float | None = None


class PackedPrimitiveCQADataset(Dataset):
    """Packed multi-type CQA samples sharing one context per shape."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        task_specs: Sequence[PackedCQATaskSpec],
        n_ctx: int = 2048,
        n_qry: int = 64,
        seed: int = 0,
        mode: str = "train",
        query_order: str | None = None,
        codec_version: str = CQA_VOCAB_VERSION,
    ) -> None:
        super().__init__()
        if len(task_specs) <= 0:
            raise ValueError("PackedPrimitiveCQADataset requires at least one task spec")
        self.paths = list(paths)
        self.task_specs = list(task_specs)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)
        self.query_order = _resolve_query_order(query_order, mode=self.mode)
        self.codec_version = str(codec_version or CQA_VOCAB_VERSION)
        self._resolved_specs: list[tuple[PackedCQATaskSpec, CQATaskSpec, int]] = []
        qname_to_id = query_name_to_id(self.codec_version)
        for packed in self.task_specs:
            if packed.task_name not in TASK_REGISTRY:
                raise KeyError(f"unknown packed task_name={packed.task_name}")
            task = TASK_REGISTRY[packed.task_name]
            if task.query_name not in qname_to_id:
                raise KeyError(
                    f"task={packed.task_name} query_name={task.query_name} is not supported "
                    f"under codec_version={self.codec_version}"
                )
            self._resolved_specs.append((packed, task, int(qname_to_id[task.query_name])))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + idx)
        first_context_source = str(self._resolved_specs[0][0].context_source)
        with np.load(path, allow_pickle=False) as npz:
            ctx_xyz, ctx_bank_idx = _sample_context_xyz(
                npz=npz,
                context_source=first_context_source,
                n_ctx=self.n_ctx,
                rng=rng,
                mode=self.mode,
                fallback_seed=self.seed + idx,
            )
            rows = [
                _build_single_task_sample(
                    npz=npz,
                    path=path,
                    task=task,
                    query_type=query_type,
                    context_source=str(packed.context_source),
                    ctx_xyz=ctx_xyz,
                    ctx_bank_idx=ctx_bank_idx,
                    n_qry=self.n_qry,
                    rng=rng,
                    query_src_filter=packed.query_src_filter,
                    query_dist_min=packed.query_dist_min,
                    query_dist_max=packed.query_dist_max,
                    query_order=self.query_order,
                    codec_version=self.codec_version,
                    task_id=int(task_idx),
                    shape_id=int(idx),
                )
                for task_idx, (packed, task, query_type) in enumerate(self._resolved_specs)
            ]
        out: Dict[str, Any] = {}
        out["ctx_xyz"] = torch.stack([r["ctx_xyz"] for r in rows], dim=0)
        out["qry_xyz"] = torch.stack([r["qry_xyz"] for r in rows], dim=0)
        out["qry_type"] = torch.stack([r["qry_type"] for r in rows], dim=0)
        out["answer_code"] = torch.stack([r["answer_code"] for r in rows], dim=0)
        out["qry_src_code"] = torch.stack([r["qry_src_code"] for r in rows], dim=0)
        out["task_id"] = torch.tensor([int(r["task_id"]) for r in rows], dtype=torch.long)
        out["shape_id"] = torch.tensor([int(r["shape_id"]) for r in rows], dtype=torch.long)
        out["task_name"] = [r["task_name"] for r in rows]
        out["context_source"] = [r["context_source"] for r in rows]
        out["cache_split"] = [r["cache_split"] for r in rows]
        out["synset"] = [r["synset"] for r in rows]
        out["path"] = [r["path"] for r in rows]
        out["context_bank_idx"] = [r.get("context_bank_idx") for r in rows]
        out["query_src_filter"] = [r.get("query_src_filter") for r in rows]
        out["query_dist_min"] = [r.get("query_dist_min") for r in rows]
        out["query_dist_max"] = [r.get("query_dist_max") for r in rows]
        out["query_order"] = [r.get("query_order") for r in rows]
        out["answer_vocab_size"] = int(answer_vocab_size(self.codec_version))
        out["vocab_version"] = self.codec_version
        out["packed_task_names"] = [str(spec.task_name) for spec in self.task_specs]
        return out


def cqa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ctx_xyz"] = torch.stack([b["ctx_xyz"] for b in batch], dim=0)
    out["qry_xyz"] = torch.stack([b["qry_xyz"] for b in batch], dim=0)
    out["qry_type"] = torch.stack([b["qry_type"] for b in batch], dim=0)
    out["answer_code"] = torch.stack([b["answer_code"] for b in batch], dim=0)
    out["qry_src_code"] = torch.stack([b["qry_src_code"] for b in batch], dim=0)
    out["task_id"] = torch.tensor([int(b.get("task_id", 0)) for b in batch], dtype=torch.long)
    out["shape_id"] = torch.tensor([int(b.get("shape_id", -1)) for b in batch], dtype=torch.long)
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
    out["answer_vocab_size"] = int(batch[0].get("answer_vocab_size", ANSWER_VOCAB_SIZE))
    out["vocab_version"] = str(batch[0].get("vocab_version", CQA_VOCAB_VERSION))
    return out


def cqa_packed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["ctx_xyz"] = torch.cat([b["ctx_xyz"] for b in batch], dim=0)
    out["qry_xyz"] = torch.cat([b["qry_xyz"] for b in batch], dim=0)
    out["qry_type"] = torch.cat([b["qry_type"] for b in batch], dim=0)
    out["answer_code"] = torch.cat([b["answer_code"] for b in batch], dim=0)
    out["qry_src_code"] = torch.cat([b["qry_src_code"] for b in batch], dim=0)
    out["task_id"] = torch.cat([b["task_id"] for b in batch], dim=0)
    out["shape_id"] = torch.cat([b["shape_id"] for b in batch], dim=0)
    out["task_name"] = [name for b in batch for name in b["task_name"]]
    out["context_source"] = [name for b in batch for name in b["context_source"]]
    out["cache_split"] = [name for b in batch for name in b["cache_split"]]
    out["synset"] = [name for b in batch for name in b["synset"]]
    out["path"] = [name for b in batch for name in b["path"]]
    out["context_bank_idx"] = [name for b in batch for name in b.get("context_bank_idx", [])]
    out["query_src_filter"] = [name for b in batch for name in b.get("query_src_filter", [])]
    out["query_dist_min"] = [name for b in batch for name in b.get("query_dist_min", [])]
    out["query_dist_max"] = [name for b in batch for name in b.get("query_dist_max", [])]
    out["query_order"] = [name for b in batch for name in b.get("query_order", [])]
    out["answer_vocab_size"] = int(batch[0].get("answer_vocab_size", ANSWER_VOCAB_SIZE))
    out["vocab_version"] = str(batch[0].get("vocab_version", CQA_VOCAB_VERSION))
    out["packed_task_names"] = [name for b in batch for name in b.get("packed_task_names", [])]
    return out
