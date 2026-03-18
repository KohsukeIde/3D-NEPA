from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from nepa3d.data.mixed_pretrain import MixtureSampler, MixedPretrainDataset, load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import (
    QUERY_SRC_CODE_TO_NAME,
    _normalize_query_src_filter,
    _select_optional_bank,
)


def _choice(n: int, k: int, rng: Any) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def _sample_eval_paths(paths: List[str], *, max_samples: int, seed: int, mode: str) -> List[str]:
    if int(max_samples) <= 0 or len(paths) <= int(max_samples):
        return list(paths)
    mode = str(mode)
    if mode == "head":
        return list(paths[: int(max_samples)])
    if mode == "random":
        rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        take = rng.choice(len(paths), size=int(max_samples), replace=False)
        return [paths[int(i)] for i in take.tolist()]
    raise KeyError(f"unknown eval_sample_mode={mode}")


def _stable_int_seed(text: str) -> int:
    x = 0
    for ch in str(text):
        x = (x * 131 + ord(ch)) % 2147483647
    return int(x)


@dataclass(frozen=True)
class WorldV3EvalSpec:
    name: str
    split: str
    cache_root: str
    context_source: str
    eval_sample_mode: str
    dataset: "WorldV3UDFDistanceDataset"


class WorldV3UDFDistanceDataset(Dataset):
    """Current world_v3-aware continuous udf_distance dataset for k-plane/tri-plane baselines.

    This intentionally mirrors the CQA udf_distance task semantics:
      - context carrier: surf_xyz or pc_ctx_bank_xyz
      - query carrier   : udf_qry_xyz
      - target          : continuous udf_qry_dist

    The baseline sees only geometry in the context carrier (ctx_dist is zero-filled).
    This keeps the comparison honest: the baseline does not receive a privileged
    scalar field on context points that the CQA model never sees.
    """

    def __init__(
        self,
        paths: Sequence[str],
        *,
        context_source: str = "surf",
        n_ctx: int = 2048,
        n_qry: int = 64,
        seed: int = 0,
        mode: str = "train",
        query_src_filter: Any = None,
        query_dist_min: float | None = None,
        query_dist_max: float | None = None,
    ) -> None:
        super().__init__()
        self.paths = list(paths)
        self.context_source = str(context_source)
        self.n_ctx = int(n_ctx)
        self.n_qry = int(n_qry)
        self.seed = int(seed)
        self.mode = str(mode)
        self.query_src_filter = _normalize_query_src_filter(query_src_filter)
        self.query_dist_min = None if query_dist_min is None else float(query_dist_min)
        self.query_dist_max = None if query_dist_max is None else float(query_dist_max)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[int(idx)]
        rng = np.random if self.mode == "train" else np.random.RandomState(self.seed + int(idx))
        with np.load(path, allow_pickle=False) as npz:
            if self.context_source == "surf":
                ctx_all = np.asarray(npz["surf_xyz"], dtype=np.float32)
                ctx_bank_idx = None
            elif self.context_source == "pc_bank":
                ctx_all, ctx_bank_idx = _select_optional_bank(
                    np.asarray(npz["pc_ctx_bank_xyz"], dtype=np.float32),
                    rng,
                    self.mode,
                    fallback_seed=self.seed + int(idx),
                )
            else:
                raise ValueError(f"unknown context_source={self.context_source}")

            ctx_idx = _choice(int(ctx_all.shape[0]), self.n_ctx, rng)
            ctx_xyz = np.asarray(ctx_all[ctx_idx], dtype=np.float32)
            # Deliberately zero-filled for a geometry-only baseline.
            ctx_dist = np.zeros((int(ctx_xyz.shape[0]),), dtype=np.float32)

            qry_xyz_all = np.asarray(npz["udf_qry_xyz"], dtype=np.float32)
            qry_dist_all = np.asarray(npz["udf_qry_dist"], dtype=np.float32).reshape(-1)
            if "udf_qry_src_code" in npz:
                qry_src_code_all = np.asarray(npz["udf_qry_src_code"], dtype=np.int64).reshape(-1)
            else:
                qry_src_code_all = np.full((int(qry_xyz_all.shape[0]),), -1, dtype=np.int64)

            q_pool = np.arange(int(qry_xyz_all.shape[0]), dtype=np.int64)
            if self.query_src_filter is not None:
                q_pool = q_pool[np.isin(qry_src_code_all, np.asarray(self.query_src_filter, dtype=np.int64))]
            if self.query_dist_min is not None or self.query_dist_max is not None:
                keep = np.ones((int(qry_dist_all.shape[0]),), dtype=np.bool_)
                if self.query_dist_min is not None:
                    keep &= qry_dist_all >= float(self.query_dist_min)
                if self.query_dist_max is not None:
                    keep &= qry_dist_all <= float(self.query_dist_max)
                q_pool = q_pool[keep[q_pool]]
            if int(q_pool.shape[0]) <= 0:
                raise RuntimeError(
                    f"empty query pool after filtering: path={path} "
                    f"query_src_filter={self.query_src_filter} query_dist_min={self.query_dist_min} "
                    f"query_dist_max={self.query_dist_max}"
                )
            if self.n_qry >= int(q_pool.shape[0]):
                q_idx = q_pool
            else:
                take = _choice(int(q_pool.shape[0]), self.n_qry, rng)
                q_idx = q_pool[take]
            qry_xyz = np.asarray(qry_xyz_all[q_idx], dtype=np.float32)
            qry_dist = np.asarray(qry_dist_all[q_idx], dtype=np.float32)
            qry_src_code = np.asarray(qry_src_code_all[q_idx], dtype=np.int64)

            if self.mode == "train":
                perm = rng.permutation(int(qry_xyz.shape[0])).astype(np.int64)
                qry_xyz = qry_xyz[perm]
                qry_dist = qry_dist[perm]
                qry_src_code = qry_src_code[perm]

            path_obj = Path(path)
            return {
                "ctx_xyz": torch.from_numpy(ctx_xyz.astype(np.float32, copy=False)),
                "ctx_dist": torch.from_numpy(ctx_dist.astype(np.float32, copy=False)),
                "qry_xyz": torch.from_numpy(qry_xyz.astype(np.float32, copy=False)),
                "qry_dist": torch.from_numpy(qry_dist.astype(np.float32, copy=False)),
                "qry_src_code": torch.from_numpy(qry_src_code.astype(np.int64, copy=False)),
                "task_name": "udf_distance",
                "context_source": self.context_source,
                "cache_split": path_obj.parent.parent.name,
                "synset": path_obj.parent.name,
                "path": path,
                "context_bank_idx": None if ctx_bank_idx is None else int(ctx_bank_idx),
                "query_src_filter": None if self.query_src_filter is None else list(self.query_src_filter),
                "query_dist_min": self.query_dist_min,
                "query_dist_max": self.query_dist_max,
            }


def collate_udfdist_worldv3(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ctx_xyz": torch.stack([b["ctx_xyz"] for b in batch], dim=0),
        "ctx_dist": torch.stack([b["ctx_dist"] for b in batch], dim=0),
        "qry_xyz": torch.stack([b["qry_xyz"] for b in batch], dim=0),
        "qry_dist": torch.stack([b["qry_dist"] for b in batch], dim=0),
        "qry_src_code": torch.stack([b["qry_src_code"] for b in batch], dim=0),
        "task_name": [b["task_name"] for b in batch],
        "context_source": [b["context_source"] for b in batch],
        "cache_split": [b["cache_split"] for b in batch],
        "synset": [b["synset"] for b in batch],
        "path": [b["path"] for b in batch],
        "context_bank_idx": [b.get("context_bank_idx") for b in batch],
        "query_src_filter": [b.get("query_src_filter") for b in batch],
        "query_dist_min": [b.get("query_dist_min") for b in batch],
        "query_dist_max": [b.get("query_dist_max") for b in batch],
    }


def build_worldv3_udfdist_mixed(
    mix_config_path: str,
    *,
    n_ctx: int,
    n_qry: int,
    mode: str = "train",
    seed: int = 0,
) -> Tuple[MixedPretrainDataset, MixtureSampler, Dict[str, Any]]:
    specs, cfg = load_mix_config(mix_config_path)
    datasets: List[Dataset] = []
    names: List[str] = []
    weights: List[float] = []
    for s in specs:
        task_name = str(s.extra.get("task_name", s.name))
        if task_name != "udf_distance":
            continue
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")
        ds = WorldV3UDFDistanceDataset(
            paths,
            context_source=str(s.extra.get("context_source", "surf")),
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(seed),
            mode=str(mode),
            query_src_filter=s.extra.get("query_src_filter", None),
            query_dist_min=s.extra.get("query_dist_min", None),
            query_dist_max=s.extra.get("query_dist_max", None),
        )
        datasets.append(ds)
        names.append(str(s.name))
        weights.append(float(s.weight))
    if not datasets:
        raise RuntimeError(f"no udf_distance datasets found in mix config: {mix_config_path}")

    mixed = MixedPretrainDataset(datasets, names)
    num_samples = int(cfg.get("mix_num_samples", len(mixed)))
    replacement = bool(cfg.get("replacement", True))
    mix_seed = int(cfg.get("mix_seed", 0))
    sampler = MixtureSampler(
        dataset_sizes=mixed.sizes,
        dataset_weights=weights,
        num_samples=num_samples,
        replacement=replacement,
        seed=mix_seed,
    )
    info = {
        "names": names,
        "weights": weights,
        "sizes": mixed.sizes,
        "num_samples": num_samples,
        "replacement": replacement,
        "seed": mix_seed,
    }
    return mixed, sampler, info


def build_worldv3_udfdist_loader(
    mix_config_path: str,
    *,
    batch_size: int,
    num_workers: int,
    n_ctx: int,
    n_qry: int,
    mode: str = "train",
    seed: int = 0,
):
    ds, sampler, info = build_worldv3_udfdist_mixed(
        mix_config_path,
        n_ctx=n_ctx,
        n_qry=n_qry,
        mode=mode,
        seed=seed,
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_udfdist_worldv3,
    )
    return dl, info


def build_worldv3_udfdist_eval_specs(
    mix_config_path: str,
    *,
    seed: int,
    n_ctx: int,
    n_qry: int,
    max_samples_per_task: int,
    split_override: str | None,
    eval_sample_mode: str,
) -> List[WorldV3EvalSpec]:
    specs, _cfg = load_mix_config(mix_config_path)
    out: List[WorldV3EvalSpec] = []
    for s in specs:
        task_name = str(s.extra.get("task_name", s.name))
        if task_name != "udf_distance":
            continue
        split = str(split_override or s.split)
        paths = list_npz(s.cache_root, split)
        if int(max_samples_per_task) > 0:
            paths = _sample_eval_paths(
                paths,
                max_samples=int(max_samples_per_task),
                seed=(int(seed) + _stable_int_seed(f"{task_name}:{split}:{s.name}")),
                mode=str(eval_sample_mode),
            )
        ds = WorldV3UDFDistanceDataset(
            paths,
            context_source=str(s.extra.get("context_source", "surf")),
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(seed),
            mode="eval",
            query_src_filter=s.extra.get("query_src_filter", None),
            query_dist_min=s.extra.get("query_dist_min", None),
            query_dist_max=s.extra.get("query_dist_max", None),
        )
        out.append(
            WorldV3EvalSpec(
                name=task_name,
                split=split,
                cache_root=str(s.cache_root),
                context_source=str(s.extra.get("context_source", "surf")),
                eval_sample_mode=str(eval_sample_mode),
                dataset=ds,
            )
        )
    return out
