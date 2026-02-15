from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..backends.mesh_backend import MeshBackend
from ..backends.pointcloud_backend import (
    PointCloudBackend,
    PointCloudMeshRayBackend,
    PointCloudNoRayBackend,
)
from ..backends.udfgrid_backend import UDFGridBackend
from ..backends.voxel_backend import VoxelBackend
from ..token.ordering import morton3d
from .mixed_pretrain import MixtureSampler, MixedPretrainDataset, load_mix_config
from .modelnet40_index import list_npz


def _stable_seed(path: str, seed: int) -> int:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
    return (int(h, 16) + int(seed)) & 0xFFFFFFFF


def _make_backend(name: str, path: str, voxel_grid: int, voxel_dilate: int, voxel_max_steps: int):
    if name == "mesh":
        return MeshBackend(path)
    if name == "pointcloud":
        return PointCloudBackend(path)
    if name == "pointcloud_meshray":
        return PointCloudMeshRayBackend(path)
    if name == "pointcloud_noray":
        return PointCloudNoRayBackend(path)
    if name == "voxel":
        return VoxelBackend(path, grid=voxel_grid, dilate=voxel_dilate, max_steps=voxel_max_steps)
    if name == "udfgrid":
        return UDFGridBackend(path)
    raise ValueError(f"unknown backend: {name}")


def _sample_indices(n_pool: int, n_context: int, n_query: int, rng: np.random.RandomState, disjoint: bool):
    total = int(n_context) + int(n_query)
    if total <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    if bool(disjoint):
        replace = n_pool < total
        idx = rng.choice(n_pool, size=total, replace=replace).astype(np.int64)
        return idx[: int(n_context)], idx[int(n_context):]
    cidx = rng.choice(n_pool, size=int(n_context), replace=n_pool < int(n_context)).astype(np.int64)
    qidx = rng.choice(n_pool, size=int(n_query), replace=n_pool < int(n_query)).astype(np.int64)
    return cidx, qidx


def _sample_udf_grid_queries(d: np.lib.npyio.NpzFile, rng: np.random.RandomState, n_query: int):
    if "udf_grid" not in d:
        raise RuntimeError("query_source=grid requires udf_grid in npz")
    udf = d["udf_grid"].astype(np.float32, copy=False)
    if udf.ndim != 3 or (udf.shape[0] != udf.shape[1]) or (udf.shape[0] != udf.shape[2]):
        raise RuntimeError(f"udf_grid must be cubic, got shape={udf.shape}")
    g = int(udf.shape[0])
    total = g * g * g
    n = int(min(int(n_query), total))
    lin = rng.choice(total, size=n, replace=False).astype(np.int64)
    i = lin // (g * g)
    j = (lin // g) % g
    k = lin % g
    step = np.float32(2.0 / float(g))
    xyz = np.stack(
        [
            np.float32(-1.0) + (i.astype(np.float32) + np.float32(0.5)) * step,
            np.float32(-1.0) + (j.astype(np.float32) + np.float32(0.5)) * step,
            np.float32(-1.0) + (k.astype(np.float32) + np.float32(0.5)) * step,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    dist = udf.reshape(-1)[lin].astype(np.float32, copy=False)
    return xyz, dist


class KPlaneContextQueryDataset(Dataset):
    """Context/query regression dataset over cached primitive pools.

    target_mode:
      - backend: query target uses selected backend distance pool (pt_dist_pool)
      - udf    : query target uses pt_dist_udf_pool (or pt_dist_pool fallback)
    """

    def __init__(
        self,
        npz_paths: List[str],
        backend: str,
        n_context: int = 256,
        n_query: int = 256,
        mode: str = "train",
        eval_seed: int = 0,
        disjoint_context_query: bool = True,
        query_source: str = "pool",
        target_mode: str = "backend",
        voxel_grid: int = 64,
        voxel_dilate: int = 1,
        voxel_max_steps: int = 0,
    ):
        self.paths = list(npz_paths)
        self.backend = str(backend)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.mode = str(mode)
        self.eval_seed = int(eval_seed)
        self.disjoint_context_query = bool(disjoint_context_query)
        self.query_source = str(query_source)
        self.target_mode = str(target_mode)
        self.voxel_grid = int(voxel_grid)
        self.voxel_dilate = int(voxel_dilate)
        self.voxel_max_steps = int(voxel_max_steps)
        if self.query_source not in ("pool", "grid"):
            raise ValueError(f"unknown query_source: {self.query_source}")
        if self.target_mode not in ("backend", "udf"):
            raise ValueError(f"unknown target_mode: {self.target_mode}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[int(idx)]
        be = _make_backend(
            self.backend,
            path,
            voxel_grid=self.voxel_grid,
            voxel_dilate=self.voxel_dilate,
            voxel_max_steps=self.voxel_max_steps,
        )
        pools = be.get_pools()
        d = np.load(path, allow_pickle=False)

        if self.mode == "eval":
            rng = np.random.RandomState(_stable_seed(path, self.eval_seed))
        else:
            rng = np.random

        pt_xyz_pool = pools["pt_xyz_pool"].astype(np.float32, copy=False)
        n_pool = int(pt_xyz_pool.shape[0])
        cidx, qidx = _sample_indices(
            n_pool=n_pool,
            n_context=self.n_context,
            n_query=self.n_query,
            rng=rng,
            disjoint=self.disjoint_context_query,
        )
        ctx_xyz = pt_xyz_pool[cidx].astype(np.float32, copy=False)
        ctx_dist = pools["pt_dist_pool"].astype(np.float32, copy=False)[cidx]

        if self.query_source == "pool":
            qry_xyz = pt_xyz_pool[qidx].astype(np.float32, copy=False)
            if self.target_mode == "udf" and ("pt_dist_udf_pool" in d):
                qry_dist = d["pt_dist_udf_pool"].astype(np.float32, copy=False)[qidx]
            else:
                qry_dist = pools["pt_dist_pool"].astype(np.float32, copy=False)[qidx]
        else:
            qry_xyz, qry_dist = _sample_udf_grid_queries(d, rng=rng, n_query=self.n_query)

        # Keep deterministic ordering for stable training/eval traces.
        if ctx_xyz.shape[0] > 0:
            order = np.argsort(morton3d(ctx_xyz))
            ctx_xyz = ctx_xyz[order]
            ctx_dist = ctx_dist[order]
        if qry_xyz.shape[0] > 0:
            order = np.argsort(morton3d(qry_xyz))
            qry_xyz = qry_xyz[order]
            qry_dist = qry_dist[order]

        return {
            "ctx_xyz": torch.from_numpy(ctx_xyz),
            "ctx_dist": torch.from_numpy(ctx_dist),
            "qry_xyz": torch.from_numpy(qry_xyz),
            "qry_dist": torch.from_numpy(qry_dist),
            "path": path,
            "backend": self.backend,
        }


def collate_kplane(batch: List[Dict[str, Any]]):
    return {
        "ctx_xyz": torch.stack([b["ctx_xyz"] for b in batch], dim=0),
        "ctx_dist": torch.stack([b["ctx_dist"] for b in batch], dim=0),
        "qry_xyz": torch.stack([b["qry_xyz"] for b in batch], dim=0),
        "qry_dist": torch.stack([b["qry_dist"] for b in batch], dim=0),
        "path": [b["path"] for b in batch],
        "backend": [b["backend"] for b in batch],
    }


def build_kplane_mixed_pretrain(
    mix_config_path: str,
    n_context: int,
    n_query: int,
    mode: str = "train",
    eval_seed: int = 0,
    disjoint_context_query: bool = True,
    query_source: str = "pool",
    target_mode: str = "backend",
    voxel_grid: int = 64,
    voxel_dilate: int = 1,
    voxel_max_steps: int = 0,
) -> Tuple[MixedPretrainDataset, MixtureSampler, Dict[str, Any]]:
    specs, cfg = load_mix_config(mix_config_path)

    datasets: List[Dataset] = []
    names: List[str] = []
    weights: List[float] = []
    for s in specs:
        paths = list_npz(s.cache_root, s.split)
        if len(paths) == 0:
            raise FileNotFoundError(f"no npz found: cache_root={s.cache_root} split={s.split}")
        ds = KPlaneContextQueryDataset(
            npz_paths=paths,
            backend=s.backend,
            n_context=int(n_context),
            n_query=int(n_query),
            mode=mode,
            eval_seed=int(eval_seed),
            disjoint_context_query=bool(disjoint_context_query),
            query_source=str(query_source),
            target_mode=str(target_mode),
            voxel_grid=int(voxel_grid if s.voxel_grid is None else s.voxel_grid),
            voxel_dilate=int(voxel_dilate if s.voxel_dilate is None else s.voxel_dilate),
            voxel_max_steps=int(voxel_max_steps if s.voxel_max_steps is None else s.voxel_max_steps),
        )
        datasets.append(ds)
        names.append(s.name)
        weights.append(float(s.weight))

    mixed = MixedPretrainDataset(datasets, names)
    num_samples = int(cfg.get("mix_num_samples", len(mixed)))
    replacement = bool(cfg.get("replacement", True))
    seed = int(cfg.get("mix_seed", 0))
    sampler = MixtureSampler(
        dataset_sizes=mixed.sizes,
        dataset_weights=weights,
        num_samples=num_samples,
        replacement=replacement,
        seed=seed,
    )
    info = {
        "names": names,
        "weights": weights,
        "sizes": mixed.sizes,
        "num_samples": num_samples,
        "replacement": replacement,
        "seed": seed,
    }
    return mixed, sampler, info


def build_kplane_loader(
    mix_config_path: str,
    batch_size: int,
    num_workers: int,
    n_context: int,
    n_query: int,
    mode: str = "train",
    eval_seed: int = 0,
    disjoint_context_query: bool = True,
    query_source: str = "pool",
    target_mode: str = "backend",
    voxel_grid: int = 64,
    voxel_dilate: int = 1,
    voxel_max_steps: int = 0,
):
    ds, sampler, info = build_kplane_mixed_pretrain(
        mix_config_path=mix_config_path,
        n_context=n_context,
        n_query=n_query,
        mode=mode,
        eval_seed=eval_seed,
        disjoint_context_query=disjoint_context_query,
        query_source=query_source,
        target_mode=target_mode,
        voxel_grid=voxel_grid,
        voxel_dilate=voxel_dilate,
        voxel_max_steps=voxel_max_steps,
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_kplane,
    )
    return dl, info
