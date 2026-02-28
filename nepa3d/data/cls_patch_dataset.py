"""Point-cloud dataset for patchified Transformer classification.

This dataset returns *raw* points (and optional normals), not NEPA Q/A token sequences.

It reuses the same cached `.npz` format produced by:
- `nepa3d/data/preprocess_modelnet40.py`
- `nepa3d/data/preprocess_scanobjectnn.py`

Keys used:
- `pc_xyz` (N,3)
- `pc_n` (N,3) optional
- `pc_fps_order` (N,) optional

Augmentation is implemented by calling `_apply_point_aug` from `nepa3d.data.dataset` so
we stay consistent with the existing NEPA classification pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import get_worker_info

from nepa3d.data.dataset import _apply_point_aug


@dataclass
class PointAugConfig:
    prob: float = 0.0
    scale_min: float = 0.9
    scale_max: float = 1.1
    # `_apply_point_aug` expects translate range, so we map shift_std -> translate.
    shift_std: float = 0.02
    jitter_std: float = 0.005
    jitter_clip: float = 0.02
    rot_axis: str = "y"  # x|y|z
    rot_deg: float = 180.0
    dropout_ratio: float = 0.0
    dropout_prob: float = 0.0


class PatchClsPointDataset(Dataset):
    def __init__(
        self,
        npz_paths: List[Path],
        *,
        cache_root: str,
        label_map: Dict[str, int],
        n_point: int = 1024,
        sample_mode: str = "random",  # random | fps
        use_normals: bool = False,
        aug: bool = False,
        aug_cfg: Optional[PointAugConfig] = None,
        rng_seed: int = 0,
        use_ray_patch: bool = False,
        n_ray: int = 256,
        ray_sample_mode: str = "random",  # random | first
        # test-time multi-crop (MC) evaluation
        mc_eval_k: int = 1,
        aug_eval: bool = False,
        deterministic_eval_sampling: bool = True,
    ) -> None:
        super().__init__()
        self.npz_paths = list(npz_paths)
        self.cache_root = str(cache_root)
        self.label_map = dict(label_map)
        self.n_point = int(n_point)
        assert sample_mode in {"random", "fps"}
        self.sample_mode = sample_mode
        self.use_normals = bool(use_normals)
        self.aug = bool(aug)
        self.aug_cfg = aug_cfg or PointAugConfig()
        self.rng_seed = int(rng_seed)
        self.use_ray_patch = bool(use_ray_patch)
        self.n_ray = int(n_ray)
        assert ray_sample_mode in {"random", "first"}
        self.ray_sample_mode = str(ray_sample_mode)
        # Initialized lazily per-worker (important when num_workers>0).
        self._rng: Optional[np.random.RandomState] = None
        self.mc_eval_k = int(mc_eval_k)
        self.aug_eval = bool(aug_eval)
        self.deterministic_eval_sampling = bool(deterministic_eval_sampling)
        self._warned_missing_fps_order = False

    def __len__(self) -> int:
        return len(self.npz_paths)

    def _sample_indices(self, pc_xyz: np.ndarray, npz: dict, *, sample_uid: int = 0) -> np.ndarray:
        n_total = pc_xyz.shape[0]
        n = min(self.n_point, n_total)
        if self.sample_mode == "fps" and "pc_fps_order" in npz:
            idx = npz["pc_fps_order"][:n]
        else:
            if self.sample_mode == "fps" and (not self._warned_missing_fps_order):
                warnings.warn(
                    "PatchClsPointDataset: sample_mode='fps' but pc_fps_order is missing; "
                    "falling back to random subset (results may differ from FPS eval protocol)."
                )
                self._warned_missing_fps_order = True
            # random subset
            if (not self.aug) and self.deterministic_eval_sampling:
                # Make eval crop selection invariant to worker/process order.
                seed = (int(self.rng_seed) * 1315423911 + int(sample_uid) * 2654435761) & 0xFFFFFFFF
                rng = np.random.RandomState(seed)
            else:
                rng = self._get_rng()
            idx = rng.choice(n_total, size=n, replace=False)
        return idx.astype(np.int64, copy=False)

    def _sample_ray_indices(self, n_total: int, *, sample_uid: int = 0) -> np.ndarray:
        n = min(self.n_ray, int(n_total))
        if self.ray_sample_mode == "first":
            return np.arange(n, dtype=np.int64)

        if (not self.aug) and self.deterministic_eval_sampling:
            seed = (int(self.rng_seed) * 2246822519 + int(sample_uid) * 3266489917) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
        else:
            rng = self._get_rng()
        idx = rng.choice(n_total, size=n, replace=False)
        return idx.astype(np.int64, copy=False)

    def _get_rng(self) -> np.random.RandomState:
        if self._rng is None:
            wi = get_worker_info()
            seed = self.rng_seed if wi is None else int(wi.seed)
            seed = int(seed) & 0xFFFFFFFF
            self._rng = np.random.RandomState(seed)
        return self._rng

    def _maybe_augment(
        self, xyz: np.ndarray, normals: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.aug:
            return xyz, normals
        rng = self._get_rng()
        if float(self.aug_cfg.prob) < 1.0 and float(rng.uniform(0.0, 1.0)) >= float(self.aug_cfg.prob):
            return xyz, normals

        # We reuse the same augmentation implementation as token-based datasets.
        # `_apply_point_aug` expects (pt_xyz, pt_dist, pools). We don't use dist here.
        pt_xyz = xyz.astype(np.float32, copy=False)
        pt_dist = np.zeros((pt_xyz.shape[0], 1), dtype=np.float32)

        pools = {}
        if normals is not None:
            pools["pc_n"] = normals.astype(np.float32, copy=False)

        pt_xyz2, _, pools2 = _apply_point_aug(
            pt_xyz,
            pt_dist,
            pools,
            rng=rng,
            rotate_z=(self.aug_cfg.rot_axis == "z") and (float(self.aug_cfg.rot_deg) > 0.0),
            scale_min=float(self.aug_cfg.scale_min),
            scale_max=float(self.aug_cfg.scale_max),
            translate=float(self.aug_cfg.shift_std),
            jitter_sigma=float(self.aug_cfg.jitter_std),
            jitter_clip=float(self.aug_cfg.jitter_clip),
        )

        normals2 = pools2.get("pc_n", None) if normals is not None else None
        return pt_xyz2, normals2

    def __getitem__(self, index: int) -> dict:
        path = self.npz_paths[index]
        # class name is the parent directory under split
        #   .../<split>/<class>/<file>.npz
        cls_name = Path(path).parent.name
        label = self.label_map[cls_name]

        with np.load(path) as npz:
            pc_xyz = npz["pc_xyz"].astype(np.float32, copy=False)
            pc_n = None
            if self.use_normals and "pc_n" in npz:
                pc_n = npz["pc_n"].astype(np.float32, copy=False)

            if self.mc_eval_k > 1:
                # MC evaluation: generate K crops (augmented if aug_eval)
                xyz_list = []
                n_list = [] if pc_n is not None else None
                for k in range(self.mc_eval_k):
                    idx = self._sample_indices(pc_xyz, npz, sample_uid=(index * 10007 + k))
                    xyz = pc_xyz[idx]
                    nn = pc_n[idx] if pc_n is not None else None
                    if self.aug_eval:
                        xyz, nn = self._maybe_augment(xyz, nn)
                    xyz_list.append(xyz)
                    if n_list is not None:
                        n_list.append(nn)
                xyz = np.stack(xyz_list, axis=0)  # (K,N,3)
                normals = np.stack(n_list, axis=0) if n_list is not None else None
            else:
                idx = self._sample_indices(pc_xyz, npz, sample_uid=index)
                xyz = pc_xyz[idx]
                normals = pc_n[idx] if pc_n is not None else None
                xyz, normals = self._maybe_augment(xyz, normals)

            ray_o = None
            ray_d = None
            ray_t = None
            ray_hit = None
            if self.use_ray_patch:
                required = ("ray_o_pool", "ray_d_pool", "ray_t_pool", "ray_hit_pool")
                missing = [k for k in required if k not in npz]
                if missing:
                    raise KeyError(f"Missing ray keys in {path}: {missing}")
                ray_o_full = npz["ray_o_pool"].astype(np.float32, copy=False)
                ray_d_full = npz["ray_d_pool"].astype(np.float32, copy=False)
                ray_t_full = npz["ray_t_pool"].astype(np.float32, copy=False)
                ray_hit_full = npz["ray_hit_pool"].astype(np.float32, copy=False)
                ridx = self._sample_ray_indices(ray_o_full.shape[0], sample_uid=(index * 7919 + 17))
                ray_o = ray_o_full[ridx]
                ray_d = ray_d_full[ridx]
                ray_t = ray_t_full[ridx]
                ray_hit = ray_hit_full[ridx]

        out = {
            "xyz": torch.from_numpy(xyz),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if normals is not None:
            out["normal"] = torch.from_numpy(normals)
        if ray_o is not None:
            out["ray_o"] = torch.from_numpy(ray_o)
            out["ray_d"] = torch.from_numpy(ray_d)
            out["ray_t"] = torch.from_numpy(ray_t)
            out["ray_hit"] = torch.from_numpy(ray_hit)
        return out


def load_scanobjectnn_h5_arrays(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load ScanObjectNN h5 arrays.

    Returns:
        points: (N, P, 3) float32
        labels: (N,) int64
    """
    with h5py.File(h5_path, "r") as f:
        points = np.asarray(f["data"], dtype=np.float32)
        labels = np.asarray(f["label"]).reshape(-1).astype(np.int64, copy=False)
    return points, labels


class PatchClsArrayDataset(Dataset):
    """Patch classification dataset from in-memory point arrays.

    This is used for direct ScanObjectNN h5 ingestion (without NPZ cache).
    """

    def __init__(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        *,
        n_point: int = 1024,
        sample_mode: str = "random",  # random | fps
        aug: bool = False,
        aug_cfg: Optional[PointAugConfig] = None,
        rng_seed: int = 0,
        mc_eval_k: int = 1,
        aug_eval: bool = False,
        deterministic_eval_sampling: bool = True,
    ) -> None:
        super().__init__()
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must be (N,P,3), got {points.shape}")
        if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
            raise ValueError(f"labels must be (N,), got {labels.shape} vs points={points.shape}")
        self.points = points.astype(np.float32, copy=False)
        self.labels = labels.astype(np.int64, copy=False)
        self.n_point = int(n_point)
        assert sample_mode in {"random", "fps"}
        self.sample_mode = sample_mode
        self.aug = bool(aug)
        self.aug_cfg = aug_cfg or PointAugConfig()
        self.rng_seed = int(rng_seed)
        self._rng: Optional[np.random.RandomState] = None
        self.mc_eval_k = int(mc_eval_k)
        self.aug_eval = bool(aug_eval)
        self.deterministic_eval_sampling = bool(deterministic_eval_sampling)
        self._warned_fps_fallback = False

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def _get_rng(self) -> np.random.RandomState:
        if self._rng is None:
            wi = get_worker_info()
            seed = self.rng_seed if wi is None else int(wi.seed)
            seed = int(seed) & 0xFFFFFFFF
            self._rng = np.random.RandomState(seed)
        return self._rng

    def _sample_indices(self, n_total: int, *, sample_uid: int = 0) -> np.ndarray:
        n = min(self.n_point, int(n_total))
        if (not self.aug) and self.deterministic_eval_sampling:
            seed = (int(self.rng_seed) * 1315423911 + int(sample_uid) * 2654435761) & 0xFFFFFFFF
            rng = np.random.RandomState(seed)
        else:
            rng = self._get_rng()
        # For direct h5 route we don't have precomputed fps order.
        # Keep behavior explicit and deterministic-per-worker.
        if self.sample_mode == "fps" and (not self._warned_fps_fallback):
            print("[warn] PatchClsArrayDataset: sample_mode=fps but no precomputed fps order in h5 route; using random subset.")
            self._warned_fps_fallback = True
        idx = rng.choice(n_total, size=n, replace=False)
        return idx.astype(np.int64, copy=False)

    def _maybe_augment(self, xyz: np.ndarray) -> np.ndarray:
        if not self.aug:
            return xyz
        rng = self._get_rng()
        if float(self.aug_cfg.prob) < 1.0 and float(rng.uniform(0.0, 1.0)) >= float(self.aug_cfg.prob):
            return xyz

        pt_xyz = xyz.astype(np.float32, copy=False)
        pt_dist = np.zeros((pt_xyz.shape[0], 1), dtype=np.float32)
        pools = {}
        pt_xyz2, _, _ = _apply_point_aug(
            pt_xyz,
            pt_dist,
            pools,
            rng=rng,
            rotate_z=(self.aug_cfg.rot_axis == "z") and (float(self.aug_cfg.rot_deg) > 0.0),
            scale_min=float(self.aug_cfg.scale_min),
            scale_max=float(self.aug_cfg.scale_max),
            translate=float(self.aug_cfg.shift_std),
            jitter_sigma=float(self.aug_cfg.jitter_std),
            jitter_clip=float(self.aug_cfg.jitter_clip),
        )
        return pt_xyz2

    def __getitem__(self, index: int) -> dict:
        pc_xyz = self.points[index]
        label = int(self.labels[index])

        if self.mc_eval_k > 1:
            xyz_list = []
            for k in range(self.mc_eval_k):
                idx = self._sample_indices(pc_xyz.shape[0], sample_uid=(index * 10007 + k))
                xyz = pc_xyz[idx]
                if self.aug_eval:
                    xyz = self._maybe_augment(xyz)
                xyz_list.append(xyz)
            xyz = np.stack(xyz_list, axis=0)  # (K,N,3)
        else:
            idx = self._sample_indices(pc_xyz.shape[0], sample_uid=index)
            xyz = pc_xyz[idx]
            xyz = self._maybe_augment(xyz)

        return {
            "xyz": torch.from_numpy(xyz),
            "label": torch.tensor(label, dtype=torch.long),
        }
