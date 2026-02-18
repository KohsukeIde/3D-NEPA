import numpy as np
import torch
from torch.utils.data import Dataset

from .modelnet40_index import label_from_path, build_label_map
from ..backends.mesh_backend import MeshBackend
from ..backends.pointcloud_backend import (
    PointCloudBackend,
    PointCloudMeshRayBackend,
    PointCloudNoRayBackend,
)
from ..backends.voxel_backend import VoxelBackend
from ..backends.udfgrid_backend import UDFGridBackend
from ..token.tokenizer import build_sequence


class ModelNet40QueryDataset(Dataset):
    def __init__(
        self,
        npz_paths,
        backend="mesh",
        n_point=512,
        n_ray=512,
        mode="train",
        eval_seed=0,
        mc_eval_k=1,
        drop_ray_prob=0.0,
        force_missing_ray=False,
        add_eos=True,
        qa_tokens=False,
        voxel_grid=64,
        voxel_dilate=1,
        voxel_max_steps=0,
        return_label=False,
        label_map=None,
        pt_xyz_key="pt_xyz_pool",
        pt_dist_key="pt_dist_pool",
        ablate_point_dist=False,
    ):
        self.paths = list(npz_paths)
        self.backend = backend
        self.n_point = n_point
        self.n_ray = n_ray
        self.mode = str(mode)
        self.eval_seed = int(eval_seed)
        self.mc_eval_k = int(mc_eval_k)
        self.drop_ray_prob = float(drop_ray_prob)
        self.force_missing_ray = bool(force_missing_ray)
        self.add_eos = bool(add_eos)
        self.qa_tokens = bool(qa_tokens)
        self.voxel_grid = int(voxel_grid)
        self.voxel_dilate = int(voxel_dilate)
        self.voxel_max_steps = int(voxel_max_steps)
        self.return_label = return_label
        self.pt_xyz_key = str(pt_xyz_key)
        self.pt_dist_key = None if pt_dist_key is None else str(pt_dist_key)
        self.ablate_point_dist = bool(ablate_point_dist)
        if return_label:
            self.label_map = label_map or build_label_map(self.paths)
        else:
            self.label_map = None

    def __len__(self):
        return len(self.paths)

    # -------------------------
    # Runtime scaling helpers
    # -------------------------
    def set_sizes(self, *, n_point: int | None = None, n_ray: int | None = None) -> None:
        """Update sampling sizes at runtime.

        Used by point/ray scaling curricula (e.g., 256 -> 512 -> 1024).
        The dataloader keeps a reference to the dataset instance, so mutating
        these fields affects subsequent __getitem__ calls.
        """

        if n_point is not None:
            self.n_point = int(n_point)
        if n_ray is not None:
            self.n_ray = int(n_ray)

    def _make_backend(self, path):
        if self.backend == "mesh":
            return MeshBackend(path)
        if self.backend == "pointcloud":
            return PointCloudBackend(path)
        if self.backend == "pointcloud_meshray":
            return PointCloudMeshRayBackend(path)
        if self.backend == "pointcloud_noray":
            return PointCloudNoRayBackend(path)
        if self.backend == "voxel":
            return VoxelBackend(
                path,
                grid=self.voxel_grid,
                dilate=self.voxel_dilate,
                max_steps=self.voxel_max_steps,
            )
        if self.backend == "udfgrid":
            return UDFGridBackend(path)

        raise ValueError(f"unknown backend: {self.backend}")

    def __getitem__(self, idx):
        path = self.paths[idx]
        be = self._make_backend(path)
        pools = be.get_pools()
        # Select point pools from configurable keys (legacy defaults keep behavior unchanged).
        pt_xyz_pool = pools.get(self.pt_xyz_key)
        if pt_xyz_pool is None:
            pt_xyz_pool = pools.get("pt_xyz_pool")
        if pt_xyz_pool is None:
            pt_xyz_pool = pools.get("pc_xyz")
        if pt_xyz_pool is None:
            raise KeyError(
                f"point xyz pool not found: pt_xyz_key={self.pt_xyz_key} available={list(pools.keys())}"
            )

        pt_dist_pool = None
        if self.pt_dist_key:
            pt_dist_pool = pools.get(self.pt_dist_key)
        if (
            pt_dist_pool is None
            or getattr(pt_dist_pool, "shape", None) is None
            or pt_dist_pool.shape[0] != pt_xyz_pool.shape[0]
        ):
            pt_dist_pool = np.zeros((pt_xyz_pool.shape[0], 1), dtype=np.float32)

        ray_available = bool(pools.get("ray_available", True))
        if self.force_missing_ray:
            ray_available = False

        # Deterministic eval: seed depends on sample path, not on global RNG / dataloader worker.
        def _stable_seed(s):
            import hashlib
            h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            return int(h, 16)

        if self.mode == "eval":
            base = (_stable_seed(path) + self.eval_seed) & 0xFFFFFFFF
            k = max(1, self.mc_eval_k)
            feats = []
            types = []
            pt_dist = pt_dist_pool
            if self.ablate_point_dist:
                pt_dist = np.zeros_like(pt_dist, dtype=np.float32)
            for i in range(k):
                rng = np.random.RandomState((base + 1000003 * i) & 0xFFFFFFFF)
                feat, type_id = build_sequence(
                    pt_xyz_pool,
                    pt_dist,
                    pools["ray_o_pool"],
                    pools["ray_d_pool"],
                    pools["ray_hit_pool"],
                    pools["ray_t_pool"],
                    pools["ray_n_pool"],
                    n_point=self.n_point,
                    n_ray=self.n_ray,
                    drop_ray_prob=0.0,
                    ray_available=ray_available,
                    add_eos=self.add_eos,
                    qa_tokens=self.qa_tokens,
                    rng=rng,
                )
                feats.append(feat)
                types.append(type_id)
            feat = np.stack(feats, axis=0) if k > 1 else feats[0]
            type_id = np.stack(types, axis=0) if k > 1 else types[0]
        else:
            pt_dist = pt_dist_pool
            if self.ablate_point_dist:
                pt_dist = np.zeros_like(pt_dist, dtype=np.float32)
            feat, type_id = build_sequence(
                pt_xyz_pool,
                pt_dist,
                pools["ray_o_pool"],
                pools["ray_d_pool"],
                pools["ray_hit_pool"],
                pools["ray_t_pool"],
                pools["ray_n_pool"],
                n_point=self.n_point,
                n_ray=self.n_ray,
                drop_ray_prob=self.drop_ray_prob,
                ray_available=ray_available,
                add_eos=self.add_eos,
                qa_tokens=self.qa_tokens,
                rng=np.random,
            )

        out = {
            "feat": torch.from_numpy(feat),
            "type_id": torch.from_numpy(type_id),
        }
        if self.return_label:
            label = label_from_path(path)
            out["label"] = torch.tensor(self.label_map[label], dtype=torch.long)
        return out


def collate(batch):
    feat0 = batch[0]["feat"]
    type0 = batch[0]["type_id"]

    # Standard: feat (T,F) -> (B,T,F)
    # MC-eval: feat (K,T,F) -> (B,K,T,F)
    if feat0.dim() == 2:
        feat = torch.stack([b["feat"] for b in batch], dim=0)
        type_id = torch.stack([b["type_id"] for b in batch], dim=0)
    elif feat0.dim() == 3:
        feat = torch.stack([b["feat"] for b in batch], dim=0)
        type_id = torch.stack([b["type_id"] for b in batch], dim=0)
    else:
        raise ValueError(f"unexpected feat ndim: {feat0.dim()}")

    out = {"feat": feat, "type_id": type_id}
    if "label" in batch[0]:
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
    return out
