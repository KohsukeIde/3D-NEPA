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


def _rot_z_matrix(angle_rad: float, dtype=np.float32) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)


def _apply_aug_to_xyz(
    xyz: np.ndarray,
    *,
    R: np.ndarray | None,
    scale: float | None,
    shift: np.ndarray | None,
) -> np.ndarray:
    out = xyz
    if R is not None:
        out = out @ R.T
    if scale is not None:
        out = out * float(scale)
    if shift is not None:
        out = out + shift
    return out


def _apply_point_aug(
    pt_xyz: np.ndarray,
    pt_dist: np.ndarray,
    pools: dict,
    rng,
    *,
    rotate_z: bool,
    scale_min: float,
    scale_max: float,
    translate: float,
    jitter_sigma: float,
    jitter_clip: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply simple augmentations.

    The augmentation is applied consistently to:
      - point pool used for token sampling (pt_xyz)
      - ray pools if present (ray_o, ray_d, ray_n, ray_t)

    Distances scale with global scaling.

    Notes:
      - Jitter is applied ONLY to point xyz (not rays).
      - rng must support uniform()/normal() (np.random or np.random.RandomState).
    """

    xyz = pt_xyz
    dist = pt_dist

    R = None
    if rotate_z:
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        R = _rot_z_matrix(angle, dtype=xyz.dtype)

    scale = None
    if (scale_min is not None) and (scale_max is not None) and (float(scale_min) != 1.0 or float(scale_max) != 1.0):
        scale = float(rng.uniform(float(scale_min), float(scale_max)))

    shift = None
    if translate and float(translate) > 0.0:
        t = float(translate)
        shift = rng.uniform(-t, t, size=(1, 3)).astype(xyz.dtype)

    xyz = _apply_aug_to_xyz(xyz, R=R, scale=scale, shift=shift)
    if scale is not None:
        dist = dist * scale

    if jitter_sigma and float(jitter_sigma) > 0.0:
        sigma = float(jitter_sigma)
        noise = rng.normal(0.0, sigma, size=xyz.shape).astype(xyz.dtype)
        if jitter_clip and float(jitter_clip) > 0.0:
            clip = float(jitter_clip)
            noise = np.clip(noise, -clip, clip)
        xyz = xyz + noise

    # Rays: apply rigid/scale to ray pools if they exist.
    # This keeps point tokens and ray tokens consistent when n_ray>0.
    pools_out = dict(pools)
    if (R is not None) or (scale is not None) or (shift is not None):
        if "ray_o_pool" in pools_out and pools_out["ray_o_pool"] is not None:
            try:
                pools_out["ray_o_pool"] = _apply_aug_to_xyz(np.asarray(pools_out["ray_o_pool"]), R=R, scale=scale, shift=shift)
            except Exception:
                pass
        if "ray_d_pool" in pools_out and pools_out["ray_d_pool"] is not None:
            try:
                rd = np.asarray(pools_out["ray_d_pool"])
                if R is not None:
                    rd = rd @ R.T
                pools_out["ray_d_pool"] = rd
            except Exception:
                pass
        if "ray_n_pool" in pools_out and pools_out["ray_n_pool"] is not None:
            try:
                rn = np.asarray(pools_out["ray_n_pool"])
                if R is not None:
                    rn = rn @ R.T
                pools_out["ray_n_pool"] = rn
            except Exception:
                pass
        if "ray_t_pool" in pools_out and pools_out["ray_t_pool"] is not None and scale is not None:
            try:
                rt = np.asarray(pools_out["ray_t_pool"])
                pools_out["ray_t_pool"] = rt * scale
            except Exception:
                pass

    return xyz, dist, pools_out


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
        qa_tokens=0,
        qa_layout="interleave",
        include_pt_grad=False,
        pt_grad_mode="raw",
        pt_grad_eps=1e-3,
        pt_grad_clip=10.0,
        pt_grad_orient="none",
        include_ray_unc=False,
        ray_unc_k=8,
        ray_unc_mode="normal_var",
        voxel_grid=64,
        voxel_dilate=1,
        voxel_max_steps=0,
        return_label=False,
        label_map=None,
        pt_xyz_key="pt_xyz_pool",
        pt_dist_key="pt_dist_pool",
        ablate_point_dist=False,
        pt_sample_mode="random",
        pt_fps_key="pt_fps_order",
        pt_rfps_m=4096,
        # Augmentations
        aug_rotate_z=False,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aug_translate=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_eval=False,
    ):
        self.paths = list(npz_paths)
        self.backend = backend
        self.n_point = int(n_point)
        self.n_ray = int(n_ray)
        self.mode = str(mode)
        self.eval_seed = int(eval_seed)
        self.mc_eval_k = int(mc_eval_k)
        self.drop_ray_prob = float(drop_ray_prob)
        self.force_missing_ray = bool(force_missing_ray)
        self.add_eos = bool(add_eos)
        self.qa_tokens = int(qa_tokens)
        self.qa_layout = str(qa_layout)
        self.include_pt_grad = bool(include_pt_grad)
        self.pt_grad_mode = str(pt_grad_mode)
        self.pt_grad_eps = float(pt_grad_eps)
        self.pt_grad_clip = float(pt_grad_clip)
        self.pt_grad_orient = str(pt_grad_orient)
        self.include_ray_unc = bool(include_ray_unc)
        self.ray_unc_k = int(ray_unc_k)
        self.ray_unc_mode = str(ray_unc_mode)
        self.voxel_grid = int(voxel_grid)
        self.voxel_dilate = int(voxel_dilate)
        self.voxel_max_steps = int(voxel_max_steps)
        self.return_label = bool(return_label)
        self.pt_xyz_key = str(pt_xyz_key)
        self.pt_dist_key = None if pt_dist_key is None else str(pt_dist_key)
        self.ablate_point_dist = bool(ablate_point_dist)
        self.pt_sample_mode = str(pt_sample_mode)
        self.pt_fps_key = str(pt_fps_key)
        self.pt_rfps_m = int(pt_rfps_m)

        self.aug_rotate_z = bool(aug_rotate_z)
        self.aug_scale_min = float(aug_scale_min)
        self.aug_scale_max = float(aug_scale_max)
        self.aug_translate = float(aug_translate)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)
        self.aug_eval = bool(aug_eval)

        if self.return_label:
            self.label_map = label_map or build_label_map(self.paths)
        else:
            self.label_map = None

    def __len__(self):
        return len(self.paths)

    # -------------------------
    # Runtime scaling helpers
    # -------------------------
    def set_sizes(self, *, n_point: int | None = None, n_ray: int | None = None) -> None:
        """Update sampling sizes at runtime."""

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

    def _want_aug(self) -> bool:
        if self.mode == "train":
            return bool(
                self.aug_rotate_z
                or self.aug_translate
                or (self.aug_scale_min != 1.0)
                or (self.aug_scale_max != 1.0)
                or self.aug_jitter_sigma
            )
        if self.mode == "eval" and self.aug_eval:
            return bool(
                self.aug_rotate_z
                or self.aug_translate
                or (self.aug_scale_min != 1.0)
                or (self.aug_scale_max != 1.0)
                or self.aug_jitter_sigma
            )
        return False

    def __getitem__(self, idx):
        path = self.paths[idx]
        be = self._make_backend(path)
        pools = be.get_pools()

        # Select pools for point tokens (defaults preserve legacy behavior).
        pt_xyz_pool = pools.get(self.pt_xyz_key)
        if pt_xyz_pool is None:
            pt_xyz_pool = pools.get("pt_xyz_pool")
        if pt_xyz_pool is None:
            pt_xyz_pool = pools.get("pc_xyz")
        if pt_xyz_pool is None:
            raise KeyError(
                f"point pool not found: pt_xyz_key={self.pt_xyz_key} (available keys: {list(pools.keys())})"
            )

        pt_dist_pool = None
        if self.pt_dist_key:
            pt_dist_pool = pools.get(self.pt_dist_key)
        # If dist pool is missing or mismatched length, fall back to zeros.
        if (pt_dist_pool is None) or (getattr(pt_dist_pool, "shape", None) is None) or (
            pt_dist_pool.shape[0] != pt_xyz_pool.shape[0]
        ):
            pt_dist_pool = np.zeros((pt_xyz_pool.shape[0], 1), dtype=np.float32)
        else:
            # enforce (N,1)
            if pt_dist_pool.ndim == 1:
                pt_dist_pool = pt_dist_pool.reshape(-1, 1)

        ray_available = bool(pools.get("ray_available", True))
        if self.force_missing_ray:
            ray_available = False

        # Deterministic eval: seed depends on sample path, not on global RNG / dataloader worker.
        def _stable_seed(s):
            import hashlib

            h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            return int(h, 16)

        def _run_one(rng, drop_ray_prob: float):
            # Optional augmentation (copy to avoid mutating cached pools).
            local_pools = pools
            xyz = pt_xyz_pool
            dist = pt_dist_pool
            if self._want_aug():
                xyz = np.asarray(xyz).copy()
                dist = np.asarray(dist).copy()
                local_pools = dict(pools)
                # Copy ray pools only if present; tokenizer will ignore when n_ray==0.
                for k in ["ray_o_pool", "ray_d_pool", "ray_hit_pool", "ray_t_pool", "ray_n_pool"]:
                    if k in local_pools and isinstance(local_pools[k], np.ndarray):
                        local_pools[k] = local_pools[k].copy()
                xyz, dist, local_pools = _apply_point_aug(
                    xyz,
                    dist,
                    local_pools,
                    rng,
                    rotate_z=self.aug_rotate_z,
                    scale_min=self.aug_scale_min,
                    scale_max=self.aug_scale_max,
                    translate=self.aug_translate,
                    jitter_sigma=self.aug_jitter_sigma,
                    jitter_clip=self.aug_jitter_clip,
                )

            if self.ablate_point_dist:
                dist = np.zeros_like(dist, dtype=np.float32)

            pt_fps_order = local_pools.get(self.pt_fps_key, None) if isinstance(local_pools, dict) else None

            feat, type_id = build_sequence(
                xyz,
                dist,
                local_pools.get("ray_o_pool", None),
                local_pools.get("ray_d_pool", None),
                local_pools.get("ray_hit_pool", None),
                local_pools.get("ray_t_pool", None),
                local_pools.get("ray_n_pool", None),
                n_point=self.n_point,
                n_ray=self.n_ray,
                drop_ray_prob=drop_ray_prob,
                ray_available=ray_available,
                add_eos=self.add_eos,
                qa_tokens=self.qa_tokens,
                qa_layout=self.qa_layout,
                pt_sample_mode=self.pt_sample_mode,
                pt_fps_order=pt_fps_order,
                pt_rfps_m=self.pt_rfps_m,
                include_pt_grad=self.include_pt_grad,
                pt_grad_mode=self.pt_grad_mode,
                pt_grad_eps=self.pt_grad_eps,
                pt_grad_clip=self.pt_grad_clip,
                pt_grad_orient=self.pt_grad_orient,
                include_ray_unc=self.include_ray_unc,
                ray_unc_k=self.ray_unc_k,
                ray_unc_mode=self.ray_unc_mode,
                rng=rng,
            )
            return feat, type_id

        if self.mode == "eval":
            base = (_stable_seed(path) + self.eval_seed) & 0xFFFFFFFF
            k = max(1, self.mc_eval_k)
            feats = []
            types = []
            for i in range(k):
                rng = np.random.RandomState((base + 1000003 * i) & 0xFFFFFFFF)
                feat, type_id = _run_one(rng, drop_ray_prob=0.0)
                feats.append(feat)
                types.append(type_id)
            feat = np.stack(feats, axis=0) if k > 1 else feats[0]
            type_id = np.stack(types, axis=0) if k > 1 else types[0]
        else:
            feat, type_id = _run_one(np.random, drop_ray_prob=self.drop_ray_prob)

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
