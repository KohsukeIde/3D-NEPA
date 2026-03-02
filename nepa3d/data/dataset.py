from __future__ import annotations

import numpy as np
import torch
import warnings
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
from ..token.tokenizer import (
    _choice,
    _order_point_tokens,
    _rand01,
    _sample_point_indices,
    build_sequence,
)


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

    # Apply rigid/scale to stored pools if present so downstream features stay aligned.
    # This keeps point tokens and ray tokens consistent when n_ray>0.
    pools_out = dict(pools)
    if (R is not None) or (scale is not None) or (shift is not None):
        if "pc_xyz" in pools_out and pools_out["pc_xyz"] is not None:
            try:
                pools_out["pc_xyz"] = _apply_aug_to_xyz(
                    np.asarray(pools_out["pc_xyz"]), R=R, scale=scale, shift=shift
                )
            except Exception:
                pass
        if "pc_n" in pools_out and pools_out["pc_n"] is not None:
            try:
                pn = np.asarray(pools_out["pc_n"])
                if R is not None:
                    pn = pn @ R.T
                pools_out["pc_n"] = pn
            except Exception:
                pass
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
        sequence_mode="block",
        event_order_mode="morton",
        ray_order_mode="theta_phi",
        ray_anchor_miss_t=4.0,
        ray_view_tol=1e-6,
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
        return_raw=False,
        label_map=None,
        pt_xyz_key="pt_xyz_pool",
        pt_dist_key="pt_dist_pool",
        ablate_point_dist=False,
        pt_sample_mode="random",
        pt_fps_key="pt_fps_order",
        pt_rfps_key="auto",
        pt_rfps_m=4096,
        point_order_mode="morton",
        # Augmentations
        aug_rotate_z=False,
        aug_scale_min=1.0,
        aug_scale_max=1.0,
        aug_translate=0.0,
        aug_jitter_sigma=0.0,
        aug_jitter_clip=0.0,
        aug_eval=False,
        aug_recompute_dist=False,
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
        self.sequence_mode = str(sequence_mode)
        self.event_order_mode = str(event_order_mode)
        self.ray_order_mode = str(ray_order_mode)
        self.ray_anchor_miss_t = float(ray_anchor_miss_t)
        self.ray_view_tol = float(ray_view_tol)
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
        self.return_raw = bool(return_raw)
        self.pt_xyz_key = str(pt_xyz_key)
        self.pt_dist_key = None if pt_dist_key is None else str(pt_dist_key)
        self.ablate_point_dist = bool(ablate_point_dist)
        self.pt_sample_mode = str(pt_sample_mode)
        self.pt_fps_key = str(pt_fps_key)
        self.pt_rfps_key = str(pt_rfps_key)
        self._warned_missing_rfps_order = False
        self._warned_ray_meta_missing = False
        self._warned_ray_pool_missing = False
        self.pt_rfps_m = int(pt_rfps_m)
        self.point_order_mode = str(point_order_mode)

        self.aug_rotate_z = bool(aug_rotate_z)
        self.aug_scale_min = float(aug_scale_min)
        self.aug_scale_max = float(aug_scale_max)
        self.aug_translate = float(aug_translate)
        self.aug_jitter_sigma = float(aug_jitter_sigma)
        self.aug_jitter_clip = float(aug_jitter_clip)
        self.aug_eval = bool(aug_eval)
        self.aug_recompute_dist = bool(aug_recompute_dist)
        self._warned_aug_recompute_dist_missing_ref = False
        self._warned_aug_recompute_dist_failed = False

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
        # Fail-fast for missing/incompatible dist pool unless explicitly ablated.
        if (pt_dist_pool is None) or (getattr(pt_dist_pool, "shape", None) is None) or (
            pt_dist_pool.shape[0] != pt_xyz_pool.shape[0]
        ):
            if self.ablate_point_dist:
                pt_dist_pool = np.zeros((pt_xyz_pool.shape[0], 1), dtype=np.float32)
            else:
                got_shape = getattr(pt_dist_pool, "shape", None)
                raise ValueError(
                    f"CRITICAL: pt_dist_key='{self.pt_dist_key}' is missing or incompatible with "
                    f"pt_xyz_key='{self.pt_xyz_key}' (path={path}). "
                    f"Expected first dim {pt_xyz_pool.shape[0]}, got {got_shape}."
                )
        else:
            # enforce (N,1)
            if pt_dist_pool.ndim == 1:
                pt_dist_pool = pt_dist_pool.reshape(-1, 1)

        ray_req_keys = ("ray_o_pool", "ray_d_pool", "ray_hit_pool", "ray_t_pool", "ray_n_pool")
        has_ray_pools = all((k in pools) and (pools.get(k) is not None) for k in ray_req_keys)
        if "ray_available" in pools:
            ray_available = bool(pools.get("ray_available", False))
            if ray_available and (not has_ray_pools):
                if not self._warned_ray_meta_missing:
                    warnings.warn(
                        f"ray_available=True but ray pools are missing (path={path}); "
                        "treating as missing-ray sample."
                    )
                    self._warned_ray_meta_missing = True
                ray_available = False
        else:
            if not has_ray_pools and not self._warned_ray_meta_missing:
                warnings.warn(
                    f"ray_available key is missing and ray pools are absent (path={path}); "
                    "treating as missing-ray sample."
                )
                self._warned_ray_meta_missing = True
            ray_available = bool(has_ray_pools)
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
                # Keep pt_dist strictly consistent with jittered query points when requested.
                if (
                    self.aug_recompute_dist
                    and (not self.ablate_point_dist)
                    and float(self.aug_jitter_sigma) > 0.0
                ):
                    surf_xyz = local_pools.get("pc_xyz", None) if isinstance(local_pools, dict) else None
                    if surf_xyz is None:
                        if not self._warned_aug_recompute_dist_missing_ref:
                            warnings.warn(
                                f"aug_recompute_dist requested but pc_xyz is missing (path={path}); "
                                "falling back to scaled dist augmentation only."
                            )
                            self._warned_aug_recompute_dist_missing_ref = True
                    else:
                        try:
                            from scipy.spatial import cKDTree

                            kdt = cKDTree(np.asarray(surf_xyz, dtype=np.float32))
                            dist_new, _ = kdt.query(np.asarray(xyz, dtype=np.float32), k=1)
                            dist = np.asarray(dist_new, dtype=np.float32).reshape(-1, 1)
                        except Exception as ex:
                            if not self._warned_aug_recompute_dist_failed:
                                warnings.warn(
                                    f"aug_recompute_dist failed (path={path}): {ex}; "
                                    "falling back to scaled dist augmentation only."
                                )
                                self._warned_aug_recompute_dist_failed = True

            if self.ablate_point_dist:
                dist = np.zeros_like(dist, dtype=np.float32)

            pt_fps_order = None
            resolved_fps_key = None
            pt_rfps_order = None
            resolved_rfps_key = None
            if isinstance(local_pools, dict):
                fps_key = self.pt_fps_key
                if str(fps_key).lower() == "auto":
                    candidates = [
                        f"{self.pt_xyz_key}_fps_order",
                        "pc_fps_order" if str(self.pt_xyz_key).startswith("pc_") else None,
                        "pt_fps_order",
                    ]
                    fps_key = None
                    for ck in candidates:
                        if ck is not None and ck in local_pools:
                            fps_key = ck
                            break
                resolved_fps_key = fps_key
                if fps_key is not None:
                    pt_fps_order = local_pools.get(fps_key, None)
                if (
                    str(self.pt_sample_mode).lower() == "fps"
                    and pt_fps_order is None
                ):
                    raise ValueError(
                        f"CRITICAL: pt_sample_mode='fps' requires a cached FPS order key, "
                        f"but key is missing (pt_fps_key={self.pt_fps_key}, "
                        f"resolved={resolved_fps_key}, path={path}). "
                        "Backfill FPS order first (e.g., pt_fps_order)."
                    )

                rfps_key = self.pt_rfps_key
                if str(rfps_key).lower() == "auto":
                    candidates = [
                        f"{self.pt_xyz_key}_rfps_order_bank",
                        "pc_rfps_order_bank" if str(self.pt_xyz_key).startswith("pc_") else None,
                        "pt_rfps_order_bank",
                        f"{self.pt_xyz_key}_rfps_order",
                        "pc_rfps_order" if str(self.pt_xyz_key).startswith("pc_") else None,
                        "pt_rfps_order",
                    ]
                    rfps_key = None
                    for ck in candidates:
                        if ck is not None and ck in local_pools:
                            rfps_key = ck
                            break
                resolved_rfps_key = rfps_key
                if rfps_key is not None:
                    pt_rfps_order = local_pools.get(rfps_key, None)
                if (
                    str(self.pt_sample_mode).lower() == "rfps_cached"
                    and pt_rfps_order is None
                ):
                    raise ValueError(
                        f"CRITICAL: pt_sample_mode='rfps_cached' requires a cached RFPS order bank, "
                        f"but key is missing (pt_rfps_key={self.pt_rfps_key}, "
                        f"resolved={resolved_rfps_key}, path={path}). "
                        "Backfill RFPS bank first (e.g., pt_rfps_order_bank)."
                    )

            if self.return_raw:
                p_idx = _sample_point_indices(
                    pt_xyz_pool=np.asarray(xyz),
                    n_point=self.n_point,
                    rng=rng,
                    pt_sample_mode=self.pt_sample_mode,
                    pt_fps_order=pt_fps_order,
                    pt_rfps_order=pt_rfps_order,
                    pt_rfps_m=self.pt_rfps_m,
                )
                pt_xyz_s = np.asarray(xyz)[p_idx].astype(np.float32, copy=False)
                dist_arr = np.asarray(dist)
                pt_dist_1d = (
                    dist_arr[:, 0]
                    if dist_arr.ndim == 2
                    else dist_arr.reshape(-1)
                )
                pt_dist_s = np.asarray(pt_dist_1d[p_idx], dtype=np.float32)
                pt_xyz_s, pt_dist_s, _ = _order_point_tokens(
                    pt_xyz_s,
                    pt_dist_s,
                    rng=rng,
                    point_order_mode=self.point_order_mode,
                )
                pt_dist_s = pt_dist_s.reshape(-1, 1)

                ray_o_s = np.zeros((0, 3), dtype=np.float32)
                ray_d_s = np.zeros((0, 3), dtype=np.float32)
                ray_t_s = np.zeros((0, 1), dtype=np.float32)
                ray_hit_s = np.zeros((0, 1), dtype=np.float32)
                ray_n_s = np.zeros((0, 3), dtype=np.float32)
                ray_unc_s = np.zeros((0, 1), dtype=np.float32)
                ray_ok = bool(ray_available)

                if int(self.n_ray) > 0:
                    drop_this = (float(drop_ray_prob) > 0.0) and (_rand01(rng) < float(drop_ray_prob))
                    if drop_this:
                        ray_ok = False

                    if ray_ok:
                        ray_o = local_pools.get("ray_o_pool", None)
                        ray_d = local_pools.get("ray_d_pool", None)
                        ray_hit = local_pools.get("ray_hit_pool", None)
                        ray_t = local_pools.get("ray_t_pool", None)
                        ray_n = local_pools.get("ray_n_pool", None)
                        ray_unc = local_pools.get("ray_unc_pool", None)
                        if ray_o is None or ray_d is None or ray_hit is None or ray_t is None or ray_n is None:
                            if not self._warned_ray_pool_missing:
                                warnings.warn(
                                    f"ray pools are missing while ray is marked available (path={path}); "
                                    "falling back to missing-ray dummy tensors."
                                )
                                self._warned_ray_pool_missing = True
                            ray_ok = False

                    if ray_ok:
                        ray_o = np.asarray(ray_o)
                        ray_d = np.asarray(ray_d)
                        ray_hit = np.asarray(ray_hit).reshape(-1)
                        ray_t = np.asarray(ray_t).reshape(-1)
                        ray_n = np.asarray(ray_n)
                        if ray_unc is not None:
                            ray_unc = np.asarray(ray_unc).reshape(-1)

                        if int(ray_o.shape[0]) <= 0:
                            if not self._warned_ray_pool_missing:
                                warnings.warn(
                                    f"ray pools are empty while ray is marked available (path={path}); "
                                    "falling back to missing-ray dummy tensors."
                                )
                                self._warned_ray_pool_missing = True
                            ray_ok = False

                    if ray_ok:
                        r_idx = _choice(ray_o.shape[0], self.n_ray, rng=rng)
                        ray_o_s = ray_o[r_idx].astype(np.float32, copy=False)
                        ray_d_s = ray_d[r_idx].astype(np.float32, copy=False)
                        ray_t_s = ray_t[r_idx].astype(np.float32, copy=False).reshape(-1, 1)
                        ray_hit_s = ray_hit[r_idx].astype(np.float32, copy=False).reshape(-1, 1)
                        ray_n_s = ray_n[r_idx].astype(np.float32, copy=False)
                        if ray_unc is None:
                            ray_unc_s = np.zeros((ray_o_s.shape[0], 1), dtype=np.float32)
                        else:
                            ray_unc_s = ray_unc[r_idx].astype(np.float32, copy=False).reshape(-1, 1)
                    else:
                        ray_o_s = np.zeros((self.n_ray, 3), dtype=np.float32)
                        ray_d_s = np.zeros((self.n_ray, 3), dtype=np.float32)
                        ray_t_s = np.zeros((self.n_ray, 1), dtype=np.float32)
                        ray_hit_s = np.zeros((self.n_ray, 1), dtype=np.float32)
                        ray_n_s = np.zeros((self.n_ray, 3), dtype=np.float32)
                        ray_unc_s = np.zeros((self.n_ray, 1), dtype=np.float32)

                return {
                    "pt_xyz": pt_xyz_s,
                    "pt_dist": pt_dist_s.astype(np.float32, copy=False),
                    "ray_o": ray_o_s,
                    "ray_d": ray_d_s,
                    "ray_t": ray_t_s,
                    "ray_hit": ray_hit_s,
                    "ray_n": ray_n_s,
                    "ray_unc": ray_unc_s,
                    "ray_available": np.asarray(int(ray_ok), dtype=np.int64),
                }

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
                sequence_mode=self.sequence_mode,
                event_order_mode=self.event_order_mode,
                ray_order_mode=self.ray_order_mode,
                ray_anchor_miss_t=self.ray_anchor_miss_t,
                ray_view_tol=self.ray_view_tol,
                pt_sample_mode=self.pt_sample_mode,
                pt_fps_order=pt_fps_order,
                pt_rfps_order=pt_rfps_order,
                pt_rfps_m=self.pt_rfps_m,
                point_order_mode=self.point_order_mode,
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
            if self.return_raw:
                rng = np.random.RandomState(base)
                raw = _run_one(rng, drop_ray_prob=0.0)
            else:
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
            if self.return_raw:
                raw = _run_one(np.random, drop_ray_prob=self.drop_ray_prob)
            else:
                feat, type_id = _run_one(np.random, drop_ray_prob=self.drop_ray_prob)

        if self.return_raw:
            out = {
                "pt_xyz": torch.from_numpy(raw["pt_xyz"]).float(),
                "pt_dist": torch.from_numpy(raw["pt_dist"]).float(),
                "ray_o": torch.from_numpy(raw["ray_o"]).float(),
                "ray_d": torch.from_numpy(raw["ray_d"]).float(),
                "ray_t": torch.from_numpy(raw["ray_t"]).float(),
                "ray_hit": torch.from_numpy(raw["ray_hit"]).float(),
                "ray_n": torch.from_numpy(raw["ray_n"]).float(),
                "ray_unc": torch.from_numpy(raw["ray_unc"]).float(),
                "ray_available": torch.tensor(int(raw["ray_available"]), dtype=torch.long),
            }
        else:
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
