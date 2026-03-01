import numpy as np
from typing import Optional
from scipy.spatial import cKDTree


class PointCloudBackend:
    """PointCloud backend with ray answers rendered from pointcloud occupancy (DDA) if available.

    Policy:
    - pointcloud / pointcloud_noray must use observation-derived pt_dist_pc_pool.
    - mesh-derived distance fallback is disabled by default (fail-fast).
    """

    def __init__(self, npz_path, *, require_pt_dist_pc: bool = True):
        self.npz_path = str(npz_path)
        self.d = np.load(npz_path, allow_pickle=False)
        self.pc = self.d["pc_xyz"].astype(np.float32, copy=False)
        self.kdt = None
        self.require_pt_dist_pc = bool(require_pt_dist_pc)

    def _pc_knn_dist(self, pt_xyz):
        if self.kdt is None:
            self.kdt = cKDTree(self.pc)
        dist, _ = self.kdt.query(pt_xyz, k=1)
        return dist.astype(np.float32, copy=False)

    def _resolve_point_dist(self, pt_xyz, *, require_pt_dist_pc: Optional[bool] = None):
        require_pc = self.require_pt_dist_pc if require_pt_dist_pc is None else bool(require_pt_dist_pc)

        # Preferred/strict path: observation-derived distances (pc->query).
        if "pt_dist_pc_pool" in self.d:
            pt_dist = self.d["pt_dist_pc_pool"].astype(np.float32, copy=False)
            # If preprocess marked distances as invalid (-1) or NaN, recompute from pc on-the-fly.
            if (not np.isfinite(pt_dist).all()) or np.any(pt_dist < 0):
                pt_dist = self._pc_knn_dist(pt_xyz)
            return pt_dist

        if require_pc:
            raise KeyError(
                "CRITICAL: pointcloud backend requires 'pt_dist_pc_pool' for fairness, "
                f"but key is missing in cache: {self.npz_path}"
            )

        # Legacy fallback (mesh-derived distances), allowed only in explicit legacy modes.
        if "pt_dist_pool" in self.d:
            return self.d["pt_dist_pool"].astype(np.float32, copy=False)

        # Backward-compat for very old caches without any distance pool.
        return self._pc_knn_dist(pt_xyz)

    def _append_cached_orders(self, pools):
        if "pt_fps_order" in self.d:
            pfo = self.d["pt_fps_order"].astype(np.int32, copy=False)
            pools["pt_fps_order"] = pfo
            pools["pt_xyz_pool_fps_order"] = pfo
        if "pc_fps_order" in self.d:
            pco = self.d["pc_fps_order"].astype(np.int32, copy=False)
            pools["pc_fps_order"] = pco
            pools["pc_xyz_fps_order"] = pco
        # Optional cached RFPS order(s) / banks.
        for k in [
            "pt_rfps_order_bank",
            "pc_rfps_order_bank",
            "pt_xyz_pool_rfps_order_bank",
            "pc_xyz_rfps_order_bank",
            "pt_rfps_order",
            "pc_rfps_order",
            "pt_xyz_pool_rfps_order",
            "pc_xyz_rfps_order",
        ]:
            if k in self.d:
                pools[k] = self.d[k].astype(np.int32, copy=False)

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)
        pc_xyz = self.pc
        pt_dist = self._resolve_point_dist(pt_xyz)

        # Preferred: pointcloud-rendered ray pools (generated in preprocess).
        if "ray_hit_pc_pool" in self.d and "ray_t_pc_pool" in self.d and "ray_n_pc_pool" in self.d:
            ray_hit = self.d["ray_hit_pc_pool"].astype(np.float32, copy=False)
            ray_t = self.d["ray_t_pc_pool"].astype(np.float32, copy=False)
            ray_n = self.d["ray_n_pc_pool"].astype(np.float32, copy=False)
        else:
            # Backward-compat: fall back to mesh ray pools (v0 cache).
            ray_hit = self.d["ray_hit_pool"].astype(np.float32, copy=False)
            ray_t = self.d["ray_t_pool"].astype(np.float32, copy=False)
            ray_n = self.d["ray_n_pool"].astype(np.float32, copy=False)

        pools = {
            "pt_xyz_pool": pt_xyz,
            "pc_xyz": pc_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": self.d["ray_o_pool"].astype(np.float32, copy=False),
            "ray_d_pool": self.d["ray_d_pool"].astype(np.float32, copy=False),
            "ray_hit_pool": ray_hit,
            "ray_t_pool": ray_t,
            "ray_n_pool": ray_n,
            "ray_available": True,
        }
        self._append_cached_orders(pools)
        return pools


class PointCloudMeshRayBackend(PointCloudBackend):
    """Legacy PointCloud backend that reuses mesh ray pools (for ablation / reproducing v0)."""

    def __init__(self, npz_path):
        # Legacy mode: allow pt_dist_pool fallback for reproducibility.
        super().__init__(npz_path, require_pt_dist_pc=False)

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)
        pc_xyz = self.pc
        pt_dist = self._resolve_point_dist(pt_xyz, require_pt_dist_pc=False)

        pools = {
            "pt_xyz_pool": pt_xyz,
            "pc_xyz": pc_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": self.d["ray_o_pool"].astype(np.float32, copy=False),
            "ray_d_pool": self.d["ray_d_pool"].astype(np.float32, copy=False),
            "ray_hit_pool": self.d["ray_hit_pool"].astype(np.float32, copy=False),
            "ray_t_pool": self.d["ray_t_pool"].astype(np.float32, copy=False),
            "ray_n_pool": self.d["ray_n_pool"].astype(np.float32, copy=False),
            "ray_available": True,
        }
        self._append_cached_orders(pools)
        return pools


class PointCloudNoRayBackend(PointCloudBackend):
    """PointCloud backend that explicitly marks ray modality as unavailable."""

    def get_pools(self):
        out = super().get_pools()
        out["ray_available"] = False
        return out
