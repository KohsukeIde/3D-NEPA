import numpy as np
from scipy.spatial import cKDTree


class PointCloudBackend:
    """PointCloud backend with ray answers rendered from pointcloud occupancy (DDA) if available."""

    def __init__(self, npz_path):
        self.d = np.load(npz_path, allow_pickle=False)
        self.pc = self.d["pc_xyz"].astype(np.float32, copy=False)
        self.kdt = None

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)
        # Prefer observation-derived distances if available (pc->query).
        # This avoids leaking mesh-derived distances (pt_dist_pool) into pointcloud experiments.
        if "pt_dist_pc_pool" in self.d:
            pt_dist = self.d["pt_dist_pc_pool"].astype(np.float32, copy=False)
            # If preprocess marked distances as invalid (-1) or NaN, recompute from pc on-the-fly.
            if (not np.isfinite(pt_dist).all()) or np.any(pt_dist < 0):
                if self.kdt is None:
                    self.kdt = cKDTree(self.pc)
                dist, _ = self.kdt.query(pt_xyz, k=1)
                pt_dist = dist.astype(np.float32, copy=False)
        # Fallback: mesh-derived distances (legacy caches).
        elif "pt_dist_pool" in self.d:
            pt_dist = self.d["pt_dist_pool"].astype(np.float32, copy=False)
        else:
            # Backward-compat for old caches without any distance pool.
            if self.kdt is None:
                self.kdt = cKDTree(self.pc)
            dist, _ = self.kdt.query(pt_xyz, k=1)
            pt_dist = dist.astype(np.float32, copy=False)

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

        return {
            "pt_xyz_pool": pt_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": self.d["ray_o_pool"].astype(np.float32, copy=False),
            "ray_d_pool": self.d["ray_d_pool"].astype(np.float32, copy=False),
            "ray_hit_pool": ray_hit,
            "ray_t_pool": ray_t,
            "ray_n_pool": ray_n,
            "ray_available": True,
        }


class PointCloudMeshRayBackend(PointCloudBackend):
    """Legacy PointCloud backend that reuses mesh ray pools (for ablation / reproducing v0)."""

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)
        # Prefer observation-derived distances if available (pc->query).
        if "pt_dist_pc_pool" in self.d:
            pt_dist = self.d["pt_dist_pc_pool"].astype(np.float32, copy=False)
            # If preprocess marked distances as invalid (-1) or NaN, recompute from pc on-the-fly.
            if (not np.isfinite(pt_dist).all()) or np.any(pt_dist < 0):
                if self.kdt is None:
                    self.kdt = cKDTree(self.pc)
                dist, _ = self.kdt.query(pt_xyz, k=1)
                pt_dist = dist.astype(np.float32, copy=False)
        elif "pt_dist_pool" in self.d:
            pt_dist = self.d["pt_dist_pool"].astype(np.float32, copy=False)
        else:
            if self.kdt is None:
                self.kdt = cKDTree(self.pc)
            dist, _ = self.kdt.query(pt_xyz, k=1)
            pt_dist = dist.astype(np.float32, copy=False)
        return {
            "pt_xyz_pool": pt_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": self.d["ray_o_pool"].astype(np.float32, copy=False),
            "ray_d_pool": self.d["ray_d_pool"].astype(np.float32, copy=False),
            "ray_hit_pool": self.d["ray_hit_pool"].astype(np.float32, copy=False),
            "ray_t_pool": self.d["ray_t_pool"].astype(np.float32, copy=False),
            "ray_n_pool": self.d["ray_n_pool"].astype(np.float32, copy=False),
            "ray_available": True,
        }


class PointCloudNoRayBackend(PointCloudBackend):
    """PointCloud backend that explicitly marks ray modality as unavailable."""

    def get_pools(self):
        out = super().get_pools()
        out["ray_available"] = False
        return out