import numpy as np


class MeshBackend:
    def __init__(self, npz_path):
        self.d = np.load(npz_path, allow_pickle=False)

    def get_pools(self):
        pools = {
            "pt_xyz_pool": self.d["pt_xyz_pool"].astype(np.float32, copy=False),
            "pt_dist_pool": self.d["pt_dist_pool"].astype(np.float32, copy=False),
            "ray_o_pool": self.d["ray_o_pool"].astype(np.float32, copy=False),
            "ray_d_pool": self.d["ray_d_pool"].astype(np.float32, copy=False),
            "ray_hit_pool": self.d["ray_hit_pool"].astype(np.float32, copy=False),
            "ray_t_pool": self.d["ray_t_pool"].astype(np.float32, copy=False),
            "ray_n_pool": self.d["ray_n_pool"].astype(np.float32, copy=False),
            "ray_available": True,
        }
        if "pt_fps_order" in self.d:
            pools["pt_fps_order"] = self.d["pt_fps_order"].astype(np.int32, copy=False)
        return pools
