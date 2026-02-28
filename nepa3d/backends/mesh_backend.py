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
        return pools
