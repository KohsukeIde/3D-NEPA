import numpy as np


class UDFGridBackend:
    """Distance-field (UDF) backend.

    This backend exposes point-query distances computed from a cached unsigned
    distance field (UDF) grid. It intentionally marks ray modality as unavailable
    (ray_available=False) so the tokenizer will emit TYPE_MISSING_RAY for ray
    tokens.

    Expected npz keys (produced by preprocess_modelnet40.py with --df_grid > 0):
      - pt_xyz_pool: (P,3) float32
      - pt_dist_udf_pool: (P,) float32  (preferred)
      - udf_grid: (G,G,G) float16/float32 (optional fallback)
      - ray_o_pool, ray_d_pool, ray_hit_pool, ray_t_pool, ray_n_pool: for shape/compat
    """

    def __init__(self, npz_path: str):
        self.d = np.load(npz_path, allow_pickle=False)

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)

        if "pt_dist_udf_pool" in self.d:
            pt_dist = self.d["pt_dist_udf_pool"].astype(np.float32, copy=False)
        else:
            # Fallback (should not happen in full experiments):
            # use the mesh-derived distances if present.
            pt_dist = self.d["pt_dist_pool"].astype(np.float32, copy=False)

        # Rays are present only to keep the Query->Answer API shape-compatible.
        # They will be ignored by tokenizer when ray_available=False.
        ray_o = self.d["ray_o_pool"].astype(np.float32, copy=False)
        ray_d = self.d["ray_d_pool"].astype(np.float32, copy=False)

        # Provide zeros for answers (safe even if original pools exist).
        m = ray_o.shape[0]
        ray_hit = np.zeros((m,), dtype=np.float32)
        ray_t = np.zeros((m,), dtype=np.float32)
        ray_n = np.zeros((m, 3), dtype=np.float32)

        pools = {
            "pt_xyz_pool": pt_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": ray_o,
            "ray_d_pool": ray_d,
            "ray_hit_pool": ray_hit,
            "ray_t_pool": ray_t,
            "ray_n_pool": ray_n,
            "ray_available": False,
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
