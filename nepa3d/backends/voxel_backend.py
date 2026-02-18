import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import cKDTree


def _world_to_grid_idx(xyz, grid, bmin=-1.0, bmax=1.0):
    voxel = (bmax - bmin) / float(grid)
    idx = np.floor((xyz - bmin) / voxel).astype(np.int32)
    return np.clip(idx, 0, grid - 1)


def _build_occ_grid(pc_xyz, grid=64, dilate=1):
    idx = _world_to_grid_idx(pc_xyz, grid=grid)
    occ = np.zeros((grid, grid, grid), dtype=np.bool_)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate > 0:
        occ = binary_dilation(occ, iterations=int(dilate))
    return occ


def _ray_aabb_intersect_batch(ray_o, ray_d, bmin=-1.0, bmax=1.0, eps=1e-9):
    d = np.where(np.abs(ray_d) < eps, np.sign(ray_d) * eps + (ray_d == 0.0) * eps, ray_d)
    inv = 1.0 / d
    t0 = (bmin - ray_o) * inv
    t1 = (bmax - ray_o) * inv
    tmin = np.max(np.minimum(t0, t1), axis=1)
    tmax = np.min(np.maximum(t0, t1), axis=1)
    t_entry = np.maximum(tmin, 0.0)
    hit = tmax >= t_entry
    return hit, t_entry.astype(np.float32), tmax.astype(np.float32)


def _dda_first_hit(occ, o, d, t_start, t_end, grid=64, max_steps=0, bmin=-1.0, bmax=1.0, eps=1e-9):
    if max_steps <= 0:
        max_steps = 3 * int(grid)
    voxel = (bmax - bmin) / float(grid)
    p = o + t_start * d

    ix, iy, iz = _world_to_grid_idx(p[None, :], grid=grid, bmin=bmin, bmax=bmax)[0]
    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])

    # Explicit axes keep this stable/readable.
    if abs(dx) < eps:
        sx, tx_max, tx_delta = 0, float("inf"), float("inf")
    else:
        sx = 1 if dx > 0 else -1
        nx = bmin + (ix + 1) * voxel if dx > 0 else bmin + ix * voxel
        tx_max = float(t_start + (nx - p[0]) / dx)
        tx_delta = float(voxel / abs(dx))

    if abs(dy) < eps:
        sy, ty_max, ty_delta = 0, float("inf"), float("inf")
    else:
        sy = 1 if dy > 0 else -1
        ny = bmin + (iy + 1) * voxel if dy > 0 else bmin + iy * voxel
        ty_max = float(t_start + (ny - p[1]) / dy)
        ty_delta = float(voxel / abs(dy))

    if abs(dz) < eps:
        sz, tz_max, tz_delta = 0, float("inf"), float("inf")
    else:
        sz = 1 if dz > 0 else -1
        nz = bmin + (iz + 1) * voxel if dz > 0 else bmin + iz * voxel
        tz_max = float(t_start + (nz - p[2]) / dz)
        tz_delta = float(voxel / abs(dz))

    t = float(t_start)
    g = int(grid)
    for _ in range(int(max_steps)):
        if occ[ix, iy, iz]:
            return True, np.float32(t)
        if tx_max < ty_max and tx_max < tz_max:
            t = tx_max
            tx_max += tx_delta
            ix += sx
        elif ty_max < tz_max:
            t = ty_max
            ty_max += ty_delta
            iy += sy
        else:
            t = tz_max
            tz_max += tz_delta
            iz += sz

        if t > float(t_end):
            break
        if ix < 0 or ix >= g or iy < 0 or iy >= g or iz < 0 or iz >= g:
            break
    return False, np.float32(0.0)


class VoxelBackend:
    """Voxelized backend built from cached point cloud samples in each .npz.

    This backend keeps the same Query->Answer interface but routes both point and
    ray answers through a voxel occupancy representation.
    """

    def __init__(self, npz_path, grid=64, dilate=1, max_steps=0):
        self.d = np.load(npz_path, allow_pickle=False)
        self.grid = int(grid)
        self.dilate = int(dilate)
        self.max_steps = int(max_steps)

        self.pc_xyz = self.d["pc_xyz"].astype(np.float32, copy=False)
        self.pc_n = self.d["pc_n"].astype(np.float32, copy=False)
        self.occ = _build_occ_grid(self.pc_xyz, grid=self.grid, dilate=self.dilate)
        self.voxel = 2.0 / float(self.grid)
        self.dist_vox = distance_transform_edt(~self.occ).astype(np.float32)
        self.kdt = cKDTree(self.pc_xyz)

    def _point_distance_from_grid(self, pt_xyz):
        idx = _world_to_grid_idx(pt_xyz, grid=self.grid)
        d = self.dist_vox[idx[:, 0], idx[:, 1], idx[:, 2]] * self.voxel
        return d.astype(np.float32, copy=False)

    def _ray_answers_from_grid(self, ray_o, ray_d):
        m = ray_o.shape[0]
        ray_hit = np.zeros((m,), dtype=np.float32)
        ray_t = np.zeros((m,), dtype=np.float32)
        ray_n = np.zeros((m, 3), dtype=np.float32)

        hit, t_entry, t_exit = _ray_aabb_intersect_batch(ray_o, ray_d)
        for i in range(m):
            if not hit[i]:
                continue
            ok, th = _dda_first_hit(
                self.occ,
                ray_o[i],
                ray_d[i],
                t_start=float(t_entry[i]),
                t_end=float(t_exit[i]),
                grid=self.grid,
                max_steps=self.max_steps,
            )
            if ok:
                ray_hit[i] = 1.0
                ray_t[i] = th

        idx = np.where(ray_hit > 0.5)[0]
        if idx.size > 0:
            pts = ray_o[idx] + ray_t[idx][:, None] * ray_d[idx]
            _, nn_idx = self.kdt.query(pts, k=1)
            ray_n[idx] = self.pc_n[nn_idx].astype(np.float32, copy=False)

        return ray_hit, ray_t, ray_n

    def get_pools(self):
        pt_xyz = self.d["pt_xyz_pool"].astype(np.float32, copy=False)
        pt_dist = self._point_distance_from_grid(pt_xyz)

        ray_o = self.d["ray_o_pool"].astype(np.float32, copy=False)
        ray_d = self.d["ray_d_pool"].astype(np.float32, copy=False)
        ray_hit, ray_t, ray_n = self._ray_answers_from_grid(ray_o, ray_d)

        pools = {
            "pt_xyz_pool": pt_xyz,
            "pt_dist_pool": pt_dist,
            "ray_o_pool": ray_o,
            "ray_d_pool": ray_d,
            "ray_hit_pool": ray_hit,
            "ray_t_pool": ray_t,
            "ray_n_pool": ray_n,
            "ray_available": True,
        }
        if "pt_fps_order" in self.d:
            pools["pt_fps_order"] = self.d["pt_fps_order"].astype(np.int32, copy=False)
        return pools
