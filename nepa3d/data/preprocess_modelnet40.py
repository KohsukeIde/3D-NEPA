import argparse
import glob
import multiprocessing as mp
import os
import zipfile

import numpy as np
import trimesh
from tqdm import tqdm

import math
from nepa3d.utils.fps import fps_order

try:
    from scipy.ndimage import binary_dilation, distance_transform_edt
except Exception:
    binary_dilation = None
    distance_transform_edt = None

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _dilate_occ(occ, iterations=1):
    if iterations <= 0:
        return occ
    if binary_dilation is not None:
        return binary_dilation(occ, iterations=iterations)
    # fallback: naive 3x3x3 OR dilation
    out = occ.copy()
    for _ in range(iterations):
        nxt = out.copy()
        for dx in (-1, 0, 1):
            xs = slice(max(0, dx), out.shape[0] + min(0, dx))
            xt = slice(max(0, -dx), out.shape[0] + min(0, -dx))
            for dy in (-1, 0, 1):
                ys = slice(max(0, dy), out.shape[1] + min(0, dy))
                yt = slice(max(0, -dy), out.shape[1] + min(0, -dy))
                for dz in (-1, 0, 1):
                    zs = slice(max(0, dz), out.shape[2] + min(0, dz))
                    zt = slice(max(0, -dz), out.shape[2] + min(0, -dz))
                    nxt[xt, yt, zt] |= out[xs, ys, zs]
        out = nxt
    return out


def build_occ_grid_from_points(pc_xyz, grid=64, dilate=1, bmin=-1.0, bmax=1.0):
    """Build occupancy grid from point cloud samples in world coordinates (assumed normalized to [-1,1])."""
    pc = pc_xyz.astype(np.float32, copy=False)
    voxel = (bmax - bmin) / float(grid)
    idx = np.floor((pc - bmin) / voxel).astype(np.int32)
    idx = np.clip(idx, 0, grid - 1)
    occ = np.zeros((grid, grid, grid), dtype=np.bool_)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if dilate > 0:
        occ = _dilate_occ(occ, iterations=dilate)
    return occ




def trilinear_sample_grid(grid, xyz, bmin=-1.0, bmax=1.0):
    """Trilinear sampling of a (G,G,G) grid defined at voxel centers.

    Grid voxel centers are assumed at:
      p(i) = bmin + (i + 0.5) * voxel, voxel=(bmax-bmin)/G
    """
    G = int(grid.shape[0])
    voxel = (bmax - bmin) / float(G)
    # map xyz -> fractional voxel-center index
    s = (xyz - bmin) / voxel - 0.5
    i0 = np.floor(s).astype(np.int32)
    w = (s - i0.astype(np.float32)).astype(np.float32)
    i1 = i0 + 1

    i0 = np.clip(i0, 0, G - 1)
    i1 = np.clip(i1, 0, G - 1)

    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    g000 = grid[i0[:, 0], i0[:, 1], i0[:, 2]]
    g100 = grid[i1[:, 0], i0[:, 1], i0[:, 2]]
    g010 = grid[i0[:, 0], i1[:, 1], i0[:, 2]]
    g110 = grid[i1[:, 0], i1[:, 1], i0[:, 2]]
    g001 = grid[i0[:, 0], i0[:, 1], i1[:, 2]]
    g101 = grid[i1[:, 0], i0[:, 1], i1[:, 2]]
    g011 = grid[i0[:, 0], i1[:, 1], i1[:, 2]]
    g111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]

    c00 = g000 * (1.0 - wx) + g100 * wx
    c10 = g010 * (1.0 - wx) + g110 * wx
    c01 = g001 * (1.0 - wx) + g101 * wx
    c11 = g011 * (1.0 - wx) + g111 * wx

    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy

    c = c0 * (1.0 - wz) + c1 * wz
    return c.astype(np.float32, copy=False)

def ray_aabb_intersect_batch(ray_o, ray_d, bmin=-1.0, bmax=1.0, eps=1e-9):
    """Slab method. Returns hit_mask, t_entry, t_exit."""
    o = ray_o
    d = ray_d
    inv = 1.0 / np.where(np.abs(d) < eps, np.sign(d) * eps + (d == 0.0) * eps, d)
    t0 = (bmin - o) * inv
    t1 = (bmax - o) * inv
    tmin = np.max(np.minimum(t0, t1), axis=1)
    tmax = np.min(np.maximum(t0, t1), axis=1)
    t_entry = np.maximum(tmin, 0.0)
    hit = tmax >= t_entry
    return hit, t_entry.astype(np.float32), tmax.astype(np.float32)


def dda_first_hit(occ, o, d, t_start, t_end, grid=64, bmin=-1.0, bmax=1.0, max_steps=None, eps=1e-9):
    """3D DDA traversal in a dense occupancy grid. Returns (hit, t_hit)."""
    G = int(grid)
    if max_steps is None or max_steps <= 0:
        max_steps = 3 * G
    voxel = (bmax - bmin) / float(G)

    # start point at entry
    p = o + t_start * d

    ix = int(math.floor((p[0] - bmin) / voxel))
    iy = int(math.floor((p[1] - bmin) / voxel))
    iz = int(math.floor((p[2] - bmin) / voxel))
    ix = min(max(ix, 0), G - 1)
    iy = min(max(iy, 0), G - 1)
    iz = min(max(iz, 0), G - 1)

    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])

    if abs(dx) < eps:
        step_x = 0
        tMaxX = float('inf')
        tDeltaX = float('inf')
    else:
        step_x = 1 if dx > 0.0 else -1
        next_x = bmin + (ix + 1) * voxel if dx > 0.0 else bmin + ix * voxel
        tMaxX = float(t_start + (next_x - p[0]) / dx)
        tDeltaX = float(voxel / abs(dx))

    if abs(dy) < eps:
        step_y = 0
        tMaxY = float('inf')
        tDeltaY = float('inf')
    else:
        step_y = 1 if dy > 0.0 else -1
        next_y = bmin + (iy + 1) * voxel if dy > 0.0 else bmin + iy * voxel
        tMaxY = float(t_start + (next_y - p[1]) / dy)
        tDeltaY = float(voxel / abs(dy))

    if abs(dz) < eps:
        step_z = 0
        tMaxZ = float('inf')
        tDeltaZ = float('inf')
    else:
        step_z = 1 if dz > 0.0 else -1
        next_z = bmin + (iz + 1) * voxel if dz > 0.0 else bmin + iz * voxel
        tMaxZ = float(t_start + (next_z - p[2]) / dz)
        tDeltaZ = float(voxel / abs(dz))

    t = float(t_start)

    for _ in range(int(max_steps)):
        if occ[ix, iy, iz]:
            return True, np.float32(t)

        # step to next voxel boundary
        if tMaxX < tMaxY and tMaxX < tMaxZ:
            t = tMaxX
            tMaxX += tDeltaX
            ix += step_x
        elif tMaxY < tMaxZ:
            t = tMaxY
            tMaxY += tDeltaY
            iy += step_y
        else:
            t = tMaxZ
            tMaxZ += tDeltaZ
            iz += step_z

        if t > float(t_end):
            break
        if ix < 0 or ix >= G or iy < 0 or iy >= G or iz < 0 or iz >= G:
            break

    return False, np.float32(0.0)


def render_rays_from_pointcloud_occ(ray_o, ray_d, pc_xyz, pc_n, grid=64, dilate=1, max_steps=None):
    """Approximate ray-mesh intersection by voxel traversal over pointcloud occupancy."""
    m = ray_o.shape[0]
    occ = build_occ_grid_from_points(pc_xyz, grid=grid, dilate=dilate)

    hit_mask, t_entry, t_exit = ray_aabb_intersect_batch(ray_o, ray_d)

    ray_hit = np.zeros((m,), dtype=np.float32)
    ray_t = np.zeros((m,), dtype=np.float32)
    ray_n = np.zeros((m, 3), dtype=np.float32)

    # build KDTree for normal lookup
    kdt = None
    if cKDTree is not None:
        kdt = cKDTree(pc_xyz.astype(np.float32, copy=False))

    for i in range(m):
        if not hit_mask[i]:
            continue
        ok, th = dda_first_hit(
            occ, ray_o[i], ray_d[i], float(t_entry[i]), float(t_exit[i]),
            grid=grid, max_steps=max_steps
        )
        if ok:
            ray_hit[i] = 1.0
            ray_t[i] = th

    if kdt is not None:
        idx = np.where(ray_hit > 0.5)[0]
        if idx.size > 0:
            pts = ray_o[idx] + ray_t[idx][:, None] * ray_d[idx]
            _, nn_idx = kdt.query(pts, k=1)
            ray_n[idx] = pc_n[nn_idx].astype(np.float32, copy=False)

    # sanitize
    bad_t = ~np.isfinite(ray_t)
    bad_n = ~np.isfinite(ray_n).all(axis=1)
    bad = bad_t | bad_n
    if np.any(bad):
        ray_hit[bad] = 0.0
        ray_t[bad] = 0.0
        ray_n[bad] = 0.0

    return ray_hit, ray_t, ray_n


def normalize_mesh(mesh):
    v = mesh.vertices
    center = (v.max(axis=0) + v.min(axis=0)) * 0.5
    v = v - center
    scale = np.max(np.linalg.norm(v, axis=1))
    if scale < 1e-9:
        scale = 1.0
    v = v / scale
    mesh.vertices = v
    return mesh


def closest_point_distance_chunked(mesh, xyz, chunk_size=2048):
    """Compute unsigned point-to-mesh distances in chunks to cap peak memory."""
    xyz = xyz.astype(np.float32, copy=False)
    n = xyz.shape[0]
    out = np.empty((n,), dtype=np.float32)
    cs = max(1, int(chunk_size))
    for s in range(0, n, cs):
        e = min(n, s + cs)
        _, dist, _ = trimesh.proximity.closest_point(mesh, xyz[s:e])
        out[s:e] = dist.astype(np.float32, copy=False)
    return out


def nearest_distance_to_points(query_xyz, ref_xyz, chunk_size=2048):
    """Unsigned nearest-neighbor distance from query points to reference points."""
    q = query_xyz.astype(np.float32, copy=False)
    r = ref_xyz.astype(np.float32, copy=False)
    if q.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if r.size == 0:
        return np.ones((q.shape[0],), dtype=np.float32)

    if cKDTree is not None:
        kdt = cKDTree(r)
        dist, _ = kdt.query(q, k=1)
        return dist.astype(np.float32, copy=False)

    out = np.empty((q.shape[0],), dtype=np.float32)
    cs = max(1, int(chunk_size))
    for s in range(0, q.shape[0], cs):
        e = min(q.shape[0], s + cs)
        diff = q[s:e, None, :] - r[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        out[s:e] = np.sqrt(np.min(d2, axis=1)).astype(np.float32)
    return out


def approx_point_distance_kdtree(mesh, xyz, ref_points=8192, fallback_pc=None):
    """Approximate point-to-surface distance using nearest sampled surface points."""
    if cKDTree is None:
        raise RuntimeError("cKDTree is unavailable")
    ref_n = max(1, int(ref_points))
    if fallback_pc is not None and fallback_pc.shape[0] >= ref_n:
        ref = fallback_pc.astype(np.float32, copy=False)
    else:
        ref = mesh.sample(ref_n).astype(np.float32, copy=False)
    tree = cKDTree(ref)
    d, _ = tree.query(xyz.astype(np.float32, copy=False), k=1)
    return d.astype(np.float32, copy=False)


def sample_cameras(n_views, radius=2.2):
    az = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)
    el = np.linspace(-0.35 * np.pi, 0.35 * np.pi, n_views)
    cams = []
    for a, e in zip(az, el):
        x = radius * np.cos(e) * np.cos(a)
        y = radius * np.cos(e) * np.sin(a)
        z = radius * np.sin(e)
        cams.append(np.array([x, y, z], dtype=np.float32))
    return np.stack(cams, axis=0)


def look_at(cam_pos):
    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-9)
    tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(tmp, forward)
    if np.linalg.norm(right) < 1e-6:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(tmp, forward)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(forward, right)
    r = np.stack([right, up, forward], axis=0).astype(np.float32)
    return r


def sample_rays(cam_pos, n_rays, fov_deg=60.0):
    r = look_at(cam_pos)
    uv = np.random.uniform(-1.0, 1.0, size=(n_rays, 2)).astype(np.float32)
    fov = np.deg2rad(fov_deg)
    z = 1.0 / np.tan(fov * 0.5)
    dirs_cam = np.concatenate([uv, np.full((n_rays, 1), z, np.float32)], axis=1)
    dirs_cam = dirs_cam / (np.linalg.norm(dirs_cam, axis=1, keepdims=True) + 1e-9)
    dirs_world = (r.T @ dirs_cam.T).T
    dirs_world = dirs_world / (np.linalg.norm(dirs_world, axis=1, keepdims=True) + 1e-9)
    origins = np.repeat(cam_pos[None, :], n_rays, axis=0)
    return origins.astype(np.float32), dirs_world.astype(np.float32)


def preprocess_one(
    mesh_path,
    out_path,
    pc_points=2048,
    pt_pool=20000,
    ray_pool=8000,
    n_views=20,
    rays_per_view=400,
    seed=None,
    pc_grid=64,
    pc_dilate=1,
    pc_max_steps=0,
    compute_pc_rays=True,
    # UDF grid (distance field) settings
    df_grid=64,
    df_dilate=1,
    compute_udf=True,
    # point-query pool distribution
    pt_surface_ratio=0.5,
    pt_surface_sigma=0.02,
    pt_query_chunk=2048,
    ray_query_chunk=2048,
    pt_dist_mode="mesh",
    dist_ref_points=8192,
):

    if seed is not None:
        np.random.seed(seed)

    try:
        geom = trimesh.load(mesh_path, process=False)
    except Exception:
        return False
    mesh = None
    pc_src = None

    if isinstance(geom, trimesh.Trimesh):
        mesh = geom
        if len(mesh.vertices) > 0:
            pc_src = np.asarray(mesh.vertices).astype(np.float32, copy=False)
    elif isinstance(geom, trimesh.Scene):
        # Scene fallback: merge all available vertices as point cloud.
        verts = []
        for g in geom.geometry.values():
            if hasattr(g, "vertices"):
                v = np.asarray(g.vertices)
                if v.size > 0:
                    verts.append(v)
        if verts:
            pc_src = np.concatenate(verts, axis=0).astype(np.float32, copy=False)
    elif hasattr(geom, "vertices"):
        v = np.asarray(geom.vertices)
        if v.size > 0:
            pc_src = v.astype(np.float32, copy=False)

    has_mesh = mesh is not None and len(getattr(mesh, "vertices", [])) > 0 and len(getattr(mesh, "faces", [])) > 0
    if has_mesh:
        try:
            mesh.remove_degenerate_faces()
            mesh.remove_duplicate_faces()
            mesh.remove_infinite_values()
            mesh.remove_unreferenced_vertices()
        except Exception:
            pass

        mesh = normalize_mesh(mesh)
        pc_xyz, face_idx = mesh.sample(pc_points, return_index=True)
        pc_xyz = pc_xyz.astype(np.float32, copy=False)
        if face_idx is None:
            pc_n = np.zeros_like(pc_xyz, dtype=np.float32)
        else:
            pc_n = mesh.face_normals[face_idx].astype(np.float32, copy=False)
    else:
        if pc_src is None or pc_src.shape[0] == 0:
            return False
        # Normalize point-only geometry to canonical cube, same convention as mesh path.
        center = (pc_src.max(axis=0) + pc_src.min(axis=0)) * 0.5
        pc_src = pc_src - center
        scale = np.max(np.linalg.norm(pc_src, axis=1))
        if scale < 1e-9:
            scale = 1.0
        pc_src = (pc_src / scale).astype(np.float32, copy=False)

        idx = np.random.choice(pc_src.shape[0], size=int(pc_points), replace=(pc_src.shape[0] < int(pc_points)))
        pc_xyz = pc_src[idx].astype(np.float32, copy=False)
        pc_n = np.zeros_like(pc_xyz, dtype=np.float32)
    # Deterministic FPS order over observed point cloud points.
    try:
        pc_fps_order = fps_order(pc_xyz, k=int(pc_xyz.shape[0])).astype(np.int32, copy=False)
    except Exception:
        pc_fps_order = np.arange(pc_xyz.shape[0], dtype=np.int32)

    # Point-query pool: mix uniform-in-cube and near-surface samples.
    pt_pool = int(pt_pool)
    n_uni = int(round(float(1.0 - pt_surface_ratio) * pt_pool))
    n_surf = pt_pool - n_uni
    pts = []
    if n_uni > 0:
        pts.append(np.random.uniform(-1.0, 1.0, size=(n_uni, 3)).astype(np.float32))
    if n_surf > 0:
        # Sample near surface by jittering mesh-sampled points.
        base = pc_xyz[np.random.choice(pc_xyz.shape[0], size=n_surf, replace=(pc_xyz.shape[0] < n_surf))]
        jitter = np.random.normal(scale=float(pt_surface_sigma), size=(n_surf, 3)).astype(np.float32)
        pts.append((base + jitter).astype(np.float32))
    pt_xyz_pool = np.concatenate(pts, axis=0) if len(pts) > 1 else pts[0]
    # Clip to canonical cube
    pt_xyz_pool = np.clip(pt_xyz_pool, -1.0, 1.0).astype(np.float32, copy=False)
    if has_mesh:
        if pt_dist_mode == "kdtree":
            try:
                pt_dist_pool = approx_point_distance_kdtree(
                    mesh, pt_xyz_pool, ref_points=dist_ref_points, fallback_pc=pc_xyz
                )
            except Exception:
                pt_dist_pool = closest_point_distance_chunked(
                    mesh, pt_xyz_pool, chunk_size=pt_query_chunk
                )
        else:
            pt_dist_pool = closest_point_distance_chunked(mesh, pt_xyz_pool, chunk_size=pt_query_chunk)
    else:
        # Point-only fallback: use nearest-neighbor distance to observed points.
        pt_dist_pool = nearest_distance_to_points(pt_xyz_pool, pc_xyz, chunk_size=pt_query_chunk)
    if not np.isfinite(pt_dist_pool).all():
        finite = np.isfinite(pt_dist_pool)
        fill = float(np.max(pt_dist_pool[finite])) if finite.any() else 1.0
        pt_dist_pool = np.nan_to_num(
            pt_dist_pool, nan=fill, posinf=fill, neginf=0.0
        )


    # PointCloud distance pool: distance from query points to observed point cloud.
    # This is used by PointCloudBackend to avoid leaking mesh-derived distances via pt_dist_pool.
    # NOTE: pt_dist_pool is mesh-derived (explicit surface). pt_dist_pc_pool is observation-derived.
    pt_dist_pc_pool = None
    try:
        pt_dist_pc_pool = nearest_distance_to_points(pt_xyz_pool, pc_xyz, chunk_size=pt_query_chunk)
    except Exception:
        pt_dist_pc_pool = None
    if pt_dist_pc_pool is None:
        # As a last resort, mark as invalid (-1). PointCloudBackend will recompute if needed.
        pt_dist_pc_pool = -np.ones((pt_xyz_pool.shape[0],), dtype=np.float32)
    if not np.isfinite(pt_dist_pc_pool).all():
        finite = np.isfinite(pt_dist_pc_pool)
        fill = float(np.max(pt_dist_pc_pool[finite])) if finite.any() else 1.0
        pt_dist_pc_pool = np.nan_to_num(pt_dist_pc_pool, nan=fill, posinf=fill, neginf=0.0).astype(np.float32)

    # UDF grid (unsigned distance field) computed from voxelized surface occupancy.
    udf_grid = None
    pt_dist_udf_pool = None
    occ_grid = None
    if compute_udf and int(df_grid) > 0 and distance_transform_edt is not None:
        try:
            occ_grid = build_occ_grid_from_points(pc_xyz, grid=int(df_grid), dilate=int(df_dilate))
            voxel = 2.0 / float(int(df_grid))
            # distance to nearest occupied voxel center, returned in world units via sampling
            udf_grid = distance_transform_edt(~occ_grid, sampling=(voxel, voxel, voxel)).astype(np.float32)
            # Precompute UDF distances at the point-query pool locations for speed.
            pt_dist_udf_pool = trilinear_sample_grid(udf_grid, pt_xyz_pool)
            # sanitize
            if not np.isfinite(pt_dist_udf_pool).all():
                finite = np.isfinite(pt_dist_udf_pool)
                fill = float(np.max(pt_dist_udf_pool[finite])) if finite.any() else 1.0
                pt_dist_udf_pool = np.nan_to_num(pt_dist_udf_pool, nan=fill, posinf=fill, neginf=0.0).astype(np.float32)
        except Exception:
            udf_grid = None
            pt_dist_udf_pool = None
            occ_grid = None

    rays_per_view = max(rays_per_view, int(np.ceil(ray_pool / max(n_views, 1))))
    ray_o_list, ray_d_list = [], []
    cams = sample_cameras(n_views)
    for c in cams:
        o, d = sample_rays(c, rays_per_view)
        ray_o_list.append(o)
        ray_d_list.append(d)

    ray_o = np.concatenate(ray_o_list, axis=0)[:ray_pool]
    ray_d = np.concatenate(ray_d_list, axis=0)[:ray_pool]
    m = ray_o.shape[0]

    ray_hit = np.zeros((m,), dtype=np.float32)
    ray_t = np.zeros((m,), dtype=np.float32)
    ray_n = np.zeros((m, 3), dtype=np.float32)
    if has_mesh:
        # Use Embree intersector if available, otherwise fall back to triangle intersector.
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector

            intersector = RayMeshIntersector(mesh)
        except Exception:
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

        rcs = max(1, int(ray_query_chunk))
        for s in range(0, m, rcs):
            e = min(m, s + rcs)
            loc, idx_ray, idx_tri = intersector.intersects_location(
                ray_o[s:e], ray_d[s:e], multiple_hits=False
            )
            if len(idx_ray) == 0:
                continue
            idx = idx_ray + s
            ray_hit[idx] = 1.0
            ray_t[idx] = np.sum((loc - ray_o[idx]) * ray_d[idx], axis=1).astype(np.float32)
            ray_n[idx] = mesh.face_normals[idx_tri].astype(np.float32)
        bad_t = ~np.isfinite(ray_t)
        bad_n = ~np.isfinite(ray_n).all(axis=1)
        bad = bad_t | bad_n
        if np.any(bad):
            ray_hit[bad] = 0.0
            ray_t[bad] = 0.0
            ray_n[bad] = 0.0
    else:
        # Point-only fallback: approximate "mesh-ray" pools from occupancy traversal.
        try:
            ray_hit, ray_t, ray_n = render_rays_from_pointcloud_occ(
                ray_o, ray_d, pc_xyz, pc_n, grid=pc_grid, dilate=pc_dilate, max_steps=pc_max_steps
            )
        except Exception:
            pass

    
    # PointCloud ray pools (approx): traverse voxelized pointcloud occupancy with DDA.
    ray_hit_pc = None
    ray_t_pc = None
    ray_n_pc = None
    if compute_pc_rays:
        try:
            if has_mesh:
                ray_hit_pc, ray_t_pc, ray_n_pc = render_rays_from_pointcloud_occ(
                    ray_o, ray_d, pc_xyz, pc_n, grid=pc_grid, dilate=pc_dilate, max_steps=pc_max_steps
                )
            else:
                # In point-only mode, reuse already computed occupancy-based rays.
                ray_hit_pc = ray_hit.copy()
                ray_t_pc = ray_t.copy()
                ray_n_pc = ray_n.copy()
        except Exception:
            # fall back to zeros so downstream can handle.
            ray_hit_pc = np.zeros_like(ray_hit, dtype=np.float32)
            ray_t_pc = np.zeros_like(ray_t, dtype=np.float32)
            ray_n_pc = np.zeros_like(ray_n, dtype=np.float32)
    else:
        ray_hit_pc = np.zeros_like(ray_hit, dtype=np.float32)
        ray_t_pc = np.zeros_like(ray_t, dtype=np.float32)
        ray_n_pc = np.zeros_like(ray_n, dtype=np.float32)

    # Ensure UDF keys exist for downstream backends (even if compute_udf=False).
    if pt_dist_udf_pool is None:
        pt_dist_udf_pool = pt_dist_pool.astype(np.float32, copy=False)
    if udf_grid is None:
        udf_grid = np.zeros((1, 1, 1), dtype=np.float16)
    else:
        # store compactly
        udf_grid = udf_grid.astype(np.float16, copy=False)
    if occ_grid is None:
        occ_grid = np.zeros((1, 1, 1), dtype=np.uint8)
    else:
        occ_grid = occ_grid.astype(np.uint8, copy=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        pc_xyz=pc_xyz,
        pc_n=pc_n,
        pc_fps_order=pc_fps_order,
        pt_xyz_pool=pt_xyz_pool,
        pt_dist_pool=pt_dist_pool,
        pt_dist_pc_pool=pt_dist_pc_pool,
        pt_dist_udf_pool=pt_dist_udf_pool,
        udf_grid=udf_grid,
        occ_grid=occ_grid,
        df_grid=np.array([int(df_grid)], dtype=np.int32),
        df_dilate=np.array([int(df_dilate)], dtype=np.int32),
        ray_o_pool=ray_o,
        ray_d_pool=ray_d,
        ray_hit_pool=ray_hit,
        ray_t_pool=ray_t,
        ray_n_pool=ray_n,
        ray_hit_pc_pool=ray_hit_pc,
        ray_t_pc_pool=ray_t_pc,
        ray_n_pc_pool=ray_n_pc,
    )
    return True


def _worker(task):
    (
        mesh_path,
        out_path,
        pc_points,
        pt_pool,
        ray_pool,
        n_views,
        rays_per_view,
        seed,
        pc_grid,
        pc_dilate,
        pc_max_steps,
        compute_pc_rays,
        df_grid,
        df_dilate,
        compute_udf,
        pt_surface_ratio,
        pt_surface_sigma,
        pt_query_chunk,
        ray_query_chunk,
        pt_dist_mode,
        dist_ref_points,
    ) = task
    try:
        ok = preprocess_one(
            mesh_path,
            out_path,
            pc_points=pc_points,
            pt_pool=pt_pool,
            ray_pool=ray_pool,
            n_views=n_views,
            rays_per_view=rays_per_view,
            seed=seed,
            pc_grid=pc_grid,
            pc_dilate=pc_dilate,
            pc_max_steps=pc_max_steps,
            compute_pc_rays=compute_pc_rays,
            df_grid=df_grid,
            df_dilate=df_dilate,
            compute_udf=compute_udf,
            pt_surface_ratio=pt_surface_ratio,
            pt_surface_sigma=pt_surface_sigma,
            pt_query_chunk=pt_query_chunk,
            ray_query_chunk=ray_query_chunk,
            pt_dist_mode=pt_dist_mode,
            dist_ref_points=dist_ref_points,
        )
    except Exception:
        ok = False
    return (mesh_path, ok)


def discover_modelnet_meshes(modelnet_root, split, mesh_glob=""):
    """Return (paths, source_pattern) for supported ModelNet layouts.

    Supported auto layouts:
      1) <root>/<class>/<split>/*.off          (official ModelNet40)
      2) <root>/<split>/<class>/*.off          (split-first off tree)
      3) <root>/ply_format/<split>/<class>/*.ply
      4) <root>/<split>/<class>/*.ply          (split-first ply tree)
    """
    if mesh_glob:
        pattern = mesh_glob if os.path.isabs(mesh_glob) else os.path.join(modelnet_root, mesh_glob)
        return sorted(glob.glob(pattern)), pattern

    patterns = [
        os.path.join(modelnet_root, "*", split, "*.off"),
        os.path.join(modelnet_root, split, "*", "*.off"),
        os.path.join(modelnet_root, "ply_format", split, "*", "*.ply"),
        os.path.join(modelnet_root, split, "*", "*.ply"),
    ]
    for pattern in patterns:
        paths = sorted(glob.glob(pattern))
        if paths:
            return paths, pattern
    return [], patterns[0]


def infer_class_from_mesh_path(mesh_path, split):
    """Infer class name from mesh path robustly across supported layouts."""
    parts = os.path.normpath(mesh_path).split(os.sep)
    # Prefer split-anchor rule: .../<split>/<class>/<file>
    for i, p in enumerate(parts):
        if p == split and i + 1 < len(parts) - 1:
            return parts[i + 1]
    # Fallback for official layout: .../<class>/<split>/<file>
    if len(parts) >= 3 and parts[-2] == split:
        return parts[-3]
    # Last safe fallback: parent directory
    if len(parts) >= 2:
        return parts[-2]
    raise ValueError(f"cannot infer class from mesh path: {mesh_path}")


def is_valid_npz(path):
    """Return True if path is a readable NPZ(zip) file with intact central dir/CRC."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
        return bad is None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelnet_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "test"], required=True)
    ap.add_argument(
        "--mesh_glob",
        type=str,
        default="",
        help="Optional mesh glob (absolute or relative to modelnet_root). Empty uses auto-discovery for off/ply layouts.",
    )
    ap.add_argument("--pc_points", type=int, default=2048)
    ap.add_argument("--pt_pool", type=int, default=20000)
    ap.add_argument("--ray_pool", type=int, default=8000)
    ap.add_argument("--n_views", type=int, default=20)
    ap.add_argument("--rays_per_view", type=int, default=400)
    ap.add_argument("--pc_grid", type=int, default=64, help="occupancy grid resolution for pointcloud ray-march")
    ap.add_argument("--pc_dilate", type=int, default=1, help="binary dilation iterations for occupancy grid")
    ap.add_argument("--pc_max_steps", type=int, default=0, help="max DDA steps (0=auto)")
    ap.add_argument("--no_pc_rays", action="store_true", help="disable pointcloud ray-march pool generation")
    ap.add_argument("--df_grid", type=int, default=64, help="UDF grid resolution (0 to disable)")
    ap.add_argument("--df_dilate", type=int, default=1, help="dilation iterations for UDF occupancy grid")
    ap.add_argument("--no_udf", action="store_true", help="disable UDF grid/pt_dist_udf_pool generation")
    ap.add_argument("--pt_surface_ratio", type=float, default=0.5, help="fraction of point-query pool sampled near surface")
    ap.add_argument("--pt_surface_sigma", type=float, default=0.02, help="std of Gaussian jitter for near-surface queries")
    ap.add_argument("--pt_query_chunk", type=int, default=2048, help="chunk size for point-to-mesh distance queries")
    ap.add_argument("--ray_query_chunk", type=int, default=2048, help="chunk size for ray intersection queries")
    ap.add_argument(
        "--pt_dist_mode",
        type=str,
        choices=["mesh", "kdtree"],
        default="mesh",
        help="point-distance source: exact mesh closest-point or KDTree approximation",
    )
    ap.add_argument(
        "--dist_ref_points",
        type=int,
        default=8192,
        help="number of sampled surface points for pt_dist_mode=kdtree",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="set -1 to disable deterministic seeding"
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument(
        "--max_tasks_per_child",
        type=int,
        default=0,
        help="restart worker after N tasks to mitigate memory growth (0=disabled)",
    )
    args = ap.parse_args()

    seed_base = None if args.seed < 0 else args.seed
    workers = max(1, args.workers)
    chunk_size = max(1, args.chunk_size)
    max_tasks_per_child = None if args.max_tasks_per_child <= 0 else int(args.max_tasks_per_child)

    paths, source_pattern = discover_modelnet_meshes(
        args.modelnet_root,
        args.split,
        mesh_glob=args.mesh_glob,
    )
    print(
        f"[discover] split={args.split} source_pattern={source_pattern} "
        f"num_meshes={len(paths)}"
    )
    if not paths:
        print("[discover] no meshes found. check --modelnet_root/--mesh_glob/layout.")
        return

    tasks = []
    skip_count = 0
    regenerate_count = 0
    for i, mesh_path in enumerate(paths):
        cls = infer_class_from_mesh_path(mesh_path, args.split)
        name = os.path.splitext(os.path.basename(mesh_path))[0]
        out_path = os.path.join(args.out_root, args.split, cls, f"{name}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            if is_valid_npz(out_path):
                skip_count += 1
                continue
            regenerate_count += 1
            print(f"[warn] invalid existing npz -> regenerate: {out_path}")
        seed = None if seed_base is None else seed_base + i
        tasks.append(
            (
                mesh_path,
                out_path,
                args.pc_points,
                args.pt_pool,
                args.ray_pool,
                args.n_views,
                args.rays_per_view,
                seed,
                args.pc_grid,
                args.pc_dilate,
                args.pc_max_steps,
                (not args.no_pc_rays),
                args.df_grid,
                args.df_dilate,
                (not args.no_udf),
                args.pt_surface_ratio,
                args.pt_surface_sigma,
                args.pt_query_chunk,
                args.ray_query_chunk,
                args.pt_dist_mode,
                args.dist_ref_points,
            )
        )

    if skip_count > 0:
        print(f"skip existing: {skip_count}")
    if regenerate_count > 0:
        print(f"regenerate invalid existing: {regenerate_count}")

    if not tasks:
        print("nothing to do")
        return

    if workers == 1:
        for task in tqdm(tasks, desc=f"preprocess {args.split}"):
            mesh_path, ok = _worker(task)
            if not ok:
                print(f"skip: {mesh_path}")
    else:
        with mp.Pool(processes=workers, maxtasksperchild=max_tasks_per_child) as pool:
            for mesh_path, ok in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=chunk_size),
                total=len(tasks),
                desc=f"preprocess {args.split} [w={workers}]",
            ):
                if not ok:
                    print(f"skip: {mesh_path}")


if __name__ == "__main__":
    main()
