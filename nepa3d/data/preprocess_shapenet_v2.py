"""Preprocess ShapeNet meshes into the v2 NPZ format.

This is a *PatchNEPA-oriented* data generator.

Key idea
--------
Store **surface context points** separately from **primitive-specific query points**.

One NPZ (one shape) can contain multiple query sets:
- mesh queries: `mesh_qry_xyz` + normals/curvature proxy
- mesh visibility answers: `mesh_qry_vis_hit` + `mesh_qry_vis_t`
- udf queries:  `udf_qry_xyz`  + unsigned distance + gradient proxy
- pc context/query: `pc_xyz`, `pc_qry_xyz` + PCA normals + density proxy
- ray queries (optional): `ray_o`, `ray_d` + hit/t/normal

Unpaired pretraining
--------------------
Unpairedness is enforced by *splitting shapes into disjoint subsets per primitive*
(see `shapenet_unpaired_split.py` + `preprocess_shapenet_unpaired.py`).
This script intentionally writes a **single NPZ per shape** containing all fields;
which primitive is used during pretrain is decided by the dataset spec (keys
and answer schema).

This mirrors the original codebase design where one NPZ contained all modalities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import trimesh
except Exception as e:
    raise RuntimeError(
        "preprocess_shapenet_v2 requires trimesh. Install trimesh or run in the project env."
    ) from e

from tqdm import tqdm

from .preprocess_modelnet40 import (
    normalize_mesh,
    build_occ_grid_from_points,
    trilinear_sample_grid,
)
from .convert_npz_to_v2 import estimate_density_knn, estimate_normals_pca

try:
    from scipy.ndimage import distance_transform_edt  # type: ignore
except Exception:
    distance_transform_edt = None


def _infer_synset_model(mesh_path: str) -> Tuple[str, str]:
    # .../<synset>/<model>/models/model_normalized.obj
    parts = mesh_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return "unknown", os.path.splitext(os.path.basename(mesh_path))[0]
    synset = parts[-4]
    model_id = parts[-3]
    return synset, model_id


def _unit_vectors(rng: np.random.RandomState, n: int) -> np.ndarray:
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def _sample_surface(mesh: "trimesh.Trimesh", n: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
    # trimesh.sample.sample_surface uses global np.random by default; we want deterministic per-shape.
    # We implement by temporarily seeding.
    state = np.random.get_state()
    np.random.seed(rng.randint(0, 2**31 - 1))
    try:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    finally:
        np.random.set_state(state)
    return pts.astype(np.float32), face_idx.astype(np.int64)


def _fibonacci_hemisphere_dirs(n: int) -> np.ndarray:
    """Deterministic quasi-uniform hemisphere directions around +Z."""
    if int(n) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    k = np.arange(int(n), dtype=np.float32) + 0.5
    z = 1.0 - (k / float(n))
    z = np.clip(z, 0.0, 1.0)
    phi = (math.pi * (3.0 - math.sqrt(5.0))) * k
    r = np.sqrt(np.clip(1.0 - (z * z), 0.0, 1.0))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    dirs = np.stack([x, y, z], axis=1).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    return dirs


def _orthonormal_basis_from_normals(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return tangent/bitangent for each normal."""
    nn = np.asarray(n, dtype=np.float32)
    nn = nn / (np.linalg.norm(nn, axis=1, keepdims=True) + 1e-8)
    helper = np.zeros_like(nn, dtype=np.float32)
    mask = np.abs(nn[:, 2]) < 0.999
    helper[mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    helper[~mask] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tangent = np.cross(helper, nn)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    bitangent = np.cross(nn, tangent)
    bitangent /= np.linalg.norm(bitangent, axis=1, keepdims=True) + 1e-8
    return tangent.astype(np.float32), bitangent.astype(np.float32)


def _closest_point_and_face(mesh: "trimesh.Trimesh", xyz: np.ndarray, *, chunk: int = 200000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return closest points, distances, and face indices (chunked)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    cps = np.zeros_like(xyz)
    dist = np.zeros((xyz.shape[0],), dtype=np.float32)
    face = np.zeros((xyz.shape[0],), dtype=np.int64)
    for i in range(0, xyz.shape[0], chunk):
        sl = slice(i, min(i + chunk, xyz.shape[0]))
        cp, d, f = trimesh.proximity.closest_point(mesh, xyz[sl])
        cps[sl] = cp.astype(np.float32)
        dist[sl] = d.astype(np.float32)
        face[sl] = f.astype(np.int64)
    return cps, dist, face


def _normal_variation_curv(xyz: np.ndarray, normals: np.ndarray, *, k: int = 20) -> np.ndarray:
    """Curvature proxy via local normal variation.

    Returns a scalar per point in [0, 2] roughly.
    0 => locally planar (normals aligned)
    larger => normals vary
    """
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(xyz)
        kk = min(k + 1, xyz.shape[0])
        _, nn = tree.query(xyz, k=kk)
        nn = nn[:, 1:]
        n0 = normals
        # compute mean 1 - |dot|
        dots = np.abs((n0[:, None, :] * normals[nn]).sum(axis=-1))  # [N,k]
        curv = 1.0 - dots.mean(axis=1)
        return curv.reshape(-1, 1).astype(np.float32)
    except Exception:
        # fallback: no curvature
        return np.zeros((xyz.shape[0], 1), dtype=np.float32)


def _make_partial_noisy_pc(
    surf_xyz: np.ndarray,
    *,
    n_pc: int,
    rng: np.random.RandomState,
    view_crop: float,
    noise_std: float,
    dropout: float,
) -> np.ndarray:
    """Simple scan-like degradation: front-side crop + gaussian noise + dropout."""
    xyz = np.asarray(surf_xyz, dtype=np.float32)

    # crop by random view direction (keep points with dot(p, v) >= thresh)
    v = _unit_vectors(rng, 1)[0]
    proj = (xyz * v[None, :]).sum(axis=1)
    thresh = np.quantile(proj, 1.0 - float(view_crop))  # view_crop=0.5 -> median (half)
    keep = proj >= thresh
    cropped = xyz[keep]
    if cropped.shape[0] < max(64, int(0.2 * n_pc)):
        cropped = xyz

    # dropout
    if dropout > 0.0 and cropped.shape[0] > 0:
        m = rng.rand(cropped.shape[0]) >= float(dropout)
        cropped = cropped[m]
        if cropped.shape[0] == 0:
            cropped = xyz

    # sample to fixed size
    if cropped.shape[0] >= n_pc:
        idx = _choice_np(cropped.shape[0], n_pc, rng)
        pc = cropped[idx]
    else:
        # pad with replacement
        idx = rng.choice(cropped.shape[0], size=n_pc, replace=True)
        pc = cropped[idx]

    # noise
    if noise_std > 0.0:
        pc = pc + rng.normal(scale=float(noise_std), size=pc.shape).astype(np.float32)

    # clamp to bounds
    pc = np.clip(pc, -1.0, 1.0)
    return pc.astype(np.float32)


def _choice_np(n: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    return rng.choice(n, size=k, replace=False).astype(np.int64)


def _sample_udf_queries(
    surf_xyz: np.ndarray,
    *,
    n_qry: int,
    rng: np.random.RandomState,
    near_ratio: float,
    near_std: float,
) -> np.ndarray:
    n_near = int(round(float(n_qry) * float(near_ratio)))
    n_uni = int(n_qry) - n_near

    uni = rng.uniform(-1.0, 1.0, size=(n_uni, 3)).astype(np.float32)

    if n_near > 0:
        idx = rng.choice(surf_xyz.shape[0], size=n_near, replace=True)
        near = surf_xyz[idx] + rng.normal(scale=float(near_std), size=(n_near, 3)).astype(np.float32)
        near = np.clip(near, -1.0, 1.0)
        xyz = np.concatenate([uni, near], axis=0)
    else:
        xyz = uni

    # shuffle
    perm = rng.permutation(xyz.shape[0])
    return xyz[perm].astype(np.float32)


def _compute_udf_dist_grad(mesh: "trimesh.Trimesh", qry_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cp, dist, face = _closest_point_and_face(mesh, qry_xyz)

    # grad: direction away from closest surface point
    v = qry_xyz - cp
    denom = dist.reshape(-1, 1) + 1e-8
    grad = v / denom

    # handle (near-)zero distance by using face normal
    zmask = dist < 1e-6
    if np.any(zmask):
        fn = mesh.face_normals[face[zmask]].astype(np.float32)
        fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
        grad[zmask] = fn

    return dist.reshape(-1, 1).astype(np.float32), grad.astype(np.float32)


def _compute_mesh_surface_features(
    mesh: "trimesh.Trimesh",
    surf_xyz: np.ndarray,
    *,
    curvature_knn: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mesh-native surface features aligned to surf_xyz."""
    _, _, surf_face = _closest_point_and_face(mesh, surf_xyz)
    surf_n = mesh.face_normals[surf_face].astype(np.float32)
    surf_n /= np.linalg.norm(surf_n, axis=1, keepdims=True) + 1e-8
    surf_curv = _normal_variation_curv(surf_xyz, surf_n, k=int(curvature_knn))
    return surf_n.astype(np.float32), surf_curv.astype(np.float32)


def _build_udf_grid_from_surface(
    surf_xyz: np.ndarray,
    *,
    grid: int,
    dilate: int,
) -> np.ndarray:
    if distance_transform_edt is None:
        raise RuntimeError("strict_udf_surface requires scipy.ndimage.distance_transform_edt")
    g = int(grid)
    occ = build_occ_grid_from_points(
        np.asarray(surf_xyz, dtype=np.float32),
        grid=g,
        dilate=int(dilate),
        bmin=-1.0,
        bmax=1.0,
    )
    voxel = 2.0 / float(g)
    udf = distance_transform_edt(~occ, sampling=(voxel, voxel, voxel)).astype(np.float32)
    return udf


def _sphere_trace_udf_grid_batch(
    udf_grid: np.ndarray,
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    *,
    max_t: float,
    n_steps: int,
    tol: float,
    min_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sphere tracing on an unsigned distance grid."""
    o = np.asarray(ray_o, dtype=np.float32)
    d = np.asarray(ray_d, dtype=np.float32)
    n = int(o.shape[0])
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    t = np.zeros((n,), dtype=np.float32)
    hit = np.zeros((n,), dtype=np.bool_)
    alive = np.ones((n,), dtype=np.bool_)
    max_t_f = float(max_t)
    tol_f = float(tol)
    min_step_f = float(max(min_step, 1e-6))

    for _ in range(int(n_steps)):
        if not bool(alive.any()):
            break
        ia = np.flatnonzero(alive)
        p = o[ia] + d[ia] * t[ia, None]
        inside = (
            (p[:, 0] >= -1.0)
            & (p[:, 0] <= 1.0)
            & (p[:, 1] >= -1.0)
            & (p[:, 1] <= 1.0)
            & (p[:, 2] >= -1.0)
            & (p[:, 2] <= 1.0)
        )
        if not bool(inside.all()):
            iout = ia[~inside]
            t[iout] = max_t_f
            alive[iout] = False
            ia = ia[inside]
            p = p[inside]
            if ia.size == 0:
                continue

        dist = trilinear_sample_grid(udf_grid, p, bmin=-1.0, bmax=1.0).astype(np.float32)
        dist = np.nan_to_num(dist, nan=max_t_f, posinf=max_t_f, neginf=0.0)
        h = dist <= tol_f
        if bool(h.any()):
            ih = ia[h]
            hit[ih] = True
            alive[ih] = False
        im = ia[~h]
        if im.size > 0:
            step = np.maximum(dist[~h], min_step_f)
            t[im] += step
            done = t[im] >= max_t_f
            if bool(done.any()):
                idn = im[done]
                t[idn] = max_t_f
                alive[idn] = False

    t = np.minimum(t, max_t_f).reshape(-1, 1).astype(np.float32)
    hit_f = hit.astype(np.float32).reshape(-1, 1)
    return t, hit_f


def _compute_surface_udf_clearance(
    surf_xyz: np.ndarray,
    surf_n: np.ndarray,
    *,
    udf_grid: np.ndarray,
    max_t: float,
    eps: float,
    n_steps: int,
    tol: float,
    min_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict UDF-like clearance along +/- surface normal via sphere tracing."""
    sxyz = np.asarray(surf_xyz, dtype=np.float32)
    sn = np.asarray(surf_n, dtype=np.float32)
    sn = sn / (np.linalg.norm(sn, axis=1, keepdims=True) + 1e-8)
    epsv = float(max(eps, 1e-6))

    o_in = sxyz - epsv * sn
    d_in = -sn
    o_out = sxyz + epsv * sn
    d_out = sn

    t_in, _hit_in = _sphere_trace_udf_grid_batch(
        udf_grid,
        o_in,
        d_in,
        max_t=float(max_t),
        n_steps=int(n_steps),
        tol=float(tol),
        min_step=float(min_step),
    )
    t_out, hit_out = _sphere_trace_udf_grid_batch(
        udf_grid,
        o_out,
        d_out,
        max_t=float(max_t),
        n_steps=int(n_steps),
        tol=float(tol),
        min_step=float(min_step),
    )
    thickness = (t_in + t_out).astype(np.float32)
    return t_in.astype(np.float32), t_out.astype(np.float32), hit_out.astype(np.float32), thickness


def _sample_rays(
    *,
    n_rays: int,
    rng: np.random.RandomState,
    radius: float = 2.5,
    jitter_std: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    if int(n_rays) <= 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    o = _unit_vectors(rng, n_rays) * float(radius)
    jitter = rng.normal(scale=float(jitter_std), size=o.shape).astype(np.float32)
    d = -o + jitter
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-8
    return o.astype(np.float32), d.astype(np.float32)


def _ray_intersect(
    mesh: "trimesh.Trimesh",
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    *,
    min_t: float = 0.0,
    max_t: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t, hit, normal) per ray."""
    if int(ray_o.shape[0]) == 0:
        return (
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )
    # Use trimesh's pure triangle intersector for portability.
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    loc, idx_ray, idx_tri = intersector.intersects_location(ray_o, ray_d, multiple_hits=False)

    n = ray_o.shape[0]
    hit = np.zeros((n, 1), dtype=np.float32)
    t = np.zeros((n, 1), dtype=np.float32)
    nrm = np.zeros((n, 3), dtype=np.float32)

    if loc.shape[0] > 0:
        dist = np.linalg.norm(loc - ray_o[idx_ray], axis=1).astype(np.float32)
        keep = dist >= float(max(min_t, 0.0))
        if max_t is not None:
            keep &= dist <= float(max_t)
        if bool(keep.any()):
            idx_ray_keep = idx_ray[keep]
            idx_tri_keep = idx_tri[keep]
            dist_keep = dist[keep]
            hit[idx_ray_keep, 0] = 1.0
            t[idx_ray_keep, 0] = dist_keep
            fn = mesh.face_normals[idx_tri_keep].astype(np.float32)
            fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8
            nrm[idx_ray_keep] = fn

    return t, hit, nrm


def _compute_mesh_query_visibility(
    mesh: "trimesh.Trimesh",
    mesh_qry_xyz: np.ndarray,
    mesh_qry_n: np.ndarray,
    *,
    n_dirs: int,
    max_t: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute local hemisphere visibility answers for mesh queries."""
    qxyz = np.asarray(mesh_qry_xyz, dtype=np.float32)
    qn = np.asarray(mesh_qry_n, dtype=np.float32)
    n_q = int(qxyz.shape[0])
    n_dirs_i = int(n_dirs)
    if n_q == 0 or n_dirs_i <= 0:
        return (
            np.zeros((n_q, max(n_dirs_i, 0)), dtype=np.float32),
            np.zeros((n_q, max(n_dirs_i, 0)), dtype=np.float32),
        )

    qn = qn / (np.linalg.norm(qn, axis=1, keepdims=True) + 1e-8)
    hemi = _fibonacci_hemisphere_dirs(n_dirs_i)  # [D,3], around +Z
    tangent, bitangent = _orthonormal_basis_from_normals(qn)

    dirs_world = (
        tangent[:, None, :] * hemi[None, :, 0:1]
        + bitangent[:, None, :] * hemi[None, :, 1:2]
        + qn[:, None, :] * hemi[None, :, 2:3]
    )
    dirs_world /= np.linalg.norm(dirs_world, axis=2, keepdims=True) + 1e-8

    ray_o = np.repeat(
        qxyz[:, None, :] + float(eps) * qn[:, None, :],
        n_dirs_i,
        axis=1,
    ).reshape(-1, 3)
    ray_d = dirs_world.reshape(-1, 3)
    t, hit, _nrm = _ray_intersect(
        mesh,
        ray_o,
        ray_d,
        min_t=max(float(eps) * 2.0, 1e-5),
        max_t=float(max_t),
    )
    hit = hit.reshape(n_q, n_dirs_i).astype(np.float32)
    t = t.reshape(n_q, n_dirs_i).astype(np.float32)
    t = np.where(hit > 0.5, t, float(max_t)).astype(np.float32)
    return hit, t


@dataclass
class V2GenConfig:
    n_surf: int = 8192
    n_mesh_qry: int = 2048
    n_udf_qry: int = 8192
    n_pc: int = 2048
    n_pc_qry: int = 1024
    n_rays: int = 4096

    pc_view_crop: float = 0.5
    pc_noise_std: float = 0.005
    pc_dropout: float = 0.1

    udf_near_ratio: float = 0.5
    udf_near_std: float = 0.05

    curvature_knn: int = 20
    pca_knn: int = 20

    ray_radius: float = 2.5
    ray_jitter_std: float = 0.05
    mesh_vis_enable: bool = True
    mesh_vis_dirs: int = 16
    mesh_vis_max_t: float = 0.25
    mesh_vis_eps: float = 1e-3
    strict_udf_surface: bool = True
    surf_udf_grid: int = 128
    surf_udf_dilate: int = 1
    surf_udf_max_t: float = 2.0
    surf_udf_eps: float = 1e-4
    surf_udf_steps: int = 64
    surf_udf_tol: float = 1e-4
    surf_udf_min_step: float = 1e-4
    augment_existing: bool = False
    skip_existing: bool = False


_REQUIRED_SURF_KEYS = (
    "mesh_surf_n",
    "mesh_surf_curv",
    "udf_surf_t_in",
    "udf_surf_t_out",
    "udf_surf_hit_out",
    "udf_surf_thickness",
    "pc_n",
    "pc_density",
)

_REQUIRED_VIS_KEYS = (
    "mesh_qry_vis_hit",
    "mesh_qry_vis_t",
)


def _has_required_keys(npz: "np.lib.npyio.NpzFile", *, cfg: V2GenConfig) -> bool:
    if not all(k in npz for k in _REQUIRED_SURF_KEYS):
        return False
    if bool(cfg.mesh_vis_enable) and not all(k in npz for k in _REQUIRED_VIS_KEYS):
        return False
    return True


def preprocess_one(mesh_path: str, out_path: str, *, cfg: V2GenConfig, seed: int) -> Optional[str]:
    synset, model_id = _infer_synset_model(mesh_path)
    rng = np.random.RandomState(seed)
    if os.path.isfile(out_path) and bool(cfg.skip_existing):
        if not bool(cfg.augment_existing):
            return None
        try:
            with np.load(out_path, allow_pickle=False) as npz_old:
                if _has_required_keys(npz_old, cfg=cfg):
                    return None
        except Exception:
            pass

    try:
        existing: Dict[str, np.ndarray] = {}
        if bool(cfg.augment_existing) and os.path.isfile(out_path):
            with np.load(out_path, allow_pickle=False) as npz_old:
                existing = {k: np.asarray(npz_old[k]) for k in npz_old.files}

        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            # some files load as Scene
            mesh = mesh.dump().sum()
        mesh = normalize_mesh(mesh)
        if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
            return f"empty mesh: {mesh_path}"

        # --- surface context ---
        if "surf_xyz" in existing:
            surf_xyz = np.asarray(existing["surf_xyz"], dtype=np.float32)
        else:
            surf_xyz, _ = _sample_surface(mesh, int(cfg.n_surf), rng)
            surf_xyz = np.asarray(surf_xyz, dtype=np.float32)

        # --- mesh query: normals + curvature proxy ---
        if ("mesh_qry_xyz" in existing) and ("mesh_qry_n" in existing) and ("mesh_qry_curv" in existing):
            mesh_qry_xyz = np.asarray(existing["mesh_qry_xyz"], dtype=np.float32)
            mesh_qry_n = np.asarray(existing["mesh_qry_n"], dtype=np.float32)
            mesh_qry_curv = np.asarray(existing["mesh_qry_curv"], dtype=np.float32)
        else:
            mesh_qry_xyz, mesh_qry_f = _sample_surface(mesh, int(cfg.n_mesh_qry), rng)
            mesh_qry_n = mesh.face_normals[mesh_qry_f].astype(np.float32)
            mesh_qry_n /= np.linalg.norm(mesh_qry_n, axis=1, keepdims=True) + 1e-8
            mesh_qry_curv = _normal_variation_curv(mesh_qry_xyz, mesh_qry_n, k=int(cfg.curvature_knn))
        if bool(cfg.mesh_vis_enable) and ("mesh_qry_vis_hit" in existing) and ("mesh_qry_vis_t" in existing):
            mesh_qry_vis_hit = np.asarray(existing["mesh_qry_vis_hit"], dtype=np.float32)
            mesh_qry_vis_t = np.asarray(existing["mesh_qry_vis_t"], dtype=np.float32)
        elif bool(cfg.mesh_vis_enable):
            mesh_qry_vis_hit, mesh_qry_vis_t = _compute_mesh_query_visibility(
                mesh,
                mesh_qry_xyz,
                mesh_qry_n,
                n_dirs=int(cfg.mesh_vis_dirs),
                max_t=float(cfg.mesh_vis_max_t),
                eps=float(cfg.mesh_vis_eps),
            )

        # --- UDF queries: distance + gradient proxy ---
        if ("udf_qry_xyz" in existing) and ("udf_qry_dist" in existing) and ("udf_qry_grad" in existing):
            udf_qry_xyz = np.asarray(existing["udf_qry_xyz"], dtype=np.float32)
            udf_qry_dist = np.asarray(existing["udf_qry_dist"], dtype=np.float32)
            udf_qry_grad = np.asarray(existing["udf_qry_grad"], dtype=np.float32)
        else:
            udf_qry_xyz = _sample_udf_queries(
                surf_xyz,
                n_qry=int(cfg.n_udf_qry),
                rng=rng,
                near_ratio=float(cfg.udf_near_ratio),
                near_std=float(cfg.udf_near_std),
            )
            udf_qry_dist, udf_qry_grad = _compute_udf_dist_grad(mesh, udf_qry_xyz)

        # --- point cloud context/query: PCA normal + density ---
        if "pc_xyz" in existing:
            pc_xyz = np.asarray(existing["pc_xyz"], dtype=np.float32)
        else:
            pc_xyz = _make_partial_noisy_pc(
                surf_xyz,
                n_pc=int(cfg.n_pc),
                rng=rng,
                view_crop=float(cfg.pc_view_crop),
                noise_std=float(cfg.pc_noise_std),
                dropout=float(cfg.pc_dropout),
            )
        pc_n_all = estimate_normals_pca(pc_xyz, k=int(cfg.pca_knn))
        pc_d_all = estimate_density_knn(pc_xyz, k=int(cfg.pca_knn))
        if pc_n_all is None:
            pc_n_all = np.zeros_like(pc_xyz, dtype=np.float32)
        if pc_d_all is None:
            pc_d_all = np.zeros((pc_xyz.shape[0], 1), dtype=np.float32)

        if ("pc_qry_xyz" in existing) and ("pc_qry_n" in existing) and ("pc_qry_density" in existing):
            pc_qry_xyz = np.asarray(existing["pc_qry_xyz"], dtype=np.float32)
            pc_qry_n = np.asarray(existing["pc_qry_n"], dtype=np.float32)
            pc_qry_density = np.asarray(existing["pc_qry_density"], dtype=np.float32)
        else:
            pc_qry_idx = _choice_np(pc_xyz.shape[0], int(cfg.n_pc_qry), rng)
            pc_qry_xyz = pc_xyz[pc_qry_idx]
            pc_qry_n = pc_n_all[pc_qry_idx].astype(np.float32)
            pc_qry_density = pc_d_all[pc_qry_idx].astype(np.float32)

        # --- surf-aligned strict mesh/UDF/PC features ---
        mesh_surf_n, mesh_surf_curv = _compute_mesh_surface_features(
            mesh, surf_xyz, curvature_knn=int(cfg.curvature_knn)
        )
        if bool(cfg.strict_udf_surface):
            udf_grid_local = _build_udf_grid_from_surface(
                surf_xyz,
                grid=int(cfg.surf_udf_grid),
                dilate=int(cfg.surf_udf_dilate),
            )
            udf_surf_t_in, udf_surf_t_out, udf_surf_hit_out, udf_surf_thickness = _compute_surface_udf_clearance(
                surf_xyz,
                mesh_surf_n,
                udf_grid=udf_grid_local,
                max_t=float(cfg.surf_udf_max_t),
                eps=float(cfg.surf_udf_eps),
                n_steps=int(cfg.surf_udf_steps),
                tol=float(cfg.surf_udf_tol),
                min_step=float(cfg.surf_udf_min_step),
            )
        else:
            n_s = int(surf_xyz.shape[0])
            udf_surf_t_in = np.zeros((n_s, 1), dtype=np.float32)
            udf_surf_t_out = np.zeros((n_s, 1), dtype=np.float32)
            udf_surf_hit_out = np.zeros((n_s, 1), dtype=np.float32)
            udf_surf_thickness = np.zeros((n_s, 1), dtype=np.float32)

        # --- rays (optional) ---
        if ("ray_o" in existing) and ("ray_d" in existing) and ("ray_t" in existing) and ("ray_hit" in existing) and ("ray_n" in existing):
            ray_o = np.asarray(existing["ray_o"], dtype=np.float32)
            ray_d = np.asarray(existing["ray_d"], dtype=np.float32)
            ray_t = np.asarray(existing["ray_t"], dtype=np.float32)
            ray_hit = np.asarray(existing["ray_hit"], dtype=np.float32)
            ray_n = np.asarray(existing["ray_n"], dtype=np.float32)
        else:
            ray_o, ray_d = _sample_rays(
                n_rays=int(cfg.n_rays),
                rng=rng,
                radius=float(cfg.ray_radius),
                jitter_std=float(cfg.ray_jitter_std),
            )
            ray_t, ray_hit, ray_n = _ray_intersect(mesh, ray_o, ray_d)

        payload: Dict[str, np.ndarray] = dict(existing) if bool(existing) else {}
        payload.update(
            {
                "v2": np.int32(2),
                "synset": np.bytes_(synset),
                "model_id": np.bytes_(model_id),
                # contexts
                "surf_xyz": surf_xyz.astype(np.float32),
                "pc_xyz": pc_xyz.astype(np.float32),
                # surf-aligned primitive-native answer sources
                "mesh_surf_n": mesh_surf_n.astype(np.float32),
                "mesh_surf_curv": mesh_surf_curv.astype(np.float32),
                "udf_surf_t_in": udf_surf_t_in.astype(np.float32),
                "udf_surf_t_out": udf_surf_t_out.astype(np.float32),
                "udf_surf_hit_out": udf_surf_hit_out.astype(np.float32),
                "udf_surf_thickness": udf_surf_thickness.astype(np.float32),
                "pc_n": pc_n_all.astype(np.float32),
                "pc_density": pc_d_all.astype(np.float32),
                # legacy v2 query path (kept for token pretrain / CPAC)
                "pc_qry_xyz": pc_qry_xyz.astype(np.float32),
                "pc_qry_n": pc_qry_n.astype(np.float32),
                "pc_qry_density": pc_qry_density.astype(np.float32),
                "mesh_qry_xyz": mesh_qry_xyz.astype(np.float32),
                "mesh_qry_n": mesh_qry_n.astype(np.float32),
                "mesh_qry_curv": mesh_qry_curv.astype(np.float32),
                "udf_qry_xyz": udf_qry_xyz.astype(np.float32),
                "udf_qry_dist": udf_qry_dist.astype(np.float32),
                "udf_qry_grad": udf_qry_grad.astype(np.float32),
                # optional rays
                "ray_o": ray_o.astype(np.float32),
                "ray_d": ray_d.astype(np.float32),
                "ray_t": ray_t.astype(np.float32),
                "ray_hit": ray_hit.astype(np.float32),
                "ray_n": ray_n.astype(np.float32),
            }
        )
        if bool(cfg.mesh_vis_enable):
            payload["mesh_qry_vis_hit"] = mesh_qry_vis_hit.astype(np.float32)
            payload["mesh_qry_vis_t"] = mesh_qry_vis_t.astype(np.float32)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez(out_path, **payload)
        return None

    except Exception as e:
        return f"{mesh_path}: {type(e).__name__}: {e}"


def _glob_meshes(shapenet_root: str, synsets: Optional[List[str]] = None) -> List[str]:
    meshes: List[str] = []
    if synsets is None or len(synsets) == 0:
        pattern = os.path.join(shapenet_root, "*", "*", "models", "model_normalized.obj")
        meshes = sorted([p for p in glob_glob(pattern)])
    else:
        for syn in synsets:
            pattern = os.path.join(shapenet_root, syn, "*", "models", "model_normalized.obj")
            meshes.extend(glob_glob(pattern))
        meshes = sorted(meshes)
    return meshes


def glob_glob(pattern: str) -> List[str]:
    import glob

    return glob.glob(pattern)


def _filter_shard(items: List[Tuple[str, str, int]], num_shards: int, shard_id: int) -> List[Tuple[str, str, int]]:
    ns = int(num_shards)
    sid = int(shard_id)
    if ns <= 1:
        return list(items)
    if sid < 0 or sid >= ns:
        raise ValueError(f"invalid shard_id={sid} for num_shards={ns}")
    return [x for i, x in enumerate(items) if (i % ns) == sid]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenet_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--synsets", type=str, default="", help="comma-separated synset ids (empty=all)")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--num_shards", type=int, default=1, help="split target task list into N shards.")
    ap.add_argument("--shard_id", type=int, default=0, help="0-based shard id when num_shards>1.")

    # sizes
    ap.add_argument("--n_surf", type=int, default=8192)
    ap.add_argument("--n_mesh_qry", type=int, default=2048)
    ap.add_argument("--n_udf_qry", type=int, default=8192)
    ap.add_argument("--n_pc", type=int, default=2048)
    ap.add_argument("--n_pc_qry", type=int, default=1024)
    ap.add_argument("--n_rays", type=int, default=4096)

    # pc
    ap.add_argument("--pc_view_crop", type=float, default=0.5)
    ap.add_argument("--pc_noise_std", type=float, default=0.005)
    ap.add_argument("--pc_dropout", type=float, default=0.1)

    # udf
    ap.add_argument("--udf_near_ratio", type=float, default=0.5)
    ap.add_argument("--udf_near_std", type=float, default=0.05)

    # knn
    ap.add_argument("--curvature_knn", type=int, default=20)
    ap.add_argument("--pca_knn", type=int, default=20)

    # rays
    ap.add_argument("--ray_radius", type=float, default=2.5)
    ap.add_argument("--ray_jitter_std", type=float, default=0.05)
    ap.add_argument("--mesh_vis_enable", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mesh_vis_dirs", type=int, default=16)
    ap.add_argument("--mesh_vis_max_t", type=float, default=0.25)
    ap.add_argument("--mesh_vis_eps", type=float, default=1e-3)
    ap.add_argument("--strict_udf_surface", type=int, default=1, choices=[0, 1])
    ap.add_argument("--surf_udf_grid", type=int, default=128)
    ap.add_argument("--surf_udf_dilate", type=int, default=1)
    ap.add_argument("--surf_udf_max_t", type=float, default=2.0)
    ap.add_argument("--surf_udf_eps", type=float, default=1e-4)
    ap.add_argument("--surf_udf_steps", type=int, default=64)
    ap.add_argument("--surf_udf_tol", type=float, default=1e-4)
    ap.add_argument("--surf_udf_min_step", type=float, default=1e-4)
    ap.add_argument(
        "--augment_existing",
        action="store_true",
        help="When output NPZ exists, load and append/update keys instead of skipping/overwriting from scratch.",
    )
    ap.add_argument("--skip_existing", action="store_true", help="skip samples whose output NPZ already exists.")
    ap.add_argument(
        "--missing_only",
        action="store_true",
        help="process only tasks whose output NPZ is currently missing.",
    )

    args = ap.parse_args()

    synsets = [s.strip() for s in args.synsets.split(",") if s.strip()]
    meshes = _glob_meshes(args.shapenet_root, synsets)
    if len(meshes) == 0:
        raise SystemExit(f"No meshes found under {args.shapenet_root}")

    rng = random.Random(int(args.seed))
    rng.shuffle(meshes)
    n_train = int(round(len(meshes) * float(args.train_ratio)))
    train_meshes = meshes[:n_train]
    test_meshes = meshes[n_train:]

    cfg = V2GenConfig(
        n_surf=args.n_surf,
        n_mesh_qry=args.n_mesh_qry,
        n_udf_qry=args.n_udf_qry,
        n_pc=args.n_pc,
        n_pc_qry=args.n_pc_qry,
        n_rays=args.n_rays,
        pc_view_crop=args.pc_view_crop,
        pc_noise_std=args.pc_noise_std,
        pc_dropout=args.pc_dropout,
        udf_near_ratio=args.udf_near_ratio,
        udf_near_std=args.udf_near_std,
        curvature_knn=args.curvature_knn,
        pca_knn=args.pca_knn,
        ray_radius=args.ray_radius,
        ray_jitter_std=args.ray_jitter_std,
        mesh_vis_enable=bool(int(args.mesh_vis_enable)),
        mesh_vis_dirs=int(args.mesh_vis_dirs),
        mesh_vis_max_t=float(args.mesh_vis_max_t),
        mesh_vis_eps=float(args.mesh_vis_eps),
        strict_udf_surface=bool(int(args.strict_udf_surface)),
        surf_udf_grid=int(args.surf_udf_grid),
        surf_udf_dilate=int(args.surf_udf_dilate),
        surf_udf_max_t=float(args.surf_udf_max_t),
        surf_udf_eps=float(args.surf_udf_eps),
        surf_udf_steps=int(args.surf_udf_steps),
        surf_udf_tol=float(args.surf_udf_tol),
        surf_udf_min_step=float(args.surf_udf_min_step),
        augment_existing=bool(args.augment_existing),
        skip_existing=bool(args.skip_existing),
    )

    def _out_path(mesh_path: str, split: str) -> str:
        syn, mid = _infer_synset_model(mesh_path)
        return os.path.join(args.out_root, split, syn, f"{mid}.npz")

    tasks_all: List[Tuple[str, str, int]] = []
    for m in train_meshes:
        tasks_all.append((m, _out_path(m, "train"), int(args.seed)))
    for m in test_meshes:
        tasks_all.append((m, _out_path(m, "test"), int(args.seed) + 12345))

    num_tasks_all = len(tasks_all)
    num_missing_tasks_total = 0
    if bool(args.missing_only):
        filtered: List[Tuple[str, str, int]] = []
        for mesh_path, out_path, base_seed in tasks_all:
            if not os.path.isfile(out_path):
                filtered.append((mesh_path, out_path, base_seed))
        tasks_all = filtered
        num_missing_tasks_total = len(tasks_all)

    tasks = _filter_shard(tasks_all, int(args.num_shards), int(args.shard_id))
    print(
        f"[shard] num_shards={int(args.num_shards)} shard_id={int(args.shard_id)} "
        f"tasks={len(tasks)}"
    )
    if bool(args.missing_only):
        print(f"[missing_only] total_missing_now={num_missing_tasks_total} total_all={num_tasks_all}")

    os.makedirs(args.out_root, exist_ok=True)

    # multiprocessing
    if int(args.num_workers) <= 1:
        errors = []
        for mesh_path, out_path, base_seed in tqdm(tasks):
            syn, mid = _infer_synset_model(mesh_path)
            import zlib
            seed = (zlib.crc32(mesh_path.encode("utf-8")) ^ base_seed) & 0x7FFFFFFF
            err = preprocess_one(mesh_path, out_path, cfg=cfg, seed=seed)
            if err is not None:
                errors.append(err)
    else:
        import multiprocessing as mp

        with mp.Pool(processes=int(args.num_workers)) as pool:
            fn = partial(_worker, cfg=cfg)
            errors = list(tqdm(pool.imap(fn, tasks), total=len(tasks)))
        errors = [e for e in errors if e is not None]

    manifest = {
        "shapenet_root": args.shapenet_root,
        "out_root": args.out_root,
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "num_meshes": len(meshes),
        "num_train": len(train_meshes),
        "num_test": len(test_meshes),
        "missing_only": bool(args.missing_only),
        "num_tasks_all": num_tasks_all,
        "num_missing_tasks_total": num_missing_tasks_total,
        "num_tasks_in_shard": len(tasks),
        "config": asdict(cfg),
        "errors": errors[:100],
    }
    if int(args.num_shards) > 1:
        manifest_name = f"v2_manifest.shard{int(args.shard_id):03d}of{int(args.num_shards):03d}.json"
    else:
        manifest_name = "v2_manifest.json"
    with open(os.path.join(args.out_root, manifest_name), "w") as f:
        json.dump(manifest, f, indent=2)

    if len(errors) > 0:
        print(f"[WARN] {len(errors)} meshes failed. See v2_manifest.json for examples.")


def _worker(task: Tuple[str, str, int], *, cfg: V2GenConfig) -> Optional[str]:
    mesh_path, out_path, base_seed = task
    import zlib
    seed = (zlib.crc32(mesh_path.encode("utf-8")) ^ base_seed) & 0x7FFFFFFF
    return preprocess_one(mesh_path, out_path, cfg=cfg, seed=seed)


if __name__ == "__main__":
    main()
