"""Convert legacy (QueryNEPA-style) cached `.npz` into the v2 key layout.

Why this exists
--------------
The original preprocessing in this repo writes QueryNEPA-oriented fields such as:
  - pc_xyz, pc_n, pt_xyz_pool, pt_dist_pool, pt_dist_udf_pool, ray_* ...

For PatchNEPA v2 we want an explicit separation:
  - surf_xyz: surface-distributed points used as PatchNEPA input
  - qry_xyz : spatial query points used for CPAC-style evaluation

This script is a *lightweight bridge* that lets you reuse existing caches while
transitioning to v2.

It also optionally computes a few additional fields useful for per-primitive
Answer packs (e.g., PCA normals / local density for point clouds).

NOTE
----
This is intentionally conservative: it does not attempt to infer "the best"
Answer features for your paper; it just makes it easy to store them and choose
later via `answer_packs` config.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _maybe_import_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return cKDTree
    except Exception:
        return None


def estimate_density_knn(xyz: np.ndarray, *, k: int = 20) -> Optional[np.ndarray]:
    """Return per-point density proxy: mean kNN distance (smaller => denser)."""
    cKDTree = _maybe_import_ckdtree()
    if cKDTree is None:
        return None
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=min(k + 1, xyz.shape[0]))  # includes self (0)
    if d.ndim == 1:
        # degenerate
        return None
    d = d[:, 1:]  # drop self
    return d.mean(axis=1, keepdims=True).astype(np.float32)


def estimate_normals_pca(xyz: np.ndarray, *, k: int = 20) -> Optional[np.ndarray]:
    """Estimate normals via PCA on kNN neighborhoods."""
    cKDTree = _maybe_import_ckdtree()
    if cKDTree is None:
        return None
    n = xyz.shape[0]
    kk = min(k + 1, n)
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz, k=kk)
    nn = nn[:, 1:]  # drop self

    normals = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pts = xyz[nn[i]]
        mu = pts.mean(axis=0, keepdims=True)
        cov = (pts - mu).T @ (pts - mu) / max(1, pts.shape[0])
        # smallest eigenvector
        w, v = np.linalg.eigh(cov)
        nrm = v[:, 0]
        # normalize
        nrm = nrm / (np.linalg.norm(nrm) + 1e-8)
        normals[i] = nrm.astype(np.float32)
    return normals


def trilinear_sample_grid(
    grid: np.ndarray,
    xyz: np.ndarray,
    *,
    bmin: float = -1.0,
    bmax: float = 1.0,
) -> np.ndarray:
    """Trilinear sample a (D,D,D) grid at xyz in [bmin,bmax]."""
    D = int(grid.shape[0])
    # normalize to [0, D-1]
    p = (xyz - bmin) / (bmax - bmin) * (D - 1)
    p = np.clip(p, 0.0, float(D - 1))
    i0 = np.floor(p).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, D - 1)
    t = (p - i0.astype(np.float32)).astype(np.float32)

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = i1[:, 0], i1[:, 1], i1[:, 2]
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    c = c0 * (1 - tz) + c1 * tz
    return c.astype(np.float32)


def convert_one(
    in_path: str,
    out_path: str,
    *,
    prim_name: str,
    compute_pc_pca: bool,
    compute_pc_density: bool,
    udf_clearance_delta: Optional[float],
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with np.load(in_path, allow_pickle=False) as npz:
        out: Dict[str, np.ndarray] = {}

        # --- primitive metadata ---
        out["prim_name"] = np.asarray(str(prim_name))

        # --- surface stream ---
        if "pc_xyz" in npz:
            out["surf_xyz"] = np.asarray(npz["pc_xyz"]).astype(np.float32)
        elif "pt_xyz_pool" in npz:
            # fallback
            out["surf_xyz"] = np.asarray(npz["pt_xyz_pool"]).astype(np.float32)
        else:
            raise KeyError(f"missing pc_xyz/pt_xyz_pool in {in_path}")

        if "pc_n" in npz:
            # legacy pc_n is actually mesh face normal for pc_xyz
            out["surf_n_mesh"] = np.asarray(npz["pc_n"]).astype(np.float32)

        if "pc_fps_order" in npz:
            out["surf_fps_order"] = np.asarray(npz["pc_fps_order"]).astype(np.int64)

        # Optional: point-cloud derived features
        xyz = out["surf_xyz"]
        if compute_pc_pca:
            nrm = estimate_normals_pca(xyz)
            if nrm is not None:
                out["pc_n_pca"] = nrm.astype(np.float32)
        if compute_pc_density:
            dens = estimate_density_knn(xyz)
            if dens is not None:
                out["pc_density"] = dens.astype(np.float32)

        # --- query stream ---
        if "pt_xyz_pool" in npz:
            out["qry_xyz"] = np.asarray(npz["pt_xyz_pool"]).astype(np.float32)
        if "pt_dist_pool" in npz:
            # distance to mesh (legacy)
            out["qry_dist_mesh"] = np.asarray(npz["pt_dist_pool"]).astype(np.float32)
        if "pt_dist_udf_pool" in npz:
            out["qry_dist_udf"] = np.asarray(npz["pt_dist_udf_pool"]).astype(np.float32)

        # --- UDF grid (optional) ---
        if "udf_grid" in npz:
            out["udf_grid"] = np.asarray(npz["udf_grid"]).astype(np.float32)
        if "occ_grid" in npz:
            out["occ_grid"] = np.asarray(npz["occ_grid"]).astype(np.uint8)

        # UDF clearance at an offset from the surface along the mesh normal
        if udf_clearance_delta is not None and ("udf_grid" in out) and ("surf_n_mesh" in out):
            delta = float(udf_clearance_delta)
            p_off = out["surf_xyz"] - delta * out["surf_n_mesh"]
            out["surf_udf_clearance"] = trilinear_sample_grid(out["udf_grid"], p_off)[:, None]

        # --- ray pools (if present) ---
        for k in ("ray_o", "ray_d", "ray_hit", "ray_t", "ray_n"):
            if k in npz:
                out[k] = np.asarray(npz[k])

        np.savez_compressed(out_path, **out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="input npz path or a glob")
    ap.add_argument("--out_root", type=str, required=True, help="output root directory")
    ap.add_argument("--prim_name", type=str, default="unknown", help="mesh/udf/pc")
    ap.add_argument("--compute_pc_pca", action="store_true")
    ap.add_argument("--compute_pc_density", action="store_true")
    ap.add_argument(
        "--udf_clearance_delta",
        type=float,
        default=0.05,
        help="if udf_grid exists, sample udf at surf_xyz - delta*surf_n_mesh and store surf_udf_clearance",
    )
    args = ap.parse_args()

    in_paths = sorted(glob.glob(args.in_path))
    if not in_paths:
        raise FileNotFoundError(f"no input matched: {args.in_path}")

    for p in in_paths:
        rel = os.path.basename(p)
        out_path = os.path.join(args.out_root, rel)
        convert_one(
            p,
            out_path,
            prim_name=str(args.prim_name),
            compute_pc_pca=bool(args.compute_pc_pca),
            compute_pc_density=bool(args.compute_pc_density),
            udf_clearance_delta=float(args.udf_clearance_delta) if args.udf_clearance_delta is not None else None,
        )


if __name__ == "__main__":
    main()
