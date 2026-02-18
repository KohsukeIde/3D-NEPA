"""Mesh evaluation metrics (Chamfer distance, F-score).

All meshes are assumed to be in the same coordinate system.

Implementation notes:
- We compute metrics on sampled surface point clouds.
- Chamfer distance is symmetric: mean_nn(P->Q) + mean_nn(Q->P).
- F-score uses precision/recall at threshold tau.

Dependencies:
- trimesh
- scipy (cKDTree)
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def _require(pkg: str):
    raise ImportError(
        f"Missing optional dependency '{pkg}'. Install it to use mesh metrics."
    )


def sample_mesh_points(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Uniformly sample points on a triangle mesh surface."""
    try:
        import trimesh
    except Exception as e:
        _require("trimesh")
        raise e

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rng = np.random.RandomState(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=rng)
    return pts.astype(np.float32)


def chamfer_distance(
    p: np.ndarray,
    q: np.ndarray,
    squared: bool = True,
) -> float:
    """Symmetric Chamfer distance between two point sets."""
    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        _require("scipy")
        raise e

    tree_q = cKDTree(q)
    d_p, _ = tree_q.query(p, k=1)

    tree_p = cKDTree(p)
    d_q, _ = tree_p.query(q, k=1)

    if squared:
        return float((d_p ** 2).mean() + (d_q ** 2).mean())
    return float(d_p.mean() + d_q.mean())


def fscore_at_tau(p: np.ndarray, q: np.ndarray, tau: float) -> float:
    """F-score between point sets at distance threshold tau."""
    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        _require("scipy")
        raise e

    tree_q = cKDTree(q)
    d_p, _ = tree_q.query(p, k=1)
    prec = float((d_p < tau).mean())

    tree_p = cKDTree(p)
    d_q, _ = tree_p.query(q, k=1)
    rec = float((d_q < tau).mean())

    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def mesh_metrics(
    pred_v: np.ndarray,
    pred_f: np.ndarray,
    gt_v: np.ndarray,
    gt_f: np.ndarray,
    n_samples: int = 20000,
    taus: Iterable[float] = (0.005, 0.01, 0.02),
    seed: int = 0,
) -> Dict[str, float]:
    """Compute mesh-to-mesh metrics via surface sampling."""
    p = sample_mesh_points(pred_v, pred_f, n_samples, seed=seed)
    q = sample_mesh_points(gt_v, gt_f, n_samples, seed=seed + 1)

    out: Dict[str, float] = {}
    out["chamfer_l2"] = chamfer_distance(p, q, squared=True)
    out["chamfer_l1"] = chamfer_distance(p, q, squared=False)
    for tau in taus:
        out[f"fscore@{tau}"] = fscore_at_tau(p, q, tau)
    return out
