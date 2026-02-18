from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def _as_float32_xyz(points: NDArray[np.floating]) -> NDArray[np.float32]:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3); got {pts.shape}")
    return pts.astype(np.float32, copy=False)


def farthest_from_centroid_start(points: NDArray[np.floating]) -> int:
    """Deterministic FPS start index: farthest point from the centroid."""
    pts = _as_float32_xyz(points)
    c = pts.mean(axis=0, keepdims=True)
    d2 = ((pts - c) ** 2).sum(axis=1)
    return int(d2.argmax())


def fps_order(
    points: NDArray[np.floating],
    k: int,
    *,
    start_idx: Optional[int] = None,
) -> NDArray[np.int32]:
    """Compute farthest-point sampling order (length=min(k,N))."""
    pts = _as_float32_xyz(points)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    kk = int(min(max(k, 0), n))
    if kk == 0:
        return np.zeros((0,), dtype=np.int32)

    if start_idx is None:
        start_idx = farthest_from_centroid_start(pts)
    start_idx = int(start_idx)
    if not (0 <= start_idx < n):
        raise ValueError(f"start_idx out of range: {start_idx} for N={n}")

    out = np.empty((kk,), dtype=np.int32)
    out[0] = start_idx

    sel = pts[start_idx]
    best_d2 = ((pts - sel) ** 2).sum(axis=1)

    for i in range(1, kk):
        far = int(best_d2.argmax())
        out[i] = far
        sel = pts[far]
        d2 = ((pts - sel) ** 2).sum(axis=1)
        best_d2 = np.minimum(best_d2, d2)

    return out


def rfps_order(
    points: NDArray[np.floating],
    k: int,
    *,
    m: int,
    rng: np.random.Generator,
    start_idx: Optional[int] = None,
) -> NDArray[np.int32]:
    """Random-FPS fallback: random candidates then FPS inside candidates."""
    pts = _as_float32_xyz(points)
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    mm = int(min(max(m, 1), n))
    cand = rng.choice(n, size=mm, replace=False)
    cand_pts = pts[cand]
    local_start = farthest_from_centroid_start(cand_pts) if start_idx is None else int(start_idx)
    local = fps_order(cand_pts, k, start_idx=local_start)
    return cand[local].astype(np.int32, copy=False)
