"""Token ordering utilities.

This module contains deterministic ordering functions for point and ray samples.

Motivation:
- Point clouds have no canonical order; we use Morton / FPS / random as proxies.
- Rays can be ordered by direction (theta/phi) or by per-view raster (u,v).
- For mixed point+ray sequences, a common "anchor" in 3D (x = o + t d) enables
  spatially consistent ordering across modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def morton3d(xyz: NDArray[np.floating]) -> NDArray[np.uint64]:
    """Compute a Morton (Z-order) code for each 3D point.

    Assumes points are roughly in a bounded cube (e.g. normalized shapes).
    The code is used for a locality-preserving sort.
    """

    pts = np.asarray(xyz)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"xyz must have shape (N,3); got {pts.shape}")

    # Normalize to [0, 1] for stable integer quantization.
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    scale = np.maximum(maxs - mins, 1e-8)
    u = (pts - mins) / scale

    # Quantize to 21 bits per axis => fits in 63 bits.
    q = np.clip((u * ((1 << 21) - 1)).astype(np.uint32), 0, (1 << 21) - 1)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]

    def _part1by2(n: NDArray[np.uint32]) -> NDArray[np.uint64]:
        n64 = n.astype(np.uint64)
        n64 = (n64 | (n64 << 32)) & 0x1F00000000FFFF
        n64 = (n64 | (n64 << 16)) & 0x1F0000FF0000FF
        n64 = (n64 | (n64 << 8)) & 0x100F00F00F00F00F
        n64 = (n64 | (n64 << 4)) & 0x10C30C30C30C30C3
        n64 = (n64 | (n64 << 2)) & 0x1249249249249249
        return n64

    return _part1by2(x) | (_part1by2(y) << 1) | (_part1by2(z) << 2)


def sort_by_ray_direction(ray_d: NDArray[np.floating]) -> NDArray[np.int32]:
    """Sort rays by (theta, phi) on the unit sphere."""

    d = np.asarray(ray_d)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"ray_d must have shape (N,3); got {d.shape}")

    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    theta = np.arctan2(dy, dx)  # azimuth
    dz_n = dz / (np.linalg.norm(d, axis=1) + 1e-8)
    dz_n = np.clip(dz_n, -1.0, 1.0)
    phi = np.arccos(dz_n)  # elevation
    return np.lexsort((phi, theta)).astype(np.int32)


def sort_rays_by_direction_fps(ray_d: NDArray[np.floating]) -> NDArray[np.int32]:
    """Order rays with FPS in direction space (S^2).

    Rays are first normalized to unit directions, then FPS is applied.
    This makes neighboring tokens angularly dissimilar, which helps reduce
    local-copy shortcuts in autoregressive ray prediction.
    """

    d = np.asarray(ray_d, dtype=np.float32)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"ray_d must have shape (N,3); got {d.shape}")
    if d.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    d_unit = d / np.maximum(nrm, 1e-8)

    from nepa3d.utils.fps import fps_order

    return fps_order(d_unit, int(d_unit.shape[0]))


def _as_float32_xyz(points: NDArray[np.floating]) -> NDArray[np.float32]:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3); got {pts.shape}")
    return pts.astype(np.float32, copy=False)


def _as_float32_dir(dirs: NDArray[np.floating]) -> NDArray[np.float32]:
    d = np.asarray(dirs)
    if d.ndim != 2 or d.shape[1] != 3:
        raise ValueError(f"dirs must have shape (N,3); got {d.shape}")
    return d.astype(np.float32, copy=False)


def _group_by_view_origin(
    ray_o: NDArray[np.floating],
    *,
    tol: float = 1e-6,
) -> NDArray[np.int32]:
    """Assign a view-id to each ray by grouping identical/near-identical origins."""

    ro = _as_float32_xyz(ray_o)
    if ro.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)

    step = float(max(tol, 1e-12))
    key = np.round(ro / step).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return inv.astype(np.int32, copy=False)


def _look_at(cam_pos: NDArray[np.floating]) -> NDArray[np.float32]:
    """Camera basis (rows: right, up, forward) that looks at origin.

    Mirrors nepa3d.data.preprocess_modelnet40.look_at.
    """

    c = np.asarray(cam_pos, dtype=np.float32).reshape(3)
    forward = -c
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    tmp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(tmp_up, forward)
    right = right / (np.linalg.norm(right) + 1e-8)

    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    return np.stack([right, up, forward], axis=0)


def sort_rays_by_view_raster(
    ray_o: NDArray[np.floating],
    ray_d: NDArray[np.floating],
    *,
    view_tol: float = 1e-6,
) -> NDArray[np.int32]:
    """Order rays by (view_id, raster(u,v)) where (u,v) are per-view camera coords.

    - view_id is derived by grouping identical ray origins.
    - view order is deterministic by sorting camera origins on the sphere (theta, phi).
    - inside each view, sort by (v, u) to mimic image raster.
    """

    ro = _as_float32_xyz(ray_o)
    rd = _as_float32_dir(ray_d)
    n = int(ro.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    vid = _group_by_view_origin(ro, tol=view_tol)
    n_view = int(vid.max()) + 1

    view_centers = np.zeros((n_view, 3), dtype=np.float32)
    for v in range(n_view):
        view_centers[v] = ro[vid == v].mean(axis=0)

    theta = np.arctan2(view_centers[:, 1], view_centers[:, 0]).astype(np.float32)
    z = np.clip(
        view_centers[:, 2] / (np.linalg.norm(view_centers, axis=1) + 1e-8),
        -1.0,
        1.0,
    )
    phi = np.arccos(z).astype(np.float32)
    view_order = np.lexsort((phi, theta)).astype(np.int32)

    out: list[int] = []
    eps = 1e-8
    for v in view_order.tolist():
        idx = np.nonzero(vid == v)[0]
        if idx.size == 0:
            continue
        r = _look_at(view_centers[v])
        d_cam = (r @ rd[idx].T).T
        denom = np.maximum(-d_cam[:, 2], eps)
        u = d_cam[:, 0] / denom
        vv = d_cam[:, 1] / denom
        local = idx[np.lexsort((u, vv))]
        out.extend(local.tolist())

    return np.asarray(out, dtype=np.int32)


def sort_rays_by_x_anchor(
    ray_o: NDArray[np.floating],
    ray_d: NDArray[np.floating],
    ray_hit: NDArray[np.floating] | NDArray[np.integer] | None,
    ray_t: NDArray[np.floating] | None,
    *,
    miss_t: float = 4.0,
    mode: Literal["morton", "fps", "random"] = "morton",
) -> NDArray[np.int32]:
    """Order rays by 3D anchor point x = o + t*d.

    If ray_hit is provided, miss rays use a fixed 'miss_t' to place an anchor in
    front of the camera; this avoids all misses collapsing to the origin.
    """

    ro = _as_float32_xyz(ray_o)
    rd = _as_float32_dir(ray_d)
    n = int(ro.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int32)

    if ray_t is None:
        t = np.full((n,), float(miss_t), dtype=np.float32)
    else:
        t = np.asarray(ray_t, dtype=np.float32).reshape(n)

    if ray_hit is not None:
        hit = np.asarray(ray_hit).reshape(n)
        miss = hit <= 0
        if np.any(miss):
            t = t.copy()
            t[miss] = float(miss_t)

    x = ro + rd * t[:, None]

    if mode == "morton":
        code = morton3d(x)
        return np.argsort(code).astype(np.int32)

    if mode == "fps":
        from nepa3d.utils.fps import fps_order

        return fps_order(x, n)

    if mode == "random":
        rng = np.random.default_rng(0)
        return rng.permutation(n).astype(np.int32)

    raise ValueError(f"Unknown mode for sort_rays_by_x_anchor: {mode}")


@dataclass
class Event:
    """A query-answer 'event' anchored at a 3D location."""

    kind: Literal["point", "ray", "missing_ray"]
    anchor: NDArray[np.float32]  # (3,)
    q: NDArray[np.float32]  # query token payload
    a: NDArray[np.float32]  # answer token payload
