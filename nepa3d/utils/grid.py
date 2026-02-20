"""Grid sampling utilities (NumPy).

Used for:
- sampling UDF grids at arbitrary points
- estimating pseudo-normals via finite-difference gradients

Coordinate convention:
- Shapes are normalized to (approximately) [-1, 1]^3.
- A grid with resolution G stores values at voxel centers:
    x_i = -1 + (i + 0.5) * (2/G),  i=0..G-1

This module is intentionally dependency-light (NumPy only).
"""

from __future__ import annotations

import numpy as np


def make_grid_centers_np(grid_res: int) -> np.ndarray:
    """Return voxel-center coordinates for a cubic grid in [-1, 1]^3.

    Args:
        grid_res: cubic resolution G

    Returns:
        centers: (G, G, G, 3) float32 array
    """

    g = int(grid_res)
    if g <= 0:
        raise ValueError(f"grid_res must be positive, got {grid_res}")
    lin = -1.0 + (np.arange(g, dtype=np.float32) + 0.5) * (2.0 / float(g))
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)


def xyz_to_grid_coords(xyz: np.ndarray, grid_res: int) -> np.ndarray:
    """Convert world coords in [-1,1] to continuous grid index coords.

    Returns coordinates in "cell-center" index space where:
      xyz=-1+0.5*voxel -> u=0
      xyz=+1-0.5*voxel -> u=G-1
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    g = float(grid_res)
    # voxel size in world
    # voxel = 2.0 / g
    # u = (xyz - (-1 + 0.5*voxel)) / voxel
    #   = (xyz + 1) / voxel - 0.5
    u = (xyz + 1.0) * (g / 2.0) - 0.5
    return u


def trilinear_sample(grid: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Trilinearly sample a cubic grid at world-space xyz.

    Args:
        grid: (G,G,G) float array
        xyz: (...,3) float array in [-1,1]

    Returns:
        values: (...) float32
    """
    grid = np.asarray(grid)
    assert grid.ndim == 3 and grid.shape[0] == grid.shape[1] == grid.shape[2]
    g = grid.shape[0]

    xyz = np.asarray(xyz, dtype=np.float32)
    u = xyz_to_grid_coords(xyz.reshape(-1, 3), g)

    u0 = np.floor(u).astype(np.int32)
    w = u - u0.astype(np.float32)
    u1 = u0 + 1

    # clamp
    u0 = np.clip(u0, 0, g - 1)
    u1 = np.clip(u1, 0, g - 1)

    x0, y0, z0 = u0[:, 0], u0[:, 1], u0[:, 2]
    x1, y1, z1 = u1[:, 0], u1[:, 1], u1[:, 2]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]

    # 8 corners
    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]

    c00 = c000 * (1 - wx) + c100 * wx
    c10 = c010 * (1 - wx) + c110 * wx
    c01 = c001 * (1 - wx) + c101 * wx
    c11 = c011 * (1 - wx) + c111 * wx

    c0 = c00 * (1 - wy) + c10 * wy
    c1 = c01 * (1 - wy) + c11 * wy

    c = c0 * (1 - wz) + c1 * wz

    return c.reshape(xyz.shape[:-1]).astype(np.float32)


def udf_pseudonormal(grid: np.ndarray, xyz: np.ndarray, eps: float | None = None) -> np.ndarray:
    """Estimate pseudo-normal as negative gradient of unsigned distance.

    Args:
        grid: (G,G,G)
        xyz: (...,3)
        eps: finite-difference step in world units. Default: one voxel.

    Returns:
        n: (...,3) float32, unit vector (0 if grad is tiny)
    """
    grid = np.asarray(grid)
    g = grid.shape[0]
    if eps is None:
        eps = 2.0 / float(g)
    xyz = np.asarray(xyz, dtype=np.float32)
    flat = xyz.reshape(-1, 3)

    ex = np.array([eps, 0.0, 0.0], dtype=np.float32)
    ey = np.array([0.0, eps, 0.0], dtype=np.float32)
    ez = np.array([0.0, 0.0, eps], dtype=np.float32)

    fxp = trilinear_sample(grid, flat + ex)
    fxm = trilinear_sample(grid, flat - ex)
    fyp = trilinear_sample(grid, flat + ey)
    fym = trilinear_sample(grid, flat - ey)
    fzp = trilinear_sample(grid, flat + ez)
    fzm = trilinear_sample(grid, flat - ez)

    gx = (fxp - fxm) / (2.0 * eps)
    gy = (fyp - fym) / (2.0 * eps)
    gz = (fzp - fzm) / (2.0 * eps)

    grad = np.stack([gx, gy, gz], axis=-1)
    # unsigned distance increases away from surface, so direction-to-surface is -grad
    n = -grad
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(norm > 1e-8, n / norm, 0.0)
    return n.reshape(xyz.shape).astype(np.float32)


def udf_local_std(udf_grid: np.ndarray, xyz: np.ndarray, eps: float = None) -> np.ndarray:
    """A cheap uncertainty proxy: local std of UDF around xyz.

    We sample the UDF at +/-eps along each axis and return the standard deviation
    across those 6 samples (+ center).
    """
    if eps is None:
        g = int(udf_grid.shape[0])
        eps = 2.0 / float(g)

    xyz = np.asarray(xyz, dtype=np.float32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [eps, 0.0, 0.0],
            [-eps, 0.0, 0.0],
            [0.0, eps, 0.0],
            [0.0, -eps, 0.0],
            [0.0, 0.0, eps],
            [0.0, 0.0, -eps],
        ],
        dtype=np.float32,
    )
    vals = []
    for off in offsets:
        vals.append(trilinear_sample(udf_grid, xyz + off))
    v = np.stack(vals, axis=-1)  # (N, 7)
    return v.std(axis=-1).astype(np.float32)
