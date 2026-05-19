from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def icosahedron_vertices(radius: float = 1.5) -> np.ndarray:
    """Return 12 camera positions on an icosahedron, normalized to radius."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for a, b in [(-1, phi), (1, phi), (-1, -phi), (1, -phi)]:
        verts.append([0, a, b])
        verts.append([a, b, 0])
        verts.append([b, 0, a])
    v = np.asarray(verts, dtype=np.float32)
    v = normalize(v) * float(radius)
    return v


def fibonacci_sphere(n: int, radius: float = 1.5) -> np.ndarray:
    pts = []
    offset = 2.0 / n
    inc = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = (i * offset - 1) + offset / 2
        r = math.sqrt(max(0.0, 1 - y * y))
        phi = i * inc
        pts.append([math.cos(phi) * r, y, math.sin(phi) * r])
    return np.asarray(pts, dtype=np.float32) * float(radius)


def look_at_frame(camera_pos: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
    """Camera-to-world frame. Columns are x(right), y(up), z(forward-to-object)."""
    if target is None:
        target = np.zeros(3, dtype=np.float32)
    forward = normalize((target - camera_pos).reshape(1, 3))[0]
    up0 = np.asarray([0, 1, 0], dtype=np.float32)
    if abs(float(np.dot(forward, up0))) > 0.95:
        up0 = np.asarray([1, 0, 0], dtype=np.float32)
    right = normalize(np.cross(up0, forward).reshape(1, 3))[0]
    up = normalize(np.cross(forward, right).reshape(1, 3))[0]
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def rot6d_from_matrix(r: np.ndarray) -> np.ndarray:
    """Zhou et al. 6D representation: first two columns."""
    return r[:, :2].reshape(-1).astype(np.float32)


@dataclass
class ViewGraph:
    camera_pos: np.ndarray       # [V, 3]
    camera_frame: np.ndarray     # [V, 3, 3]
    edges: np.ndarray            # [E, 2]
    action_id: np.ndarray        # [E]
    action_vec: np.ndarray       # [E, D]
    neighbor_index: np.ndarray   # [V, K]


def build_view_graph(num_views: int = 12, radius: float = 1.5, k_neighbors: int = 5,
                     mode: str = "icosahedron") -> ViewGraph:
    if mode == "icosahedron":
        if num_views != 12:
            raise ValueError("icosahedron mode currently uses exactly 12 views")
        pos = icosahedron_vertices(radius=radius)
    elif mode == "fibonacci":
        pos = fibonacci_sphere(num_views, radius=radius)
    else:
        raise ValueError(f"unknown mode: {mode}")
    frames = np.stack([look_at_frame(p) for p in pos], axis=0)
    dirs = normalize(-pos)
    dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    nbr = np.argsort(dist, axis=1)[:, :k_neighbors]
    edges = []
    avecs = []
    aids = []
    for i in range(pos.shape[0]):
        local_moves = []
        for j in nbr[i]:
            delta_world = pos[int(j)] - pos[i]
            delta_local = frames[i].T @ delta_world
            angle = math.atan2(float(delta_local[1]), float(delta_local[0]))
            local_moves.append((angle, int(j), delta_local))
        local_moves.sort(key=lambda item: item[0])
        for action_rank, (_, j, delta_local) in enumerate(local_moves):
            edges.append([i, int(j)])
            # Relative camera motion only. Do not include absolute source/target
            # directions: those leak view identity in the fixed 12-view graph.
            r_rel = frames[i].T @ frames[int(j)]
            rot6 = rot6d_from_matrix(r_rel)
            dist_ij = np.asarray([float(np.linalg.norm(delta_local))], dtype=np.float32)
            delta_unit = delta_local.astype(np.float32) / max(float(dist_ij[0]), 1e-8)
            avec = np.concatenate([rot6, delta_unit, dist_ij], axis=0).astype(np.float32)
            avecs.append(avec)
            # Shared local move class, not a globally unique edge id.
            aids.append(action_rank)
    return ViewGraph(
        camera_pos=pos.astype(np.float32),
        camera_frame=frames.astype(np.float32),
        edges=np.asarray(edges, dtype=np.int64),
        action_id=np.asarray(aids, dtype=np.int64),
        action_vec=np.asarray(avecs, dtype=np.float32),
        neighbor_index=nbr.astype(np.int64),
    )


def shortest_path_distance(num_views: int, edges: np.ndarray) -> np.ndarray:
    d = np.full((num_views, num_views), 10**9, dtype=np.int64)
    np.fill_diagonal(d, 0)
    for i, j in edges:
        d[i, j] = 1
    for k in range(num_views):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d
