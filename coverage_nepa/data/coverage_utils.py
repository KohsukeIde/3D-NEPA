from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def stable_int_seed(*parts: object) -> int:
    raw = "::".join(str(p) for p in parts)
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def normalize_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    pts = pts[:, :3]
    pts = pts - pts.mean(axis=0, keepdims=True)
    s = np.max(np.linalg.norm(pts, axis=1))
    return pts / max(float(s), 1e-6)


def fps_numpy(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if len(points) <= n:
        if len(points) == n:
            return points.astype(np.float32)
        idx = rng.choice(len(points), size=n - len(points), replace=True)
        return np.concatenate([points, points[idx]], axis=0).astype(np.float32)
    start = int(rng.integers(len(points)))
    selected = np.empty(n, dtype=np.int64)
    selected[0] = start
    dist = np.full(len(points), np.inf, dtype=np.float32)
    last = points[start]
    for i in range(1, n):
        d = np.sum((points - last) ** 2, axis=1)
        dist = np.minimum(dist, d)
        selected[i] = int(np.argmax(dist))
        last = points[selected[i]]
    return points[selected].astype(np.float32)


def random_sample_fixed(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    replace = len(points) < n
    idx = rng.choice(len(points), size=n, replace=replace)
    return points[idx].astype(np.float32)


def voxel_keys(points: np.ndarray, voxel_size: float = 0.04) -> set[tuple[int, int, int]]:
    if len(points) == 0:
        return set()
    q = np.floor((points + 1.5) / voxel_size).astype(np.int32)
    return {tuple(row.tolist()) for row in q}


def coverage_order_greedy(views: np.ndarray, voxel_size: float = 0.04) -> tuple[list[int], list[int]]:
    v = views.shape[0]
    view_keys = [voxel_keys(views[i], voxel_size=voxel_size) for i in range(v)]
    used: set[int] = set()
    covered: set[tuple[int, int, int]] = set()
    order: list[int] = []
    gains: list[int] = []
    for _ in range(v):
        best_i = None
        best_gain = -1
        for i in range(v):
            if i in used:
                continue
            gain = len(view_keys[i] - covered)
            if gain > best_gain:
                best_i = i
                best_gain = gain
        assert best_i is not None
        used.add(best_i)
        order.append(best_i)
        gains.append(best_gain)
        covered.update(view_keys[best_i])
    return order, gains


def coverage_order_random(num_views: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    order = rng.permutation(num_views).astype(int).tolist()
    return order, [0 for _ in order]


def raw_stats_features(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros(22, dtype=np.float32)
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    bbox = mx - mn
    cov = np.cov(pts.T) if len(pts) > 3 else np.eye(3, dtype=np.float32)
    eig = np.linalg.eigvalsh(cov).astype(np.float32)
    norms = np.linalg.norm(pts - mean[None, :], axis=1)
    out = np.concatenate([
        mean, std, mn, mx, bbox, eig,
        np.asarray([norms.mean(), norms.std(), norms.min(), norms.max()], dtype=np.float32),
    ])
    return out.astype(np.float32)


def find_view_npz(cache_root: Path) -> list[Path]:
    rows = []
    manifest = cache_root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        for row in payload.get("rows", []):
            rel = row.get("cache_path")
            if rel:
                p = cache_root / rel
                if p.exists():
                    rows.append(p)
        if rows:
            return rows
    return sorted([p for p in cache_root.rglob("*.npz") if p.name not in {"view_graph.npz"}])
