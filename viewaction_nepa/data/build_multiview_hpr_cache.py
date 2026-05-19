#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from viewaction_nepa.data.camera_graph import build_view_graph


def sample_rows(pts: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or len(pts) <= max_points:
        return pts
    idx = rng.choice(len(pts), size=max_points, replace=False)
    idx.sort()
    return pts[idx]


def load_ply_points(path: Path, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Load xyz from a PLY file, sampling before materializing huge room-scale clouds."""
    try:
        from plyfile import PlyData
    except Exception as exc:
        raise RuntimeError("PLY input requires the `plyfile` package") from exc

    ply = PlyData.read(str(path), mmap=True)
    if "vertex" not in ply:
        raise ValueError(f"no vertex element in {path}")
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"PLY vertex has no x/y/z fields: {path}")
    n = len(vertex)
    if max_points > 0 and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        idx.sort()
        vertex = vertex[idx]
    pts = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return pts


def load_points(path: Path, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if path.suffix == ".npy":
        pts = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        for k in ["points", "pc", "xyz", "arr_0"]:
            if k in data:
                pts = data[k]
                break
        else:
            raise ValueError(f"no point array found in {path}")
    elif path.suffix == ".ply":
        pts = load_ply_points(path, max_points=max_points, rng=rng)
    else:
        raise ValueError(f"unsupported file {path}")
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"bad point shape {pts.shape} in {path}")
    pts = pts[:, :3]
    return sample_rows(pts, max_points=max_points, rng=rng)


def normalize_points(pts: np.ndarray, mode: str = "unit_sphere") -> np.ndarray:
    pts = pts.astype(np.float32)
    pts = pts - pts.mean(axis=0, keepdims=True)
    if mode == "unit_sphere":
        s = np.max(np.linalg.norm(pts, axis=1))
    elif mode == "unit_cube":
        s = np.max(np.abs(pts))
    else:
        raise ValueError(mode)
    return pts / max(float(s), 1e-6)


def make_camera_basis(camera_pos: np.ndarray):
    forward = -camera_pos / (np.linalg.norm(camera_pos) + 1e-8)
    up0 = np.asarray([0, 1, 0], dtype=np.float32)
    if abs(float(np.dot(forward, up0))) > 0.95:
        up0 = np.asarray([1, 0, 0], dtype=np.float32)
    right = np.cross(up0, forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)
    return right, up, forward


def visible_zbuffer(points: np.ndarray, camera_pos: np.ndarray, grid: int = 96, margin: float = 1.05) -> np.ndarray:
    """Fast z-buffer-style visibility fallback.

    Projects points into a camera plane and keeps the closest point per grid cell.
    This is not a physical renderer, but it gives stable partial views without Open3D.
    """
    right, up, forward = make_camera_basis(camera_pos)
    rel = points - camera_pos.reshape(1, 3)
    x = rel @ right
    y = rel @ up
    z = rel @ forward
    valid = z > 0
    if valid.sum() < 16:
        return points[:0]
    x, y, z = x[valid], y[valid], z[valid]
    idx_orig = np.nonzero(valid)[0]
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-6) * margin
    xi = np.clip(((x / scale + 1) * 0.5 * (grid - 1)).astype(np.int64), 0, grid - 1)
    yi = np.clip(((y / scale + 1) * 0.5 * (grid - 1)).astype(np.int64), 0, grid - 1)
    key = yi * grid + xi
    order = np.argsort(z)  # closest first along camera ray
    seen = set()
    keep = []
    for o in order:
        k = int(key[o])
        if k not in seen:
            seen.add(k)
            keep.append(idx_orig[o])
    return points[np.asarray(keep, dtype=np.int64)]


def fps_numpy(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if len(points) <= n:
        extra = rng.choice(len(points), size=n - len(points), replace=True) if len(points) < n else []
        out = points if len(points) == n else np.concatenate([points, points[extra]], axis=0)
        return out.astype(np.float32)
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


def find_point_files(root: Path, exts: set[str], pattern: str = "*") -> list[Path]:
    return sorted([p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() in exts])


def slugify(raw: str, max_len: int = 96) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not text:
        text = "item"
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{text[:max_len-11]}_{digest}"


def category_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if len(rel.parts) > 1:
        return slugify(rel.parts[0])
    # ShapeNet55 shapenet_pc is a flat directory: <synset>-<shape_id>.npy.
    if "-" in path.stem:
        return slugify(path.stem.split("-", 1)[0])
    return "unknown"


def shape_id_from_path(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return slugify("__".join(rel.parts), max_len=140)


def parse_split_files(split_specs: list[str], input_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for spec in split_specs:
        if ":" in spec:
            split, raw_path = spec.split(":", 1)
        else:
            split = Path(spec).stem
            raw_path = spec
        split = split.strip()
        path = Path(raw_path)
        if not path.is_absolute():
            path = input_root / path
        if not path.exists():
            raise FileNotFoundError(f"split file not found: {path}")
        for line in path.read_text().splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            mapping[Path(item).name] = split
    return mapping


def aggregate_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    visible = [r for r in ok if "min_visible" in r]
    if visible:
        min_vals = np.asarray([r["min_visible"] for r in visible], dtype=np.float64)
        mean_vals = np.asarray([r["mean_visible"] for r in visible], dtype=np.float64)
        max_vals = np.asarray([r["max_visible"] for r in visible], dtype=np.float64)
        vis_summary = {
            "min_visible": int(min_vals.min()),
            "mean_visible": float(mean_vals.mean()),
            "max_visible": int(max_vals.max()),
            "per_shape_mean_visible_min": float(mean_vals.min()),
            "per_shape_mean_visible_max": float(mean_vals.max()),
        }
    else:
        vis_summary = {}
    return {
        "num_rows": len(rows),
        "num_ok": len(ok),
        "num_error": len(rows) - len(ok),
        **vis_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--num-views", type=int, default=12)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--k-neighbors", type=int, default=5)
    ap.add_argument("--points-per-view", type=int, default=1024)
    ap.add_argument("--source-sample-points", type=int, default=200000)
    ap.add_argument("--normalize", default="unit_sphere", choices=["unit_sphere", "unit_cube"])
    ap.add_argument("--camera-mode", default="icosahedron", choices=["icosahedron", "fibonacci"])
    ap.add_argument("--include-exts", default=".npy,.npz,.ply")
    ap.add_argument("--file-pattern", default="*")
    ap.add_argument("--split-file", action="append", default=[],
                    help="Optional split:path file. Lines are filenames or relative paths.")
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--shuffle-files", action="store_true")
    ap.add_argument("--min-visible", type=int, default=128)
    ap.add_argument("--on-low-visible", default="warn", choices=["warn", "fail"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    inp = Path(args.input_root)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    exts = {x.strip().lower() for x in args.include_exts.split(",") if x.strip()}
    split_map = parse_split_files(args.split_file, inp) if args.split_file else {}
    files = find_point_files(inp, exts=exts, pattern=args.file_pattern)
    if split_map:
        files = [p for p in files if p.name in split_map]
    if args.shuffle_files:
        files = list(rng.permutation(np.asarray(files, dtype=object)))
    if args.max_shapes:
        files = files[: args.max_shapes]
    graph = build_view_graph(args.num_views, args.radius, args.k_neighbors, mode=args.camera_mode)
    meta = {
        "num_views": args.num_views,
        "radius": args.radius,
        "k_neighbors": args.k_neighbors,
        "points_per_view": args.points_per_view,
        "source_sample_points": args.source_sample_points,
        "normalize": args.normalize,
        "camera_mode": args.camera_mode,
        "include_exts": sorted(exts),
        "file_pattern": args.file_pattern,
        "split_files": args.split_file,
        "num_shapes": len(files),
        "input_root": str(inp),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    np.savez_compressed(out / "view_graph.npz",
                        camera_pos=graph.camera_pos,
                        camera_frame=graph.camera_frame,
                        edges=graph.edges,
                        action_id=graph.action_id,
                        action_vec=graph.action_vec,
                        neighbor_index=graph.neighbor_index)
    stats = []
    manifest_rows = []
    for idx, p in enumerate(files):
        try:
            pts = normalize_points(load_points(p, args.source_sample_points, rng), args.normalize)
            views = []
            visible_counts = []
            for cam in graph.camera_pos:
                vis = visible_zbuffer(pts, cam)
                visible_counts.append(len(vis))
                views.append(fps_numpy(vis, args.points_per_view, rng))
            sid = shape_id_from_path(p, inp)
            cat = category_from_path(p, inp)
            source_split = split_map.get(p.name, "all")
            target_dir = out / cat
            target_dir.mkdir(parents=True, exist_ok=True)
            cache_path = target_dir / f"{sid}.npz"
            np.savez_compressed(cache_path,
                                views=np.stack(views, axis=0).astype(np.float32),
                                visible_counts=np.asarray(visible_counts, dtype=np.int64),
                                shape_id=np.asarray(sid),
                                category=np.asarray(cat),
                                source_split=np.asarray(source_split),
                                source_path=np.asarray(str(p)))
            row = {
                "shape_id": sid,
                "canonical_shape_id": sid,
                "category": cat,
                "source_split": source_split,
                "source_path": str(p),
                "cache_path": str(cache_path.relative_to(out)),
                "source_path_hash": hashlib.sha1(str(p).encode("utf-8")).hexdigest(),
                "min_visible": int(np.min(visible_counts)),
                "mean_visible": float(np.mean(visible_counts)),
                "max_visible": int(np.max(visible_counts)),
            }
            stats.append(row)
            manifest_rows.append(row)
        except Exception as e:
            stats.append({"shape_id": shape_id_from_path(p, inp), "source_path": str(p), "error": str(e)})
        if (idx + 1) % 100 == 0:
            print(f"[build] {idx+1}/{len(files)}")
    summary = aggregate_stats(stats)
    (out / "build_stats.json").write_text(json.dumps({"summary": summary, "rows": stats}, indent=2))
    (out / "manifest.json").write_text(json.dumps({"metadata": meta, "rows": manifest_rows}, indent=2))
    print(f"[done] wrote cache to {out}")
    print(json.dumps(summary, indent=2))
    if summary.get("num_ok", 0) and summary.get("min_visible", 0) < args.min_visible:
        msg = f"min_visible={summary.get('min_visible')} is below threshold {args.min_visible}"
        if args.on_low_visible == "fail":
            raise SystemExit(f"[error] {msg}")
        print(f"[warn] {msg}")


if __name__ == "__main__":
    main()
