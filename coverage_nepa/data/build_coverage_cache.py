#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from coverage_nepa.data.coverage_utils import (
    coverage_order_greedy,
    coverage_order_random,
    find_view_npz,
    fps_numpy,
    random_sample_fixed,
    raw_stats_features,
    stable_int_seed,
)


def as_str(x) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        return str(x)
    return str(x)


def build_one(path: Path, out_path: Path, args) -> dict:
    data = np.load(path, allow_pickle=True)
    if "views" not in data:
        raise ValueError(f"{path} has no 'views' key")
    views = np.asarray(data["views"], dtype=np.float32)
    if views.ndim != 3 or views.shape[-1] < 3:
        raise ValueError(f"bad views shape {views.shape} in {path}")
    views = views[:, :, :3]
    num_views = views.shape[0]
    sid = as_str(data["shape_id"]) if "shape_id" in data else path.stem
    cat = as_str(data["category"]) if "category" in data else path.parent.name
    split = as_str(data["source_split"]) if "source_split" in data else "all"
    rng = np.random.default_rng(stable_int_seed(args.seed, sid, args.order_mode))
    if args.order_mode == "greedy_new_coverage":
        order, gains = coverage_order_greedy(views, voxel_size=args.voxel_size)
    elif args.order_mode == "random":
        order, gains = coverage_order_random(num_views, rng)
    else:
        raise ValueError(args.order_mode)
    if args.num_levels > 0:
        order = order[: args.num_levels]
        gains = gains[: args.num_levels]
    coverage = []
    raw_counts = []
    raw_features = []
    union = np.zeros((0, 3), dtype=np.float32)
    for li, vi in enumerate(order):
        union = np.concatenate([union, views[vi]], axis=0)
        raw_counts.append(int(len(union)))
        raw_features.append(raw_stats_features(union))
        if args.sample_mode == "fps":
            fixed = fps_numpy(union, args.points_per_level, rng)
        else:
            fixed = random_sample_fixed(union, args.points_per_level, rng)
        coverage.append(fixed)
    coverage_np = np.stack(coverage, axis=0).astype(np.float32)
    raw_features_np = np.stack(raw_features, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coverage=coverage_np,
        levels=np.arange(len(order), dtype=np.int64),
        view_order=np.asarray(order, dtype=np.int64),
        new_voxel_gains=np.asarray(gains, dtype=np.int64),
        raw_union_count=np.asarray(raw_counts, dtype=np.int64),
        raw_stats=np.asarray(raw_features_np, dtype=np.float32),
        shape_id=np.asarray(sid),
        category=np.asarray(cat),
        source_split=np.asarray(split),
        source_path=np.asarray(str(path)),
    )
    return {
        "shape_id": sid,
        "category": cat,
        "source_split": split,
        "cache_path": str(out_path.relative_to(args.output_root)),
        "source_path": str(path),
        "num_levels": int(len(order)),
        "points_per_level": int(args.points_per_level),
        "raw_union_count_min": int(min(raw_counts)),
        "raw_union_count_max": int(max(raw_counts)),
        "view_order": order,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view-cache-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--order-mode", default="greedy_new_coverage", choices=["greedy_new_coverage", "random"])
    ap.add_argument("--num-levels", type=int, default=6)
    ap.add_argument("--points-per-level", type=int, default=1024)
    ap.add_argument("--sample-mode", default="fps", choices=["fps", "random"])
    ap.add_argument("--voxel-size", type=float, default=0.04)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--shuffle-before-limit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output_root = Path(args.output_root)
    view_root = Path(args.view_cache_root)
    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    files = find_view_npz(view_root)
    if args.shuffle_before_limit:
        rng = np.random.default_rng(args.seed)
        files = list(files)
        rng.shuffle(files)
    if args.max_shapes:
        files = files[: args.max_shapes]
    rows = []
    errors = []
    for i, p in enumerate(files):
        try:
            with np.load(p, allow_pickle=True) as data:
                cat = as_str(data["category"]) if "category" in data else p.parent.name
                sid = as_str(data["shape_id"]) if "shape_id" in data else p.stem
            out_path = out_root / cat / f"{sid}.npz"
            rows.append(build_one(p, out_path, args))
        except Exception as exc:
            errors.append({"path": str(p), "error": str(exc)})
        if (i + 1) % 100 == 0:
            print(f"[coverage] {i+1}/{len(files)}")
    meta = {
        "view_cache_root": str(view_root),
        "output_root": str(out_root),
        "order_mode": args.order_mode,
        "num_levels": args.num_levels,
        "points_per_level": args.points_per_level,
        "sample_mode": args.sample_mode,
        "voxel_size": args.voxel_size,
        "max_shapes": args.max_shapes,
        "shuffle_before_limit": bool(args.shuffle_before_limit),
        "seed": args.seed,
        "num_ok": len(rows),
        "num_error": len(errors),
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out_root / "manifest.json").write_text(json.dumps({"metadata": meta, "rows": rows, "errors": errors}, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
