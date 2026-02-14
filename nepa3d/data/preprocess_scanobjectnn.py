import argparse
import glob
import os
from collections import defaultdict

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def normalize_points(pc):
    pc = pc.astype(np.float32, copy=False)
    center = pc.mean(axis=0, keepdims=True)
    pc = pc - center
    scale = np.max(np.linalg.norm(pc, axis=1))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return pc / scale


def make_pools(pc_xyz, pt_pool=2000, ray_pool=1000, rng=None, pt_surface_ratio=0.5, pt_surface_sigma=0.02):
    if rng is None:
        rng = np.random
    kdt = cKDTree(pc_xyz)

    # Point-query pool: mix uniform and near-point samples (surface-biased).
    pt_pool = int(pt_pool)
    n_uni = int(round(float(1.0 - pt_surface_ratio) * pt_pool))
    n_surf = pt_pool - n_uni
    pts = []
    if n_uni > 0:
        pts.append(rng.uniform(-1.0, 1.0, size=(n_uni, 3)).astype(np.float32))
    if n_surf > 0:
        base = pc_xyz[rng.choice(pc_xyz.shape[0], size=n_surf, replace=(pc_xyz.shape[0] < n_surf))]
        jitter = rng.normal(scale=float(pt_surface_sigma), size=(n_surf, 3)).astype(np.float32)
        pts.append((base + jitter).astype(np.float32))
    pt_xyz_pool = np.concatenate(pts, axis=0) if len(pts) > 1 else pts[0]
    pt_xyz_pool = np.clip(pt_xyz_pool, -1.0, 1.0).astype(np.float32, copy=False)
    pt_dist_pool, _ = kdt.query(pt_xyz_pool, k=1)
    pt_dist_pool = pt_dist_pool.astype(np.float32, copy=False)

    # Point-cloud-only setting: ray channels are explicitly absent in training
    # via backend=pointcloud_noray or --force_missing_ray. Keep shape for API.
    ray_o_pool = np.zeros((ray_pool, 3), dtype=np.float32)
    ray_d_pool = np.zeros((ray_pool, 3), dtype=np.float32)
    ray_hit_pool = np.zeros((ray_pool,), dtype=np.float32)
    ray_t_pool = np.zeros((ray_pool,), dtype=np.float32)
    ray_n_pool = np.zeros((ray_pool, 3), dtype=np.float32)

    return {
        "pt_xyz_pool": pt_xyz_pool,
        "pt_dist_pool": pt_dist_pool,
        "ray_o_pool": ray_o_pool,
        "ray_d_pool": ray_d_pool,
        "ray_hit_pool": ray_hit_pool,
        "ray_t_pool": ray_t_pool,
        "ray_n_pool": ray_n_pool,
    }


def iter_h5_samples(h5_paths):
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f or "label" not in f:
                continue
            data = f["data"][:]
            label = f["label"][:]
        label = label.reshape(-1).astype(np.int64)
        for i in range(data.shape[0]):
            yield h5_path, i, data[i], int(label[i])


def find_split_files(scan_root, split):
    split = split.lower()
    all_h5 = glob.glob(os.path.join(scan_root, "**", "*.h5"), recursive=True)
    keys = ["train", "training"] if split == "train" else ["test"]
    paths = []
    for p in all_h5:
        name = os.path.basename(p).lower()
        parent = os.path.basename(os.path.dirname(p)).lower()
        if any(k in name for k in keys) or any(k in parent for k in keys):
            paths.append(p)
    return sorted(set(paths))


def _assert_unique_stems(h5_paths, allow_duplicate_stems=False):
    by_stem = defaultdict(list)
    for p in h5_paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        by_stem[stem].append(p)
    dup = {k: v for k, v in by_stem.items() if len(v) > 1}
    if dup and not allow_duplicate_stems:
        msg_lines = [
            "duplicate h5 stems detected across scan_root.",
            "This can silently overwrite cache outputs when multiple ScanObjectNN splits are mixed.",
            "Use a split-specific SCAN_ROOT (e.g., .../h5_files/main_split) or pass --allow_duplicate_stems to override.",
            f"duplicate_stems={len(dup)}",
        ]
        # Show a few concrete examples for quick debugging.
        for stem, paths in list(sorted(dup.items()))[:5]:
            msg_lines.append(f"  - {stem}: " + " | ".join(paths))
        raise RuntimeError("\n".join(msg_lines))


def _write_preprocess_meta(out_root, split, scan_root, h5_paths, pt_pool, ray_pool, pt_surface_ratio, pt_surface_sigma, seed):
    meta_dir = os.path.join(out_root, "_meta")
    os.makedirs(meta_dir, exist_ok=True)
    meta_path = os.path.join(meta_dir, f"scanobjectnn_{split}_source.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"scan_root={os.path.abspath(scan_root)}\n")
        f.write(f"split={split}\n")
        f.write(f"seed={int(seed)}\n")
        f.write(f"pt_pool={int(pt_pool)}\n")
        f.write(f"ray_pool={int(ray_pool)}\n")
        f.write(f"pt_surface_ratio={float(pt_surface_ratio)}\n")
        f.write(f"pt_surface_sigma={float(pt_surface_sigma)}\n")
        f.write(f"h5_count={len(h5_paths)}\n")
        for p in h5_paths:
            f.write(f"{os.path.abspath(p)}\n")


def preprocess_split(
    scan_root,
    out_root,
    split,
    pt_pool=2000,
    ray_pool=1000,
    seed=0,
    overwrite=False,
    pt_surface_ratio=0.5,
    pt_surface_sigma=0.02,
    allow_duplicate_stems=False,
):
    h5_paths = find_split_files(scan_root, split)
    if not h5_paths:
        raise FileNotFoundError(f"no {split} h5 files under: {scan_root}")

    _assert_unique_stems(h5_paths, allow_duplicate_stems=allow_duplicate_stems)
    _write_preprocess_meta(
        out_root,
        split,
        scan_root,
        h5_paths,
        pt_pool,
        ray_pool,
        pt_surface_ratio,
        pt_surface_sigma,
        seed,
    )

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    items = list(iter_h5_samples(h5_paths))
    print(f"{split}: {len(items)} samples from {len(h5_paths)} files")

    for h5_path, idx, pc, cls_id in tqdm(items, desc=f"preprocess {split}"):
        cls = f"class_{cls_id:03d}"
        stem = os.path.splitext(os.path.basename(h5_path))[0]
        name = f"{stem}_{idx:06d}"
        out_path = os.path.join(out_root, split, cls, f"{name}.npz")
        if (not overwrite) and os.path.exists(out_path):
            continue

        pc_xyz = normalize_points(pc)
        pc_n = np.zeros_like(pc_xyz, dtype=np.float32)
        pools = make_pools(pc_xyz, pt_pool=pt_pool, ray_pool=ray_pool, rng=rng, pt_surface_ratio=pt_surface_ratio, pt_surface_sigma=pt_surface_sigma)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            pc_xyz=pc_xyz.astype(np.float32, copy=False),
            pc_n=pc_n,
            **pools,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan_root", type=str, required=True, help="directory containing ScanObjectNN .h5 files")
    ap.add_argument("--out_root", type=str, required=True, help="output cache root")
    ap.add_argument("--split", type=str, choices=["train", "test", "all"], default="all")
    ap.add_argument("--pt_pool", type=int, default=2000)
    ap.add_argument("--ray_pool", type=int, default=1000)
    ap.add_argument("--pt_surface_ratio", type=float, default=0.5)
    ap.add_argument("--pt_surface_sigma", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--allow_duplicate_stems",
        action="store_true",
        help="allow duplicate h5 basenames across scan_root (unsafe; can overwrite outputs)",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.split in ("train", "all"):
        preprocess_split(
            args.scan_root,
            args.out_root,
            "train",
            pt_pool=args.pt_pool,
            ray_pool=args.ray_pool,
            seed=args.seed,
            overwrite=args.overwrite,
            pt_surface_ratio=args.pt_surface_ratio,
            pt_surface_sigma=args.pt_surface_sigma,
            allow_duplicate_stems=args.allow_duplicate_stems,
        )
    if args.split in ("test", "all"):
        preprocess_split(
            args.scan_root,
            args.out_root,
            "test",
            pt_pool=args.pt_pool,
            ray_pool=args.ray_pool,
            seed=args.seed + 12345,
            overwrite=args.overwrite,
            pt_surface_ratio=args.pt_surface_ratio,
            pt_surface_sigma=args.pt_surface_sigma,
            allow_duplicate_stems=args.allow_duplicate_stems,
        )


if __name__ == "__main__":
    main()
