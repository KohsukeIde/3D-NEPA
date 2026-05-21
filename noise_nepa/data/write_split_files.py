#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import discover_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    if args.train_ratio <= 0.0 or args.val_ratio <= 0.0 or args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("[error] require train_ratio > 0, val_ratio > 0, and train_ratio + val_ratio < 1")
    files = discover_files(root, max_shapes=args.max_shapes)
    rels = [str(p.resolve().relative_to(root)) for p in files]
    if len(rels) < 3:
        raise SystemExit(f"[error] need at least 3 files for disjoint train/val/test splits, got {len(rels)}")
    rng = random.Random(args.seed)
    rng.shuffle(rels)
    n = len(rels)
    n_train = max(1, int(round(n * args.train_ratio)))
    n_val = max(1, int(round(n * args.val_ratio)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train <= 0 or n_val <= 0 or n_train + n_val >= n:
        raise SystemExit(
            f"[error] cannot create non-empty disjoint splits from {n} files "
            f"(train={n_train}, val={n_val})"
        )
    train = rels[:n_train]
    val = rels[n_train : n_train + n_val]
    test = rels[n_train + n_val :]
    if not test:
        raise SystemExit("[error] empty test split after ratio rounding")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        (out / f"{name}.txt").write_text("\n".join(rows) + "\n")
    meta = {
        "data_root": str(root),
        "max_shapes": args.max_shapes,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
