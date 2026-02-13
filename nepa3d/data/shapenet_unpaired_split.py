import argparse
import json
import os
from collections import defaultdict

import numpy as np

from .modelnet40_index import list_npz


def _key_from_path(npz_path: str) -> str:
    synset = npz_path.split(os.sep)[-2]
    inst = os.path.splitext(os.path.basename(npz_path))[0]
    return f"{synset}/{inst}"


def _stratified_assign(train_paths, ratios, seed):
    groups = defaultdict(list)
    for p in train_paths:
        synset = p.split(os.sep)[-2]
        groups[synset].append(p)

    w = np.asarray(ratios, dtype=np.float64)
    if w.ndim != 1 or w.size != 3:
        raise ValueError("--ratios must have exactly 3 values: mesh pc udf")
    if (w < 0).any() or not np.isfinite(w).all() or float(w.sum()) <= 0.0:
        raise ValueError("--ratios must be non-negative and sum > 0")
    w = w / w.sum()

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    out = {}
    split_names = ("train_mesh", "train_pc", "train_udf")

    for synset, paths in sorted(groups.items()):
        paths = sorted(paths)
        rng.shuffle(paths)
        n = len(paths)

        n0 = int(round(w[0] * n))
        n1 = int(round(w[1] * n))
        n0 = min(max(n0, 0), n)
        n1 = min(max(n1, 0), n - n0)
        n2 = n - n0 - n1

        if n >= 3:
            counts = [n0, n1, n2]
            for i in range(3):
                if counts[i] == 0:
                    j = int(np.argmax(counts))
                    if counts[j] > 1:
                        counts[j] -= 1
                        counts[i] += 1
            n0, n1, n2 = counts

        buckets = (
            paths[:n0],
            paths[n0 : n0 + n1],
            paths[n0 + n1 :],
        )
        for split_name, bucket in zip(split_names, buckets):
            for p in bucket:
                out[_key_from_path(p)] = split_name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--eval_split", type=str, default="test")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.34, 0.33, 0.33),
        metavar=("MESH", "PC", "UDF"),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache_root = os.path.abspath(args.cache_root)
    train_paths = list_npz(cache_root, args.train_split)
    eval_paths = list_npz(cache_root, args.eval_split)
    if not train_paths:
        raise RuntimeError(f"no npz under {cache_root}/{args.train_split}")
    if not eval_paths:
        raise RuntimeError(f"no npz under {cache_root}/{args.eval_split}")

    mapping = _stratified_assign(train_paths, args.ratios, args.seed)
    records = []

    for p in sorted(train_paths):
        key = _key_from_path(p)
        split = mapping[key]
        records.append(
            {
                "key": key,
                "split": split,
                "src_split": args.train_split,
                "src_relpath": os.path.relpath(p, start=cache_root),
            }
        )
    for p in sorted(eval_paths):
        key = _key_from_path(p)
        records.append(
            {
                "key": key,
                "split": "eval",
                "src_split": args.eval_split,
                "src_relpath": os.path.relpath(p, start=cache_root),
            }
        )

    counts = defaultdict(int)
    for r in records:
        counts[r["split"]] += 1

    out = {
        "meta": {
            "cache_root": cache_root,
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "seed": int(args.seed),
            "ratios_mesh_pc_udf": [float(x) for x in args.ratios],
            "counts": dict(counts),
        },
        "records": records,
    }

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[split] wrote {out_json}")
    print(f"[split] counts: {dict(counts)}")


if __name__ == "__main__":
    main()
