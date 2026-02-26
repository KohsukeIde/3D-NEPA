#!/usr/bin/env python3
"""Sanity check for ScanObjectNN variant NPZ caches.

Checks:
- NPZ readability (corruption detection)
- required key presence
- class/sample counts for train/test
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from glob import glob
from typing import Dict, List

import numpy as np


REQUIRED_KEYS = (
    "pc_xyz",
    "pc_n",
    "pt_xyz_pool",
    "pt_dist_pool",
    "ray_o_pool",
    "ray_d_pool",
    "ray_t_pool",
    "ray_hit_pool",
)


def list_npz(cache_root: str, split: str) -> List[str]:
    return sorted(glob(os.path.join(cache_root, split, "*", "*.npz")))


def class_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def validate_file(path: str) -> Dict[str, object]:
    missing = []
    try:
        with np.load(path, allow_pickle=False) as d:
            keys = set(d.files)
            for k in REQUIRED_KEYS:
                if k not in keys:
                    missing.append(k)
    except Exception as e:  # noqa: BLE001 - explicit sanity report
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "missing_keys": []}
    return {"ok": True, "error": "", "missing_keys": missing}


def run_for_root(cache_root: str) -> Dict[str, object]:
    out: Dict[str, object] = {"cache_root": os.path.abspath(cache_root), "splits": {}}
    for split in ("train", "test"):
        paths = list_npz(cache_root, split)
        cls_counter = Counter(class_from_path(p) for p in paths)
        bad_files: List[Dict[str, str]] = []
        missing_key_files: List[Dict[str, object]] = []
        missing_key_counter = defaultdict(int)

        for p in paths:
            r = validate_file(p)
            if not r["ok"]:
                bad_files.append({"path": p, "error": str(r["error"])})
                continue
            miss = list(r["missing_keys"])
            if miss:
                missing_key_files.append({"path": p, "missing_keys": miss})
                for k in miss:
                    missing_key_counter[k] += 1

        out["splits"][split] = {
            "n_files": len(paths),
            "n_classes": len(cls_counter),
            "class_hist": dict(sorted(cls_counter.items())),
            "n_bad_files": len(bad_files),
            "n_missing_key_files": len(missing_key_files),
            "missing_key_hist": dict(sorted(missing_key_counter.items())),
            "bad_files_head": bad_files[:20],
            "missing_key_files_head": missing_key_files[:20],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Check ScanObjectNN variant NPZ cache sanity.")
    ap.add_argument(
        "--cache_roots",
        type=str,
        required=True,
        help="Comma-separated roots, e.g. data/scanobjectnn_obj_bg_v2,data/scanobjectnn_obj_only_v2",
    )
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    roots = [s.strip() for s in args.cache_roots.split(",") if s.strip()]
    report = {"reports": [run_for_root(r) for r in roots]}

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[write] {args.out_json}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

