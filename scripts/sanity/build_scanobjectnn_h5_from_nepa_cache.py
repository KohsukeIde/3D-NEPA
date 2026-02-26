#!/usr/bin/env python3
"""Build Point-MAE-style ScanObjectNN H5 files from NEPA NPZ variant cache.

Input cache layout:
  <cache_root>/{train,test}/class_XXX/*.npz

Output H5 layout (Point-MAE expected filenames):
  variant=pb_t50_rs:
    <out_root>/training_objectdataset_augmentedrot_scale75.h5
    <out_root>/test_objectdataset_augmentedrot_scale75.h5
  variant=obj_bg|obj_only:
    <out_root>/training_objectdataset.h5
    <out_root>/test_objectdataset.h5
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Tuple

import h5py
import numpy as np


def _list_npz(cache_root: str, split: str) -> List[str]:
    return sorted(glob.glob(os.path.join(cache_root, split, "class_*", "*.npz")))


def _class_id_from_path(path: str) -> int:
    cls_name = os.path.basename(os.path.dirname(path))
    if not cls_name.startswith("class_"):
        raise ValueError(f"unexpected class folder: {cls_name} ({path})")
    return int(cls_name.split("_", 1)[1])


def _load_split(cache_root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    paths = _list_npz(cache_root, split)
    if not paths:
        raise RuntimeError(f"no npz files found: cache_root={cache_root} split={split}")

    pts_all: List[np.ndarray] = []
    lbl_all: List[int] = []

    for p in paths:
        with np.load(p) as z:
            if "pc_xyz" not in z:
                raise KeyError(f"missing key 'pc_xyz' in {p}")
            pc_xyz = z["pc_xyz"].astype(np.float32)
        if pc_xyz.ndim != 2 or pc_xyz.shape[1] != 3:
            raise ValueError(f"invalid pc_xyz shape {pc_xyz.shape} in {p}")
        pts_all.append(pc_xyz)
        lbl_all.append(_class_id_from_path(p))

    points = np.stack(pts_all, axis=0).astype(np.float32)
    labels = np.asarray(lbl_all, dtype=np.int64).reshape(-1, 1)
    return points, labels


def _resolve_names(variant: str) -> Tuple[str, str]:
    if variant == "pb_t50_rs":
        return (
            "training_objectdataset_augmentedrot_scale75.h5",
            "test_objectdataset_augmentedrot_scale75.h5",
        )
    if variant in ("obj_bg", "obj_only"):
        return ("training_objectdataset.h5", "test_objectdataset.h5")
    raise ValueError(f"unsupported variant={variant}")


def _write_h5(path: str, points: np.ndarray, labels: np.ndarray, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=points, compression="gzip")
        f.create_dataset("label", data=labels, compression="gzip")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Point-MAE ScanObjectNN H5 from NEPA cache.")
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--variant", required=True, choices=["pb_t50_rs", "obj_bg", "obj_only"])
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    train_name, test_name = _resolve_names(args.variant)
    train_points, train_labels = _load_split(args.cache_root, "train")
    test_points, test_labels = _load_split(args.cache_root, "test")

    train_h5 = os.path.join(args.out_root, train_name)
    test_h5 = os.path.join(args.out_root, test_name)
    ow = bool(args.overwrite)
    _write_h5(train_h5, train_points, train_labels, ow)
    _write_h5(test_h5, test_points, test_labels, ow)

    print(
        "[done] variant={} cache_root={} out_root={} train_shape={} test_shape={}".format(
            args.variant,
            os.path.abspath(args.cache_root),
            os.path.abspath(args.out_root),
            tuple(train_points.shape),
            tuple(test_points.shape),
        )
    )
    print(f"[files] {train_h5}")
    print(f"[files] {test_h5}")


if __name__ == "__main__":
    main()

