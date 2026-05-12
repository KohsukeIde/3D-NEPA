#!/usr/bin/env python
"""Create deterministic ScanObjectNN train/heldout H5 roots for selection diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from object_ssl_common import repo_root_from_script


VARIANT_FILES = {
    "obj_bg": ("main_split", "training_objectdataset.h5", "test_objectdataset.h5", "standard"),
    "obj_only": ("main_split_nobg", "training_objectdataset.h5", "test_objectdataset.h5", "standard"),
    "pb_t50_rs": (
        "main_split",
        "training_objectdataset_augmentedrot_scale75.h5",
        "test_objectdataset_augmentedrot_scale75.h5",
        "hardest",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Build deterministic ScanObjectNN heldout split roots")
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_FILES))
    p.add_argument("--data-root", default="")
    p.add_argument("--out-root", required=True)
    p.add_argument("--split-json", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--heldout-frac", type=float, default=0.1)
    return p.parse_args()


def read_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        return np.asarray(f["data"], dtype=np.float32), np.asarray(f["label"], dtype=np.int64)


def write_h5(path: Path, data: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("label", data=labels)


def stratified_indices(labels: np.ndarray, seed: int, heldout_frac: float) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    labels_flat = labels.reshape(-1)
    train_idx = []
    heldout_idx = []
    for cls in sorted(np.unique(labels_flat).tolist()):
        idx = np.where(labels_flat == cls)[0]
        rng.shuffle(idx)
        n_hold = max(1, int(round(len(idx) * heldout_frac)))
        heldout_idx.extend(idx[:n_hold].tolist())
        train_idx.extend(idx[n_hold:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(heldout_idx)
    return train_idx, heldout_idx


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    data_root = Path(args.data_root) if args.data_root else root / "data" / "ScanObjectNN" / "h5_files"
    subdir, train_name, target_name, dataset_kind = VARIANT_FILES[args.variant]
    source_train = data_root / subdir / train_name
    source_target = data_root / subdir / target_name
    if not source_train.is_file():
        raise FileNotFoundError(source_train)
    if not source_target.is_file():
        raise FileNotFoundError(source_target)

    train_data, train_labels = read_h5(source_train)
    target_data, target_labels = read_h5(source_target)
    train_idx, heldout_idx = stratified_indices(train_labels, args.seed, args.heldout_frac)

    out_root = Path(args.out_root) / args.variant
    if dataset_kind == "hardest":
        split_train_name = "training_objectdataset_augmentedrot_scale75.h5"
        split_val_name = "test_objectdataset_augmentedrot_scale75.h5"
    else:
        split_train_name = "training_objectdataset.h5"
        split_val_name = "test_objectdataset.h5"
    write_h5(out_root / split_train_name, train_data[train_idx], train_labels[train_idx])
    write_h5(out_root / split_val_name, train_data[heldout_idx], train_labels[heldout_idx])
    write_h5(out_root / "target_test.h5", target_data, target_labels)

    split_path = Path(args.split_json)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if split_path.is_file():
        payload = json.loads(split_path.read_text())
    payload[args.variant] = {
        "seed": args.seed,
        "heldout_frac": args.heldout_frac,
        "source_train": str(source_train.resolve()),
        "source_target": str(source_target.resolve()),
        "split_root": str(out_root.resolve()),
        "train_indices": train_idx,
        "heldout_indices": heldout_idx,
        "n_train_sub": len(train_idx),
        "n_heldout": len(heldout_idx),
    }
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[done] wrote {out_root} and {split_path}")


if __name__ == "__main__":
    main()
