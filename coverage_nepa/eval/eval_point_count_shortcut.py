#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from coverage_nepa.data.coverage_dataset import CoverageStateDataset
from coverage_nepa.data.coverage_utils import raw_stats_features, voxel_keys


def train_linear_eval(train_x, train_y, test_x, test_y, seed=0, epochs=200):
    torch.manual_seed(seed)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    test_x = torch.tensor(test_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_y = torch.tensor(test_y, dtype=torch.long)
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    num_classes = int(max(train_y.max().item(), test_y.max().item()) + 1)
    head = nn.Linear(train_x.shape[1], num_classes)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(train_x), train_y)
        loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((head(test_x).argmax(1) == test_y).float().mean()) if len(test_y) else 0.0
    return acc


def collect(ds: CoverageStateDataset):
    rows = {
        "raw_stats": [],
        "input_stats": [],
        "raw_count": [],
        "fixed_count": [],
        "unique_voxel_count": [],
        "duplicate_rate": [],
        "levels": [],
    }
    for i in range(len(ds)):
        row = ds[i]
        pts = row["points"].numpy()
        unique_exact = np.unique(np.round(pts, 6), axis=0).shape[0]
        rows["raw_stats"].append(row["raw_stats"].numpy())
        rows["input_stats"].append(raw_stats_features(pts))
        rows["raw_count"].append([float(row["raw_count"].item())])
        rows["fixed_count"].append([float(pts.shape[0])])
        rows["unique_voxel_count"].append([float(len(voxel_keys(pts)))])
        rows["duplicate_rate"].append([float(1.0 - unique_exact / max(pts.shape[0], 1))])
        rows["levels"].append(int(row["level"].item()))
    return {k: np.asarray(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train = collect(CoverageStateDataset(args.cache_root, split="train", seed=args.seed))
    test = collect(CoverageStateDataset(args.cache_root, split="test", seed=args.seed))
    y_train = train["levels"].astype(np.int64)
    y_test = test["levels"].astype(np.int64)
    all_y = np.concatenate([y_train, y_test])
    out = {
        "train_n": int(len(y_train)),
        "test_n": int(len(y_test)),
        "chance": float(1.0 / max(len(np.unique(all_y)), 1)),
        "fixed_point_count_unique": sorted({int(x[0]) for x in np.concatenate([train["fixed_count"], test["fixed_count"]])}),
        "duplicate_rate_mean": float(np.concatenate([train["duplicate_rate"], test["duplicate_rate"]]).mean()),
        "unique_voxel_count_mean": float(np.concatenate([train["unique_voxel_count"], test["unique_voxel_count"]]).mean()),
        "acc_raw_union_count": train_linear_eval(train["raw_count"], y_train, test["raw_count"], y_test, seed=args.seed),
        "acc_fixed_input_count": train_linear_eval(train["fixed_count"], y_train, test["fixed_count"], y_test, seed=args.seed),
        "acc_input_unique_voxel_count": train_linear_eval(train["unique_voxel_count"], y_train, test["unique_voxel_count"], y_test, seed=args.seed),
        "acc_input_duplicate_rate": train_linear_eval(train["duplicate_rate"], y_train, test["duplicate_rate"], y_test, seed=args.seed),
        "acc_input_stats": train_linear_eval(train["input_stats"], y_train, test["input_stats"], y_test, seed=args.seed),
        "acc_raw_union_stats": train_linear_eval(train["raw_stats"], y_train, test["raw_stats"], y_test, seed=args.seed),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    lines = ["# Point-count / raw-stat shortcut", "", "| metric | value |", "|---|---:|"]
    for k, v in out.items():
        lines.append(f"| {k} | {v} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
