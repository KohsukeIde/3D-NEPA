#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from coverage_nepa.data.coverage_dataset import CoverageStateDataset
from coverage_nepa.models.simple_point_encoder import SimplePointEncoder


def linear_acc(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor, seed: int, epochs: int = 200) -> float:
    torch.manual_seed(seed)
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
        return float((head(test_x).argmax(1) == test_y).float().mean()) if len(test_y) else 0.0


def encode_dataset(ds: CoverageStateDataset, encoder: str, seed: int, batch_size: int):
    level_labels = []
    cat_names = []
    feats = []
    if encoder == "simple_random":
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc = SimplePointEncoder(dim=256).to(device).eval()
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        with torch.no_grad():
            for b in loader:
                z = enc(b["points"].to(device)).cpu()
                feats.append(z)
                level_labels.extend(b["level"].tolist())
                cat_names.extend(list(b["category"]))
        feat = torch.cat(feats, 0)
    else:
        for i in range(len(ds)):
            row = ds[i]
            feats.append(row["raw_stats"].float().unsqueeze(0))
            level_labels.append(int(row["level"]))
            cat_names.append(row["category"])
        feat = torch.cat(feats, 0)
    return feat, level_labels, cat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--encoder", default="simple_random", choices=["simple_random", "raw_stats"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    train_ds = CoverageStateDataset(args.cache_root, split="train", seed=args.seed)
    test_ds = CoverageStateDataset(args.cache_root, split="test", seed=args.seed)
    train_feat, train_levels, train_cats = encode_dataset(train_ds, args.encoder, args.seed, args.batch_size)
    test_feat, test_levels, test_cats = encode_dataset(test_ds, args.encoder, args.seed, args.batch_size)
    train_level_y = torch.tensor(train_levels, dtype=torch.long)
    test_level_y = torch.tensor(test_levels, dtype=torch.long)
    cats = {c: i for i, c in enumerate(sorted(set(train_cats + test_cats)))}
    train_cat_y = torch.tensor([cats[c] for c in train_cats], dtype=torch.long)
    test_cat_y = torch.tensor([cats[c] for c in test_cats], dtype=torch.long)
    out = {
        "encoder": args.encoder,
        "train_n": int(len(train_ds)),
        "test_n": int(len(test_ds)),
        "feature_dim": int(train_feat.shape[1]),
        "level_acc": linear_acc(train_feat, train_level_y, test_feat, test_level_y, args.seed),
        "category_acc_all_levels": linear_acc(train_feat, train_cat_y, test_feat, test_cat_y, args.seed + 11),
        "num_categories": int(len(cats)),
    }
    # category accuracy by each coverage level
    by_level = {}
    for lv in sorted(set(train_levels) & set(test_levels)):
        tr = train_level_y == lv
        te = test_level_y == lv
        if int(tr.sum()) >= max(10, len(cats)) and int(te.sum()) > 0:
            by_level[str(lv)] = linear_acc(train_feat[tr], train_cat_y[tr], test_feat[te], test_cat_y[te], args.seed + 100 + lv)
    out["category_acc_by_level"] = by_level
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    lines = ["# Frozen coverage-axis analysis", "", f"encoder: `{args.encoder}`", "", "| metric | value |", "|---|---:|"]
    for k, v in out.items():
        lines.append(f"| {k} | {v} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
