#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from coverage_nepa.data.coverage_dataset import CoverageStateDataset
from coverage_nepa.eval.supervised_common import (
    PointLabelDataset,
    PointNetSmall,
    SimpleEncoderClassifier,
    collect_feature_matrix,
    collect_labels,
    infer_num_levels,
    materialize_point_dataset,
    resolve_device,
    set_seed,
)


FEATURE_MODELS = {
    "raw_union_stats_mlp": "raw_union_stats",
    "input_stats_mlp": "input_stats",
}
POINT_MODELS = {"pointnet_small", "simple_encoder"}


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_feature_probe(args, train_x, train_y, test_x, num_classes, seed):
    device = resolve_device(args.device)
    set_seed(seed)
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std
    model = FeatureMLP(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    for _ in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(test_x), args.batch_size):
            logits.append(model(test_x[start : start + args.batch_size].to(device)).cpu())
    return torch.cat(logits, dim=0) if logits else torch.empty(0, num_classes)


def train_point_probe(args, train_ds, test_ds, model_name, num_classes, seed):
    device = resolve_device(args.device)
    set_seed(seed)
    if model_name == "pointnet_small":
        model = PointNetSmall(num_classes=num_classes)
    elif model_name == "simple_encoder":
        model = SimpleEncoderClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"unknown point model: {model_name}")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = materialize_point_dataset(train_ds)
    test_ds = materialize_point_dataset(test_ds)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    for _ in range(args.epochs):
        model.train()
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(points), labels)
            loss.backward()
            opt.step()
    model.eval()
    logits = []
    with torch.no_grad():
        eval_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        for points, _labels in eval_loader:
            logits.append(model(points.to(device)).cpu())
    return torch.cat(logits, dim=0) if logits else torch.empty(0, num_classes)


def train_category_map(train_ds: CoverageStateDataset) -> dict[str, int]:
    cats: set[str] = set()
    for i in range(len(train_ds)):
        cats.add(str(train_ds[i]["category"]))
    return {c: i for i, c in enumerate(sorted(cats))}


def seen_indices(ds: CoverageStateDataset, category_map: dict[str, int]) -> tuple[list[int], int]:
    idxs = []
    excluded = 0
    for i in range(len(ds)):
        if str(ds[i]["category"]) in category_map:
            idxs.append(i)
        else:
            excluded += 1
    return idxs, excluded


def metadata_for_indices(ds: CoverageStateDataset, indices: list[int], category_map: dict[str, int]) -> list[dict]:
    rows = []
    for i in indices:
        row = ds[i]
        cat = str(row["category"])
        rows.append(
            {
                "shape_id": str(row["shape_id"]),
                "level": int(row["level"].item()),
                "category": cat,
                "label": category_map[cat],
            }
        )
    return rows


def rankdata_average(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(levels: list[int], values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xr = rankdata_average([float(x) for x in levels])
    yr = rankdata_average(values)
    if float(np.std(xr)) == 0.0 or float(np.std(yr)) == 0.0:
        return 0.0
    return float(np.corrcoef(xr, yr)[0, 1])


def summarize_predictions(logits: torch.Tensor, labels: torch.Tensor, meta: list[dict], num_levels: int) -> dict:
    pred = logits.argmax(dim=1)
    logp = torch.log_softmax(logits, dim=1)
    true_scores = logp[torch.arange(labels.numel()), labels].numpy()
    correct = (pred == labels).numpy()

    acc_by_level: list[float | None] = []
    score_by_level: list[float | None] = []
    n_by_level: list[int] = []
    valid_levels: list[int] = []
    valid_accs: list[float] = []
    for level in range(num_levels):
        idx = [i for i, m in enumerate(meta) if m["level"] == level]
        n_by_level.append(len(idx))
        if not idx:
            acc_by_level.append(None)
            score_by_level.append(None)
            continue
        acc = float(np.mean(correct[idx]))
        score = float(np.mean(true_scores[idx]))
        acc_by_level.append(acc)
        score_by_level.append(score)
        valid_levels.append(level)
        valid_accs.append(acc)

    shape_levels: dict[str, dict[int, float]] = {}
    for i, m in enumerate(meta):
        shape_levels.setdefault(m["shape_id"], {})[m["level"]] = float(true_scores[i])
    first_level = min(valid_levels) if valid_levels else 0
    last_level = max(valid_levels) if valid_levels else 0
    deltas = []
    for level_scores in shape_levels.values():
        if first_level in level_scores and last_level in level_scores:
            deltas.append(level_scores[last_level] - level_scores[first_level])
    sign_count = int(sum(1 for d in deltas if d > 0.0))
    paired_n = len(deltas)

    adjacent_drops = []
    for a, b in zip(acc_by_level[:-1], acc_by_level[1:]):
        if a is not None and b is not None:
            adjacent_drops.append(float(b - a))
    max_adjacent_drop = float(min(adjacent_drops)) if adjacent_drops else 0.0

    return {
        "test_acc_all_levels": float(np.mean(correct)) if len(correct) else 0.0,
        "acc_by_level": acc_by_level,
        "true_logp_by_level": score_by_level,
        "n_by_level": n_by_level,
        "acc_delta_last_first": float(valid_accs[-1] - valid_accs[0]) if len(valid_accs) >= 2 else 0.0,
        "true_logp_delta_mean": float(np.mean(deltas)) if deltas else 0.0,
        "true_logp_delta_median": float(np.median(deltas)) if deltas else 0.0,
        "paired_positive_count": sign_count,
        "paired_shape_count": paired_n,
        "paired_positive_rate": float(sign_count / paired_n) if paired_n else 0.0,
        "spearman_acc_level": spearman(valid_levels, valid_accs),
        "max_adjacent_acc_drop": max_adjacent_drop,
    }


def run_model(args, train_ds, test_ds, test_seen_idx, category_map, model_name, seed, feature_cache, point_cache):
    num_classes = len(category_map)
    if model_name in FEATURE_MODELS:
        train_x, train_y, test_x = feature_cache[FEATURE_MODELS[model_name]]
        logits = train_feature_probe(args, train_x, train_y, test_x, num_classes, seed)
    elif model_name in POINT_MODELS:
        train_points, test_points = point_cache
        logits = train_point_probe(args, train_points, test_points, model_name, num_classes, seed)
    else:
        raise ValueError(f"unknown model: {model_name}")
    labels = collect_labels(test_ds, test_seen_idx, "category", category_map)
    meta = metadata_for_indices(test_ds, test_seen_idx, category_map)
    return summarize_predictions(logits, labels, meta, num_levels=max(infer_num_levels(train_ds), infer_num_levels(test_ds)))


def aggregate_seed_results(seed_rows: list[dict]) -> dict:
    keys = [
        "test_acc_all_levels",
        "acc_delta_last_first",
        "true_logp_delta_mean",
        "paired_positive_rate",
        "spearman_acc_level",
        "max_adjacent_acc_drop",
    ]
    out = {"seed_results": seed_rows}
    for key in keys:
        vals = [float(r[key]) for r in seed_rows]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else 0.0
        out[f"worst_{key}"] = float(np.min(vals)) if vals else 0.0
        out[f"best_{key}"] = float(np.max(vals)) if vals else 0.0
    if seed_rows:
        accs = np.asarray([r["acc_by_level"] for r in seed_rows], dtype=np.float64)
        out["mean_acc_by_level"] = np.nanmean(accs, axis=0).tolist()
        logps = np.asarray([r["true_logp_by_level"] for r in seed_rows], dtype=np.float64)
        out["mean_true_logp_by_level"] = np.nanmean(logps, axis=0).tolist()
        out["paired_shape_count"] = seed_rows[0]["paired_shape_count"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--models", default="raw_union_stats_mlp,input_stats_mlp,pointnet_small,simple_encoder")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    train_ds = CoverageStateDataset(args.cache_root, split="train", seed=args.seed)
    test_ds = CoverageStateDataset(args.cache_root, split="test", seed=args.seed)
    category_map = train_category_map(train_ds)
    test_seen_idx, excluded_rows = seen_indices(test_ds, category_map)
    excluded_shapes = {
        str(test_ds[i]["shape_id"])
        for i in range(len(test_ds))
        if str(test_ds[i]["category"]) not in category_map
    }
    seen_shapes = {str(test_ds[i]["shape_id"]) for i in test_seen_idx}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    train_idx = list(range(len(train_ds)))
    train_y = collect_labels(train_ds, train_idx, "category", category_map)
    feature_cache = {}
    for feature_kind in sorted({FEATURE_MODELS[m] for m in models if m in FEATURE_MODELS}):
        print(f"[category-monotonicity] collecting {feature_kind}", flush=True)
        feature_cache[feature_kind] = (
            collect_feature_matrix(train_ds, train_idx, feature_kind),
            train_y,
            collect_feature_matrix(test_ds, test_seen_idx, feature_kind),
        )
    point_cache = (
        materialize_point_dataset(PointLabelDataset(train_ds, None, task="category", category_map=category_map)),
        materialize_point_dataset(PointLabelDataset(test_ds, test_seen_idx, task="category", category_map=category_map)),
    )

    out = {
        "cache_root": args.cache_root,
        "train_n": len(train_ds),
        "test_n": len(test_ds),
        "test_seen_category_n": len(test_seen_idx),
        "excluded_unseen_category_rows": excluded_rows,
        "test_seen_shapes": len(seen_shapes),
        "excluded_unseen_category_shapes": len(excluded_shapes),
        "num_train_categories": len(category_map),
        "num_levels": max(infer_num_levels(train_ds), infer_num_levels(test_ds)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seeds": seeds,
        "models": {},
    }
    for model_i, model_name in enumerate(models):
        print(f"[category-monotonicity] training shared probe {model_name}", flush=True)
        seed_rows = []
        for seed in seeds:
            seed_rows.append(
                run_model(
                    args,
                    train_ds,
                    test_ds,
                    test_seen_idx,
                    category_map,
                    model_name,
                    seed=seed + 1009 * model_i,
                    feature_cache=feature_cache,
                    point_cache=point_cache,
                )
            )
        out["models"][model_name] = aggregate_seed_results(seed_rows)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = [
        "# Category Monotonicity By Coverage Level",
        "",
        f"cache: `{args.cache_root}`",
        "",
        f"test seen shapes: `{out['test_seen_shapes']}`",
        f"excluded unseen-category shapes: `{out['excluded_unseen_category_shapes']}`",
        "",
        "| model | mean_acc_by_level | mean_acc_delta | mean_logp_delta | sign_rate | spearman | max_adjacent_drop |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["models"].items():
        acc_txt = ", ".join(f"{float(v):.4f}" for v in row.get("mean_acc_by_level", []))
        lines.append(
            f"| {name} | {acc_txt} | {row['mean_acc_delta_last_first']:.4f} | "
            f"{row['mean_true_logp_delta_mean']:.4f} | {row['mean_paired_positive_rate']:.4f} | "
            f"{row['mean_spearman_acc_level']:.4f} | {row['mean_max_adjacent_acc_drop']:.4f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
