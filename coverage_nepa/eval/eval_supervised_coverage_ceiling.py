#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coverage_nepa.data.coverage_dataset import CoverageStateDataset
from coverage_nepa.eval.supervised_common import (
    PointLabelDataset,
    collect_feature_matrix,
    collect_labels,
    infer_num_levels,
    materialize_point_dataset,
    per_class_counts,
    resolve_device,
    train_feature_mlp,
    train_point_classifier,
)


FEATURE_MODELS = {
    "raw_union_stats_mlp": "raw_union_stats",
    "input_stats_mlp": "input_stats",
    "raw_union_count_mlp": "raw_union_count",
    "fixed_input_count_mlp": "fixed_input_count",
    "input_unique_voxel_count_mlp": "input_unique_voxel_count",
    "input_duplicate_rate_mlp": "input_duplicate_rate",
}
POINT_MODELS = {"pointnet_small", "simple_encoder"}


def run_model(
    args,
    train_ds: CoverageStateDataset,
    test_ds: CoverageStateDataset,
    model_name: str,
    num_levels: int,
    seed: int,
    feature_cache: dict[str, tuple],
    point_cache: tuple,
):
    device = resolve_device(args.device)
    if model_name in FEATURE_MODELS:
        feature_kind = FEATURE_MODELS[model_name]
        train_x, test_x, train_y, test_y = feature_cache[feature_kind]
        result = train_feature_mlp(
            train_x,
            train_y,
            test_x,
            test_y,
            num_classes=num_levels,
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif model_name in POINT_MODELS:
        train_points, test_points = point_cache
        result = train_point_classifier(
            train_points,
            test_points,
            model_name=model_name,
            num_classes=num_levels,
            seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
        )
    else:
        raise ValueError(f"unknown model: {model_name}")
    return {
        "train_acc": result.train_acc,
        "test_acc": result.test_acc,
        "majority_test_acc": result.majority_test_acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--models", default="raw_union_stats_mlp,input_stats_mlp,input_unique_voxel_count_mlp,input_duplicate_rate_mlp,pointnet_small,simple_encoder")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    train_ds = CoverageStateDataset(args.cache_root, split="train", seed=args.seed)
    val_ds = CoverageStateDataset(args.cache_root, split="val", seed=args.seed)
    test_ds = CoverageStateDataset(args.cache_root, split="test", seed=args.seed)
    num_levels = max(infer_num_levels(train_ds), infer_num_levels(val_ds), infer_num_levels(test_ds))
    train_y = collect_labels(train_ds, None, "level")
    val_y = collect_labels(val_ds, None, "level")
    test_y = collect_labels(test_ds, None, "level")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    feature_cache = {}
    for feature_kind in sorted({FEATURE_MODELS[m] for m in models if m in FEATURE_MODELS}):
        print(f"[coverage-ceiling] collecting {feature_kind}", flush=True)
        feature_cache[feature_kind] = (
            collect_feature_matrix(train_ds, None, feature_kind),
            collect_feature_matrix(test_ds, None, feature_kind),
            train_y,
            test_y,
        )
    point_cache = (
        materialize_point_dataset(PointLabelDataset(train_ds, None, task="level")),
        materialize_point_dataset(PointLabelDataset(test_ds, None, task="level")),
    )

    out = {
        "cache_root": args.cache_root,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "test_n": len(test_ds),
        "num_levels": num_levels,
        "chance": 1.0 / max(num_levels, 1),
        "train_level_counts": per_class_counts(train_y),
        "val_level_counts": per_class_counts(val_y),
        "test_level_counts": per_class_counts(test_y),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seeds": seeds,
        "models": {},
    }
    for i, model_name in enumerate(models):
        print(f"[coverage-ceiling] training {model_name}", flush=True)
        seed_rows = []
        for seed in seeds:
            seed_rows.append(
                run_model(
                    args,
                    train_ds,
                    test_ds,
                    model_name,
                    num_levels,
                    seed=seed + 1009 * i,
                    feature_cache=feature_cache,
                    point_cache=point_cache,
                )
            )
        test_accs = [r["test_acc"] for r in seed_rows]
        train_accs = [r["train_acc"] for r in seed_rows]
        out["models"][model_name] = {
            "seed_results": seed_rows,
            "mean_train_acc": float(sum(train_accs) / max(len(train_accs), 1)),
            "mean_test_acc": float(sum(test_accs) / max(len(test_accs), 1)),
            "worst_test_acc": float(min(test_accs)) if test_accs else 0.0,
            "best_test_acc": float(max(test_accs)) if test_accs else 0.0,
            "majority_test_acc": seed_rows[0]["majority_test_acc"] if seed_rows else 0.0,
        }

    input_shortcut_names = [
        "input_stats_mlp",
        "input_unique_voxel_count_mlp",
        "input_duplicate_rate_mlp",
        "fixed_input_count_mlp",
    ]
    best_input_shortcut = max(
        [out["models"][m]["mean_test_acc"] for m in input_shortcut_names if m in out["models"]],
        default=0.0,
    )
    out["best_input_shortcut_mean_test_acc"] = best_input_shortcut
    for model_name in POINT_MODELS:
        if model_name in out["models"]:
            out["models"][model_name]["margin_over_best_input_shortcut"] = (
                out["models"][model_name]["mean_test_acc"] - best_input_shortcut
            )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))

    lines = [
        "# Supervised Coverage-Level Ceiling",
        "",
        f"cache: `{args.cache_root}`",
        "",
        f"chance: `{out['chance']:.4f}`",
        f"best input-side shortcut mean test acc: `{best_input_shortcut:.4f}`",
        "",
        "| model | mean_train_acc | mean_test_acc | worst_test_acc | majority_test_acc | margin_over_input_shortcut |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in out["models"].items():
        margin = row.get("margin_over_best_input_shortcut", "")
        margin_txt = "" if margin == "" else f"{margin:.4f}"
        lines.append(
            f"| {name} | {row['mean_train_acc']:.4f} | {row['mean_test_acc']:.4f} | "
            f"{row['worst_test_acc']:.4f} | {row['majority_test_acc']:.4f} | {margin_txt} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
