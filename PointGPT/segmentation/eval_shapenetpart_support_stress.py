#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import PartNormalDataset
from main import get_model_loss, seg_classes, seg_label_to_cat, to_categorical


def parse_args():
    p = argparse.ArgumentParser("ShapeNetPart support-stress evaluation")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--model", default="pt")
    p.add_argument("--model_name", default="PointGPT_S", choices=["PointGPT_S", "PointGPT_B", "PointGPT_L"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--npoint", type=int, default=2048)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--normal", action="store_true", default=False)
    p.add_argument("--group_mode", default="fps_knn",
                   choices=["fps_knn", "random_center_knn", "voxel_center_knn", "radius_fps", "random_group"])
    p.add_argument("--group_radius", type=float, default=0.22)
    p.add_argument("--group_voxel_grid", type=int, default=6)
    p.add_argument("--local_noise_sigma", type=float, default=0.08,
                   help="Gaussian noise sigma as a fraction of object bbox diagonal for local_jitter.")
    p.add_argument("--output_json", default="")
    p.add_argument("--output_md", default="")
    return p.parse_args()


def _resample(points, target, keep_idx, npoint, rng):
    kept_points = points[keep_idx]
    kept_target = target[keep_idx]
    if kept_points.shape[0] >= npoint:
        idx = rng.choice(kept_points.shape[0], npoint, replace=False)
    else:
        idx = rng.choice(kept_points.shape[0], npoint, replace=True)
    return kept_points[idx], kept_target[idx]


def _resample_indices(keep_idx, npoint, rng):
    if keep_idx.shape[0] >= npoint:
        local_idx = rng.choice(keep_idx.shape[0], npoint, replace=False)
    else:
        local_idx = rng.choice(keep_idx.shape[0], npoint, replace=True)
    return keep_idx[local_idx]


def _ratio_suffix(condition, prefix, suffix=""):
    if not condition.startswith(prefix):
        return None
    tail = condition[len(prefix):]
    if suffix:
        if not tail.endswith(suffix):
            return None
        tail = tail[: -len(suffix)]
    return int(tail) / 100.0


def label_entropy(counts):
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def support_indices_one(points, target, condition, rng, local_noise_sigma=0.08):
    npoint = points.shape[0]
    if condition == "clean":
        keep_idx = np.arange(npoint, dtype=np.int64)
        return keep_idx, keep_idx.copy(), -1
    if condition == "xyz_zero":
        keep_idx = np.arange(npoint, dtype=np.int64)
        return keep_idx, keep_idx.copy(), -1

    random_ratio = _ratio_suffix(condition, "random_keep")
    if random_ratio is not None:
        keep_n = max(1, int(round(random_ratio * npoint)))
        keep_idx = rng.choice(npoint, keep_n, replace=False)
        keep_idx = np.unique(keep_idx).astype(np.int64)
        return keep_idx, _resample_indices(keep_idx, npoint, rng), -1

    structured_ratio = _ratio_suffix(condition, "structured_keep")
    if structured_ratio is not None:
        keep_n = max(1, int(round(structured_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((points[:, :3] - points[anchor : anchor + 1, :3]) ** 2, axis=1)
        keep_idx = np.argsort(dist)[-keep_n:]
        keep_idx = np.unique(keep_idx).astype(np.int64)
        return keep_idx, _resample_indices(keep_idx, npoint, rng), -1

    local_jitter_ratio = _ratio_suffix(condition, "local_jitter")
    if local_jitter_ratio is not None:
        keep_idx = np.arange(npoint, dtype=np.int64)
        return keep_idx, keep_idx.copy(), -1

    local_replace_ratio = _ratio_suffix(condition, "local_replace")
    if local_replace_ratio is not None:
        keep_idx = np.arange(npoint, dtype=np.int64)
        return keep_idx, keep_idx.copy(), -1

    if condition == "part_drop_largest":
        labels, counts = np.unique(target, return_counts=True)
        removed_part = -1
        if len(labels) <= 1:
            keep_idx = rng.choice(npoint, max(1, int(round(0.2 * npoint))), replace=False)
        else:
            drop_label = labels[np.argmax(counts)]
            removed_part = int(drop_label)
            keep_idx = np.flatnonzero(target != drop_label)
            if keep_idx.size == 0:
                keep_idx = rng.choice(npoint, max(1, int(round(0.2 * npoint))), replace=False)
                removed_part = -1
        keep_idx = np.unique(keep_idx).astype(np.int64)
        return keep_idx, _resample_indices(keep_idx, npoint, rng), removed_part

    part_keep_ratio = _ratio_suffix(condition, "part_keep", "_per_part")
    if part_keep_ratio is not None:
        keep_parts = []
        for label in np.unique(target):
            idx = np.flatnonzero(target == label)
            k = max(1, int(round(part_keep_ratio * idx.size)))
            keep_parts.append(rng.choice(idx, k, replace=False))
        keep_idx = np.concatenate(keep_parts, axis=0)
        keep_idx = np.unique(keep_idx).astype(np.int64)
        return keep_idx, _resample_indices(keep_idx, npoint, rng), -1
    raise ValueError(condition)


def stress_one(points, target, condition, rng, local_noise_sigma=0.08):
    npoint = points.shape[0]
    keep_idx, forward_idx, _ = support_indices_one(points, target, condition, rng, local_noise_sigma)
    out = points[forward_idx].copy()
    if condition == "xyz_zero":
        out[:, :3] = 0.0
    local_jitter_ratio = _ratio_suffix(condition, "local_jitter")
    if local_jitter_ratio is not None:
        patch_n = max(1, int(round(local_jitter_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((out[:, :3] - out[anchor : anchor + 1, :3]) ** 2, axis=1)
        patch_idx = np.argsort(dist)[:patch_n]
        diag = float(np.linalg.norm(out[:, :3].max(axis=0) - out[:, :3].min(axis=0)))
        sigma = max(1e-6, float(local_noise_sigma) * diag)
        out[patch_idx, :3] = out[patch_idx, :3] + rng.normal(0.0, sigma, size=(patch_idx.size, 3)).astype(out.dtype)
    local_replace_ratio = _ratio_suffix(condition, "local_replace")
    if local_replace_ratio is not None:
        patch_n = max(1, int(round(local_replace_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((out[:, :3] - out[anchor : anchor + 1, :3]) ** 2, axis=1)
        patch_idx = np.argsort(dist)[:patch_n]
        mins = out[:, :3].min(axis=0)
        maxs = out[:, :3].max(axis=0)
        out[patch_idx, :3] = mins + rng.rand(patch_idx.size, 3).astype(out.dtype) * (maxs - mins)
    return out, target[forward_idx]


def build_forward_points(points, forward_idx, condition, rng, local_noise_sigma=0.08):
    npoint = points.shape[0]
    out = points[forward_idx].copy()
    if condition == "xyz_zero":
        out[:, :3] = 0.0
    local_jitter_ratio = _ratio_suffix(condition, "local_jitter")
    if local_jitter_ratio is not None:
        patch_n = max(1, int(round(local_jitter_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((out[:, :3] - out[anchor : anchor + 1, :3]) ** 2, axis=1)
        patch_idx = np.argsort(dist)[:patch_n]
        diag = float(np.linalg.norm(out[:, :3].max(axis=0) - out[:, :3].min(axis=0)))
        sigma = max(1e-6, float(local_noise_sigma) * diag)
        out[patch_idx, :3] = out[patch_idx, :3] + rng.normal(0.0, sigma, size=(patch_idx.size, 3)).astype(out.dtype)
    local_replace_ratio = _ratio_suffix(condition, "local_replace")
    if local_replace_ratio is not None:
        patch_n = max(1, int(round(local_replace_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((out[:, :3] - out[anchor : anchor + 1, :3]) ** 2, axis=1)
        patch_idx = np.argsort(dist)[:patch_n]
        mins = out[:, :3].min(axis=0)
        maxs = out[:, :3].max(axis=0)
        out[patch_idx, :3] = mins + rng.rand(patch_idx.size, 3).astype(out.dtype) * (maxs - mins)
    return out


def average_logits_by_original_index(logits_forward, forward_idx, keep_idx):
    local_map = {int(orig): pos for pos, orig in enumerate(keep_idx.tolist())}
    local_idx = np.asarray([local_map[int(orig)] for orig in forward_idx.tolist()], dtype=np.int64)
    logits_unique = np.zeros((keep_idx.shape[0], logits_forward.shape[1]), dtype=np.float64)
    counts = np.zeros((keep_idx.shape[0], 1), dtype=np.float64)
    np.add.at(logits_unique, local_idx, logits_forward.astype(np.float64))
    np.add.at(counts, local_idx, 1.0)
    return logits_unique / np.maximum(counts, 1.0)


def category_restricted_prediction(logits, target):
    cat = seg_label_to_cat[int(target[0])]
    part_ids = np.asarray(seg_classes[cat], dtype=np.int64)
    pred_local = np.argmax(logits[:, part_ids], axis=1)
    return part_ids[pred_local].astype(np.int32)


def new_accumulator():
    return {
        "total_correct": 0,
        "total_seen": 0,
        "total_seen_class": np.zeros(50, dtype=np.int64),
        "total_correct_class": np.zeros(50, dtype=np.int64),
        "shape_ious": {cat: [] for cat in seg_classes.keys()},
    }


def update_accumulator(acc, pred, target):
    acc["total_correct"] += int(np.sum(pred == target))
    acc["total_seen"] += int(target.size)
    for part_id in range(50):
        acc["total_seen_class"][part_id] += int(np.sum(target == part_id))
        acc["total_correct_class"][part_id] += int(np.sum((pred == part_id) & (target == part_id)))
    cat = seg_label_to_cat[int(target[0])]
    part_ious = []
    for part_id in seg_classes[cat]:
        denom = np.sum((target == part_id) | (pred == part_id))
        if denom == 0:
            part_ious.append(1.0)
        else:
            part_ious.append(float(np.sum((target == part_id) & (pred == part_id)) / denom))
    acc["shape_ious"][cat].append(float(np.mean(part_ious)))


def finalize_accumulator(acc):
    per_cat_iou = {
        cat: float(np.mean(vals)) if vals else float("nan")
        for cat, vals in acc["shape_ious"].items()
    }
    all_shape_ious = [x for vals in acc["shape_ious"].values() for x in vals]
    class_acc = acc["total_correct_class"] / np.maximum(1, acc["total_seen_class"].astype(float))
    return {
        "accuracy": float(acc["total_correct"] / max(1, acc["total_seen"])),
        "class_avg_accuracy": float(np.nanmean(class_acc)),
        "class_avg_iou": float(np.nanmean(list(per_cat_iou.values()))),
        "instance_avg_iou": float(np.mean(all_shape_ious)) if all_shape_ious else float("nan"),
        "per_category_iou": per_cat_iou,
    }


def build_model(args):
    import importlib

    module = importlib.import_module(args.model)
    classifier, _ = get_model_loss(module, args, num_part=50)
    try:
        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.ckpt, map_location="cpu")
    state_dict = state.get("model_state_dict", state)
    classifier.load_state_dict(state_dict, strict=False)
    classifier.eval()
    return classifier.cuda()


def evaluate_condition(model, loader, condition, args):
    condition_offsets = {
        "clean": 0,
        "random_keep80": 1,
        "random_keep50": 2,
        "random_keep20": 3,
        "random_keep10": 4,
        "structured_keep80": 5,
        "structured_keep50": 6,
        "structured_keep20": 7,
        "structured_keep10": 8,
        "local_jitter80": 9,
        "local_jitter50": 10,
        "local_jitter20": 11,
        "local_jitter10": 12,
        "local_replace80": 13,
        "local_replace50": 14,
        "local_replace20": 15,
        "local_replace10": 16,
        "part_drop_largest": 17,
        "part_keep80_per_part": 18,
        "part_keep50_per_part": 19,
        "part_keep20_per_part": 20,
        "part_keep10_per_part": 21,
        "xyz_zero": 22,
    }
    rng = np.random.RandomState(args.seed + condition_offsets[condition])
    eval_acc = new_accumulator()
    clean_subset_acc = new_accumulator()
    before_hist = np.zeros(50, dtype=np.int64)
    retained_hist = np.zeros(50, dtype=np.int64)
    removed_hist = np.zeros(50, dtype=np.int64)
    retained_counts = []
    forward_counts = []
    repeated_forward_counts = []

    with torch.no_grad():
        for points, label, target in tqdm(loader, desc=condition):
            pts_np = points.numpy()
            tgt_np = target.numpy()
            stressed_pts = []
            keep_indices = []
            forward_indices = []
            removed_parts = []
            for i in range(pts_np.shape[0]):
                keep_idx, forward_idx, removed_part = support_indices_one(
                    pts_np[i],
                    tgt_np[i],
                    condition,
                    rng,
                    args.local_noise_sigma,
                )
                pts_i = build_forward_points(pts_np[i], forward_idx, condition, rng, args.local_noise_sigma)
                stressed_pts.append(pts_i)
                keep_indices.append(keep_idx)
                forward_indices.append(forward_idx)
                removed_parts.append(removed_part)
            points_t = torch.tensor(np.stack(stressed_pts), dtype=torch.float32).cuda().transpose(2, 1)
            label_t = label.long().cuda()

            seg_pred = model(points_t, to_categorical(label_t, 16))
            logits_np = seg_pred.cpu().numpy()
            if logits_np.ndim == 3 and logits_np.shape[1] == 50:
                logits_np = np.transpose(logits_np, (0, 2, 1))

            if condition == "clean":
                clean_logits_np = logits_np
            else:
                clean_pred = model(points.float().cuda().transpose(2, 1), to_categorical(label_t, 16))
                clean_logits_np = clean_pred.cpu().numpy()
                if clean_logits_np.ndim == 3 and clean_logits_np.shape[1] == 50:
                    clean_logits_np = np.transpose(clean_logits_np, (0, 2, 1))

            for i, keep_idx in enumerate(keep_indices):
                forward_idx = forward_indices[i]
                target_unique = tgt_np[i][keep_idx].astype(np.int64)
                logits_unique = average_logits_by_original_index(logits_np[i], forward_idx, keep_idx)
                pred_unique = category_restricted_prediction(logits_unique, target_unique)
                clean_pred_unique = category_restricted_prediction(clean_logits_np[i][keep_idx], target_unique)
                update_accumulator(eval_acc, pred_unique, target_unique)
                update_accumulator(clean_subset_acc, clean_pred_unique, target_unique)

                before_hist += np.bincount(tgt_np[i].reshape(-1), minlength=50).astype(np.int64)
                retained_hist += np.bincount(target_unique.reshape(-1), minlength=50).astype(np.int64)
                if removed_parts[i] >= 0:
                    removed_hist[removed_parts[i]] += 1
                retained_counts.append(int(keep_idx.size))
                forward_counts.append(int(forward_idx.size))
                repeated_forward_counts.append(int(forward_idx.size - np.unique(forward_idx).size))

    result = finalize_accumulator(eval_acc)
    clean_subset = finalize_accumulator(clean_subset_acc)
    retained_counts_np = np.asarray(retained_counts, dtype=np.float64)
    forward_counts_np = np.asarray(forward_counts, dtype=np.float64)
    repeated_counts_np = np.asarray(repeated_forward_counts, dtype=np.float64)
    result.update({
        "condition": condition,
        "clean_subset_accuracy": clean_subset["accuracy"],
        "clean_subset_class_avg_iou": clean_subset["class_avg_iou"],
        "clean_subset_instance_avg_iou": clean_subset["instance_avg_iou"],
        "damage_class_avg_iou": clean_subset["class_avg_iou"] - result["class_avg_iou"],
        "damage_instance_avg_iou": clean_subset["instance_avg_iou"] - result["instance_avg_iou"],
        "metric_scope": "unique_retained_original_points",
        "logit_aggregation": "mean_by_original_index",
        "mean_retained_unique_points": float(retained_counts_np.mean()),
        "min_retained_unique_points": int(retained_counts_np.min()),
        "max_retained_unique_points": int(retained_counts_np.max()),
        "mean_forward_points": float(forward_counts_np.mean()),
        "mean_repeated_forward_points": float(repeated_counts_np.mean()),
        "mean_forward_duplication_factor": float(np.mean(forward_counts_np / np.maximum(retained_counts_np, 1.0))),
        "part_histogram_before": before_hist.tolist(),
        "part_histogram_retained": retained_hist.tolist(),
        "removed_part_histogram": removed_hist.tolist(),
        "part_entropy_before": label_entropy(before_hist),
        "part_entropy_retained": label_entropy(retained_hist),
    })
    return result


def to_md(summary):
    lines = [
        "# ShapeNetPart Support Stress",
        "",
        f"- ckpt: `{summary['ckpt']}`",
        f"- root: `{summary['root']}`",
        "",
        "| condition | accuracy | class avg IoU | instance avg IoU | clean subset inst IoU | damage inst IoU | retained unique pts | repeated forward pts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| `{row['condition']}` | `{row['accuracy']:.4f}` | `{row['class_avg_iou']:.4f}` | "
            f"`{row['instance_avg_iou']:.4f}` | `{row['clean_subset_instance_avg_iou']:.4f}` | "
            f"`{row['damage_instance_avg_iou']:.4f}` | `{row['mean_retained_unique_points']:.1f}` | "
            f"`{row['mean_repeated_forward_points']:.1f}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Metrics are computed on unique retained original point indices.")
    lines.append("- When retained support is resampled for the fixed-size forward pass, repeated logits are averaged back to the original retained point before scoring.")
    lines.append("- `clean subset inst IoU` evaluates clean full-input predictions on the same retained point set; `damage inst IoU` is the matched retained-subset delta.")
    lines.append("- `part_drop_largest` removes the largest ground-truth part within each object before fixed-size forward resampling.")
    lines.append("- `part_keepXX_per_part` keeps XX% of each ground-truth part before fixed-size forward resampling. These are support-stress probes, not official ShapeNetPart scores.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = PartNormalDataset(root=args.root, npoints=args.npoint, split="test", normal_channel=args.normal)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args)
    conditions = [
        "clean",
        "random_keep80",
        "random_keep50",
        "random_keep20",
        "random_keep10",
        "structured_keep80",
        "structured_keep50",
        "structured_keep20",
        "structured_keep10",
        "local_jitter80",
        "local_jitter50",
        "local_jitter20",
        "local_jitter10",
        "local_replace80",
        "local_replace50",
        "local_replace20",
        "local_replace10",
        "part_drop_largest",
        "part_keep80_per_part",
        "part_keep50_per_part",
        "part_keep20_per_part",
        "part_keep10_per_part",
        "xyz_zero",
    ]
    rows = [evaluate_condition(model, loader, cond, args) for cond in conditions]
    summary = {
        "ckpt": args.ckpt,
        "root": args.root,
        "conditions": rows,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(to_md(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
