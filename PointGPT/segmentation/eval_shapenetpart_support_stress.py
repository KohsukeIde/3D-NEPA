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


def _ratio_suffix(condition, prefix, suffix=""):
    if not condition.startswith(prefix):
        return None
    tail = condition[len(prefix):]
    if suffix:
        if not tail.endswith(suffix):
            return None
        tail = tail[: -len(suffix)]
    return int(tail) / 100.0


def stress_one(points, target, condition, rng):
    npoint = points.shape[0]
    if condition == "clean":
        return points, target
    if condition == "xyz_zero":
        out = points.copy()
        out[:, :3] = 0.0
        return out, target

    random_ratio = _ratio_suffix(condition, "random_keep")
    if random_ratio is not None:
        keep_n = max(1, int(round(random_ratio * npoint)))
        keep_idx = rng.choice(npoint, keep_n, replace=False)
        return _resample(points, target, keep_idx, npoint, rng)

    structured_ratio = _ratio_suffix(condition, "structured_keep")
    if structured_ratio is not None:
        keep_n = max(1, int(round(structured_ratio * npoint)))
        anchor = int(rng.randint(0, npoint))
        dist = np.sum((points[:, :3] - points[anchor : anchor + 1, :3]) ** 2, axis=1)
        keep_idx = np.argsort(dist)[-keep_n:]
        return _resample(points, target, keep_idx, npoint, rng)

    if condition == "part_drop_largest":
        labels, counts = np.unique(target, return_counts=True)
        if len(labels) <= 1:
            keep_idx = rng.choice(npoint, max(1, int(round(0.2 * npoint))), replace=False)
        else:
            drop_label = labels[np.argmax(counts)]
            keep_idx = np.flatnonzero(target != drop_label)
            if keep_idx.size == 0:
                keep_idx = rng.choice(npoint, max(1, int(round(0.2 * npoint))), replace=False)
        return _resample(points, target, keep_idx, npoint, rng)

    part_keep_ratio = _ratio_suffix(condition, "part_keep", "_per_part")
    if part_keep_ratio is not None:
        keep_parts = []
        for label in np.unique(target):
            idx = np.flatnonzero(target == label)
            k = max(1, int(round(part_keep_ratio * idx.size)))
            keep_parts.append(rng.choice(idx, k, replace=False))
        keep_idx = np.concatenate(keep_parts, axis=0)
        return _resample(points, target, keep_idx, npoint, rng)
    raise ValueError(condition)


def build_model(args):
    import importlib

    module = importlib.import_module(args.model)
    classifier, _ = get_model_loss(module, args, num_part=50)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
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
        "part_drop_largest": 9,
        "part_keep80_per_part": 10,
        "part_keep50_per_part": 11,
        "part_keep20_per_part": 12,
        "part_keep10_per_part": 13,
        "xyz_zero": 14,
    }
    rng = np.random.RandomState(args.seed + condition_offsets[condition])
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(50)]
    total_correct_class = [0 for _ in range(50)]
    shape_ious = {cat: [] for cat in seg_classes.keys()}

    with torch.no_grad():
        for points, label, target in tqdm(loader, desc=condition):
            pts_np = points.numpy()
            tgt_np = target.numpy()
            stressed_pts = []
            stressed_tgt = []
            for i in range(pts_np.shape[0]):
                pts_i, tgt_i = stress_one(pts_np[i], tgt_np[i], condition, rng)
                stressed_pts.append(pts_i)
                stressed_tgt.append(tgt_i)
            points_t = torch.tensor(np.stack(stressed_pts), dtype=torch.float32).cuda().transpose(2, 1)
            target_np = np.stack(stressed_tgt)
            label_t = label.long().cuda()

            seg_pred = model(points_t, to_categorical(label_t, 16))
            logits_np = seg_pred.cpu().numpy()
            if logits_np.ndim == 3 and logits_np.shape[1] == 50:
                logits_np = np.transpose(logits_np, (0, 2, 1))
            pred_np = np.zeros(target_np.shape, dtype=np.int32)
            for i in range(target_np.shape[0]):
                cat = seg_label_to_cat[int(target_np[i, 0])]
                part_ids = seg_classes[cat]
                cat_logits = np.take(logits_np[i], part_ids, axis=1)
                pred_np[i] = np.argmax(cat_logits, axis=1) + part_ids[0]

            total_correct += int(np.sum(pred_np == target_np))
            total_seen += int(np.prod(target_np.shape))
            for part_id in range(50):
                total_seen_class[part_id] += int(np.sum(target_np == part_id))
                total_correct_class[part_id] += int(np.sum((pred_np == part_id) & (target_np == part_id)))

            for i in range(target_np.shape[0]):
                segp = pred_np[i]
                segl = target_np[i]
                cat = seg_label_to_cat[int(segl[0])]
                part_ious = []
                for part_id in seg_classes[cat]:
                    denom = np.sum((segl == part_id) | (segp == part_id))
                    if denom == 0:
                        part_ious.append(1.0)
                    else:
                        part_ious.append(float(np.sum((segl == part_id) & (segp == part_id)) / denom))
                shape_ious[cat].append(float(np.mean(part_ious)))

    per_cat_iou = {cat: float(np.mean(vals)) if vals else float("nan") for cat, vals in shape_ious.items()}
    all_shape_ious = [x for vals in shape_ious.values() for x in vals]
    class_acc = np.array(total_correct_class) / np.maximum(1, np.array(total_seen_class, dtype=float))
    return {
        "condition": condition,
        "accuracy": float(total_correct / max(1, total_seen)),
        "class_avg_accuracy": float(np.nanmean(class_acc)),
        "class_avg_iou": float(np.nanmean(list(per_cat_iou.values()))),
        "instance_avg_iou": float(np.mean(all_shape_ious)) if all_shape_ious else float("nan"),
        "per_category_iou": per_cat_iou,
    }


def to_md(summary):
    lines = [
        "# ShapeNetPart Support Stress",
        "",
        f"- ckpt: `{summary['ckpt']}`",
        f"- root: `{summary['root']}`",
        "",
        "| condition | accuracy | class avg IoU | instance avg IoU |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["conditions"]:
        lines.append(
            f"| `{row['condition']}` | `{row['accuracy']:.4f}` | `{row['class_avg_iou']:.4f}` | `{row['instance_avg_iou']:.4f}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `part_drop_largest` removes the largest ground-truth part within each object before resampling.")
    lines.append("- `part_keepXX_per_part` keeps XX% of each ground-truth part before resampling. These are support-stress probes, not official ShapeNetPart scores.")
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
