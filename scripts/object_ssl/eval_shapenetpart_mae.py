#!/usr/bin/env python
"""ShapeNetPart Q3/Q4 diagnostics for Point-MAE-family segmentation checkpoints."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

from object_ssl_common import (
    GROUPING_MODES,
    PART_CONDITIONS,
    SEG_CLASSES,
    SEG_LABEL_TO_CAT,
    file_sha256,
    git_commit,
    jsonable,
    patch_eval_grouping,
    resample_indices,
    repo_root_from_script,
    shape_iou_metrics,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Point-MAE / PCP-MAE ShapeNetPart diagnostics")
    p.add_argument("--model", required=True, choices=["pointmae", "pcpmae"])
    p.add_argument("--repo-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--normal", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--conditions", nargs="*", default=[x[0] for x in PART_CONDITIONS])
    p.add_argument("--grouping-mode", default="fps_knn", choices=GROUPING_MODES)
    p.add_argument("--selection-protocol", default="official_checkpoint")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def import_repo(repo_root: Path):
    seg_root = repo_root / "segmentation"
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(seg_root))
    sys.path.insert(0, str(seg_root / "models"))
    os.chdir(seg_root)
    dataset_mod = importlib.import_module("dataset")
    model_mod = importlib.import_module("pt")
    return dataset_mod.PartNormalDataset, model_mod


def to_categorical(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = labels.view(-1).detach().to(device)
    return torch.eye(num_classes, device=device)[labels]


def load_seg_checkpoint(classifier: torch.nn.Module, checkpoint: str) -> str:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        checkpoint_state = {k.replace("module.", ""): v for k, v in state["model_state_dict"].items()}
    elif isinstance(state, dict) and "base_model" in state:
        checkpoint_state = {k.replace("module.", ""): v for k, v in state["base_model"].items()}
    elif isinstance(state, dict) and all(torch.is_tensor(v) for v in state.values()):
        checkpoint_state = {k.replace("module.", ""): v for k, v in state.items()}
    else:
        raise RuntimeError(f"Unsupported ShapeNetPart checkpoint format: {checkpoint}")
    incompatible = classifier.load_state_dict(checkpoint_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "ShapeNetPart checkpoint is missing model keys: "
            + ", ".join(incompatible.missing_keys[:20])
        )
    if incompatible.unexpected_keys:
        note = "ignored unexpected checkpoint keys: " + ", ".join(incompatible.unexpected_keys[:20])
        print(f"[checkpoint] {note}")
        return note
    return "checkpoint loaded with exact model-key coverage"


def category_restricted_predictions(logits: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top1/top2/top5 predictions under the standard known-category protocol."""
    logits_np = logits.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    bsz, npoints, _ = logits_np.shape
    top1 = np.zeros((bsz, npoints), dtype=np.int64)
    top2_hit = np.zeros((bsz, npoints), dtype=bool)
    top5_hit = np.zeros((bsz, npoints), dtype=bool)
    for i in range(bsz):
        cat = SEG_LABEL_TO_CAT[int(target_np[i, 0])]
        parts = SEG_CLASSES[cat]
        allowed_logits = logits_np[i][:, parts]
        order = np.argsort(-allowed_logits, axis=1)
        sorted_parts = np.asarray(parts, dtype=np.int64)[order]
        top1[i] = sorted_parts[:, 0]
        top2 = sorted_parts[:, : min(2, sorted_parts.shape[1])]
        top5 = sorted_parts[:, : min(5, sorted_parts.shape[1])]
        top2_hit[i] = np.any(top2 == target_np[i, :, None], axis=1)
        top5_hit[i] = np.any(top5 == target_np[i, :, None], axis=1)
    return top1, top2_hit, top5_hit


def oracle_predictions(top1: np.ndarray, hit: np.ndarray, target: np.ndarray) -> np.ndarray:
    out = top1.copy()
    out[hit] = target[hit]
    return out


def label_entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0].astype(np.float64) / total
    return float(-(probs * np.log2(probs)).sum())


def part_hist(labels: torch.Tensor) -> np.ndarray:
    return np.bincount(labels.detach().cpu().numpy().reshape(-1), minlength=50).astype(np.int64)


def support_indices_for_batch(
    points: torch.Tensor,
    target: torch.Tensor,
    condition: str,
    ratio: float,
    generator: torch.Generator,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int]]:
    bsz, npoints, _ = points.shape
    keep_n = max(1, int(round(npoints * ratio)))
    keep_indices = []
    forward_indices = []
    removed_parts = []
    for b in range(bsz):
        pts = points[b]
        tgt = target[b]
        removed_part = -1
        if condition in {"clean", "xyz_zero"}:
            keep_idx = torch.arange(npoints)
        elif condition == "random_drop":
            keep_idx = torch.randperm(npoints, generator=generator)[:keep_n]
        elif condition == "structured_drop":
            anchor = int(torch.randint(0, npoints, (1,), generator=generator).item())
            dist = torch.sum((pts[:, :3] - pts[anchor : anchor + 1, :3]) ** 2, dim=-1)
            keep_idx = torch.topk(dist, k=keep_n, largest=True).indices
        elif condition == "largest_part_removed":
            uniq, counts = torch.unique(tgt, return_counts=True)
            largest = uniq[counts.argmax()]
            removed_part = int(largest.item())
            keep_idx = torch.nonzero(tgt != largest, as_tuple=False).view(-1)
            if keep_idx.numel() == 0:
                keep_idx = torch.arange(npoints)
                removed_part = -1
        elif condition == "part_keep":
            keep_chunks = []
            for part in torch.unique(tgt):
                part_idx = torch.nonzero(tgt == part, as_tuple=False).view(-1)
                part_keep = max(1, int(round(part_idx.numel() * ratio)))
                perm = torch.randperm(part_idx.numel(), generator=generator)[:part_keep]
                keep_chunks.append(part_idx[perm])
            keep_idx = torch.cat(keep_chunks, dim=0)
        else:
            raise ValueError(f"Unsupported segmentation condition: {condition}")
        keep_idx = torch.unique(keep_idx.long(), sorted=True)
        if keep_idx.numel() == npoints:
            forward_idx = keep_idx.clone().long()
        else:
            forward_idx = resample_indices(keep_idx, npoints, generator).long()
        keep_indices.append(keep_idx)
        forward_indices.append(forward_idx)
        removed_parts.append(removed_part)
    return keep_indices, forward_indices, removed_parts


def build_forward_points(points: torch.Tensor, forward_indices: list[torch.Tensor], condition: str) -> torch.Tensor:
    rows = []
    for b, idx in enumerate(forward_indices):
        row = points[b, idx].clone()
        if condition == "xyz_zero":
            row[:, :3] = 0
        rows.append(row)
    return torch.stack(rows, dim=0)


def average_logits_by_original_index(
    logits_forward: torch.Tensor,
    forward_idx: torch.Tensor,
    keep_idx: torch.Tensor,
) -> torch.Tensor:
    inverse = torch.full(
        (int(max(int(forward_idx.max().item()), int(keep_idx.max().item()))) + 1,),
        -1,
        dtype=torch.long,
        device=logits_forward.device,
    )
    keep_idx_dev = keep_idx.to(logits_forward.device)
    forward_idx_dev = forward_idx.to(logits_forward.device)
    inverse[keep_idx_dev] = torch.arange(keep_idx.numel(), device=logits_forward.device)
    local_idx = inverse[forward_idx_dev]
    if torch.any(local_idx < 0):
        raise RuntimeError("forward_idx contains a point outside keep_idx")
    logits_unique = torch.zeros(
        keep_idx.numel(),
        logits_forward.shape[-1],
        dtype=logits_forward.dtype,
        device=logits_forward.device,
    )
    counts = torch.zeros(keep_idx.numel(), 1, dtype=logits_forward.dtype, device=logits_forward.device)
    logits_unique.index_add_(0, local_idx, logits_forward)
    counts.index_add_(0, local_idx, torch.ones(logits_forward.shape[0], 1, dtype=logits_forward.dtype, device=logits_forward.device))
    return logits_unique / counts.clamp_min(1.0)


def category_restricted_prediction_1d(logits: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits_np = logits.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    cat = SEG_LABEL_TO_CAT[int(target_np[0])]
    parts = SEG_CLASSES[cat]
    allowed_logits = logits_np[:, parts]
    order = np.argsort(-allowed_logits, axis=1)
    sorted_parts = np.asarray(parts, dtype=np.int64)[order]
    pred = sorted_parts[:, 0]
    top2 = sorted_parts[:, : min(2, sorted_parts.shape[1])]
    top5 = sorted_parts[:, : min(5, sorted_parts.shape[1])]
    hit2 = np.any(top2 == target_np[:, None], axis=1)
    hit5 = np.any(top5 == target_np[:, None], axis=1)
    return pred, hit2, hit5


def summarize_prediction_lists(
    *,
    pred_batches: list[np.ndarray],
    target_batches: list[np.ndarray],
    top2_hits: list[np.ndarray],
    top5_hits: list[np.ndarray],
) -> dict:
    hit2_all = np.concatenate(top2_hits, axis=0)
    hit5_all = np.concatenate(top5_hits, axis=0)
    target_all = np.concatenate(target_batches, axis=0)
    pred_all = np.concatenate(pred_batches, axis=0)
    oracle2_batches = [oracle_predictions(pred, hit, target) for pred, hit, target in zip(pred_batches, top2_hits, target_batches)]
    oracle5_batches = [oracle_predictions(pred, hit, target) for pred, hit, target in zip(pred_batches, top5_hits, target_batches)]
    base_iou = shape_iou_metrics(pred_batches, target_batches)
    oracle2_iou = shape_iou_metrics(oracle2_batches, target_batches)
    oracle5_iou = shape_iou_metrics(oracle5_batches, target_batches)
    return {
        "point_top1": float((pred_all == target_all).mean() * 100.0),
        "point_top2_hit": float(hit2_all.mean() * 100.0),
        "point_top5_hit": float(hit5_all.mean() * 100.0),
        "class_avg_miou": base_iou["class_avg_miou"],
        "instance_avg_miou": base_iou["instance_avg_miou"],
        "oracle2_class_avg_miou": oracle2_iou["class_avg_miou"],
        "oracle2_instance_avg_miou": oracle2_iou["instance_avg_miou"],
        "oracle5_class_avg_miou": oracle5_iou["class_avg_miou"],
        "oracle5_instance_avg_miou": oracle5_iou["instance_avg_miou"],
        "n_shapes": len(target_batches),
        "n_points": int(target_all.size),
    }


def new_condition_accumulator() -> dict:
    return {
        "pred_batches": [],
        "target_batches": [],
        "top2_hits": [],
        "top5_hits": [],
        "clean_pred_batches": [],
        "clean_top2_hits": [],
        "clean_top5_hits": [],
        "before_hist": np.zeros(50, dtype=np.int64),
        "retained_hist": np.zeros(50, dtype=np.int64),
        "removed_hist": np.zeros(50, dtype=np.int64),
        "retained_counts": [],
        "forward_counts": [],
        "repeated_forward_counts": [],
    }


def update_condition_accumulator(
    acc: dict,
    *,
    target_full: torch.Tensor,
    keep_idx: torch.Tensor,
    forward_idx: torch.Tensor,
    removed_part: int,
    logits_forward: torch.Tensor,
    clean_logits_full: torch.Tensor,
) -> None:
    target_unique = target_full[keep_idx].long()
    logits_unique = average_logits_by_original_index(logits_forward, forward_idx, keep_idx)
    pred, hit2, hit5 = category_restricted_prediction_1d(logits_unique, target_unique)
    clean_pred, clean_hit2, clean_hit5 = category_restricted_prediction_1d(clean_logits_full[keep_idx], target_unique)

    target_np = target_unique.numpy()
    acc["pred_batches"].append(pred)
    acc["target_batches"].append(target_np)
    acc["top2_hits"].append(hit2)
    acc["top5_hits"].append(hit5)
    acc["clean_pred_batches"].append(clean_pred)
    acc["clean_top2_hits"].append(clean_hit2)
    acc["clean_top5_hits"].append(clean_hit5)

    acc["before_hist"] += part_hist(target_full)
    acc["retained_hist"] += part_hist(target_unique)
    if removed_part >= 0:
        acc["removed_hist"][removed_part] += 1
    acc["retained_counts"].append(int(keep_idx.numel()))
    acc["forward_counts"].append(int(forward_idx.numel()))
    acc["repeated_forward_counts"].append(int(forward_idx.numel() - keep_idx.numel()))


def finalize_condition(condition_name: str, acc: dict) -> dict:
    perturbed = summarize_prediction_lists(
        pred_batches=acc["pred_batches"],
        target_batches=acc["target_batches"],
        top2_hits=acc["top2_hits"],
        top5_hits=acc["top5_hits"],
    )
    clean_subset = summarize_prediction_lists(
        pred_batches=acc["clean_pred_batches"],
        target_batches=acc["target_batches"],
        top2_hits=acc["clean_top2_hits"],
        top5_hits=acc["clean_top5_hits"],
    )
    retained_counts_np = np.asarray(acc["retained_counts"], dtype=np.float64)
    forward_counts_np = np.asarray(acc["forward_counts"], dtype=np.float64)
    repeated_counts_np = np.asarray(acc["repeated_forward_counts"], dtype=np.float64)
    return {
        "condition": condition_name,
        **perturbed,
        "clean_subset_point_top1": clean_subset["point_top1"],
        "clean_subset_point_top2_hit": clean_subset["point_top2_hit"],
        "clean_subset_point_top5_hit": clean_subset["point_top5_hit"],
        "clean_subset_class_avg_miou": clean_subset["class_avg_miou"],
        "clean_subset_instance_avg_miou": clean_subset["instance_avg_miou"],
        "damage_pp": float(clean_subset["instance_avg_miou"] - perturbed["instance_avg_miou"]),
        "metric_scope": "unique_retained_original_points",
        "logit_aggregation": "mean_by_original_index",
        "n_forward_points": int(forward_counts_np.sum()),
        "support_provenance": {
            "mean_retained_unique_points": float(retained_counts_np.mean()),
            "min_retained_unique_points": int(retained_counts_np.min()),
            "max_retained_unique_points": int(retained_counts_np.max()),
            "mean_forward_points": float(forward_counts_np.mean()),
            "mean_repeated_forward_points": float(repeated_counts_np.mean()),
            "mean_forward_duplication_factor": float(np.mean(forward_counts_np / retained_counts_np)),
            "part_histogram_before": acc["before_hist"].tolist(),
            "part_histogram_retained": acc["retained_hist"].tolist(),
            "largest_removed_part_histogram": acc["removed_hist"].tolist(),
            "part_entropy_before": label_entropy(acc["before_hist"]),
            "part_entropy_retained": label_entropy(acc["retained_hist"]),
        },
    }


def repeat_labels_for_conditions(labels: torch.Tensor, n_conditions: int) -> torch.Tensor:
    reps = [n_conditions] + [1] * (labels.dim() - 1)
    return labels.repeat(*reps)


def eval_conditions(
    *,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    conditions: list[tuple[str, str, float, int]],
    max_batches: int,
) -> list[dict]:
    accumulators = {name: new_condition_accumulator() for name, _, _, _ in conditions}
    generators = {
        name: torch.Generator(device="cpu").manual_seed(seed)
        for name, _, _, seed in conditions
    }
    clean_index = next((i for i, (_, kind, _, _) in enumerate(conditions) if kind == "clean"), None)

    classifier.eval()
    with torch.no_grad():
        for batch_idx, (points, label, target) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            points = points.float()
            target = target.long()
            condition_inputs = []
            condition_support = {}
            for name, kind, ratio, _ in conditions:
                keep_indices, forward_indices, removed_parts = support_indices_for_batch(
                    points,
                    target,
                    kind,
                    ratio,
                    generators[name],
                )
                condition_inputs.append(build_forward_points(points, forward_indices, kind))
                condition_support[name] = (keep_indices, forward_indices, removed_parts)

            device = torch.device("cuda")
            batch_size = points.shape[0]
            stacked_points = torch.cat(condition_inputs, dim=0).to(device)
            stacked_labels = repeat_labels_for_conditions(label.long(), len(conditions)).to(device)
            logits_all = classifier(stacked_points.transpose(2, 1), to_categorical(stacked_labels, 16, device)).detach().cpu()
            logits_by_condition = list(torch.split(logits_all, batch_size, dim=0))
            if clean_index is None:
                clean_logits = classifier(points.to(device).transpose(2, 1), to_categorical(label.long().to(device), 16, device)).detach().cpu()
            else:
                clean_logits = logits_by_condition[clean_index]

            for cond_idx, (name, _, _, _) in enumerate(conditions):
                keep_indices, forward_indices, removed_parts = condition_support[name]
                logits_condition = logits_by_condition[cond_idx]
                for b, keep_idx in enumerate(keep_indices):
                    update_condition_accumulator(
                        accumulators[name],
                        target_full=target[b],
                        keep_idx=keep_idx,
                        forward_idx=forward_indices[b],
                        removed_part=removed_parts[b],
                        logits_forward=logits_condition[b],
                        clean_logits_full=clean_logits[b],
                    )
    return [finalize_condition(name, accumulators[name]) for name, _, _, _ in conditions]


def eval_condition(
    *,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    condition_name: str,
    condition_kind: str,
    ratio: float,
    seed: int,
    max_batches: int,
) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pred_batches = []
    target_batches = []
    top2_hits = []
    top5_hits = []
    clean_pred_batches = []
    clean_top2_hits = []
    clean_top5_hits = []
    before_hist = np.zeros(50, dtype=np.int64)
    retained_hist = np.zeros(50, dtype=np.int64)
    removed_hist = np.zeros(50, dtype=np.int64)
    retained_counts = []
    forward_counts = []
    repeated_forward_counts = []

    classifier.eval()
    with torch.no_grad():
        for batch_idx, (points, label, target) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            points = points.float()
            target = target.long()
            keep_indices, forward_indices, removed_parts = support_indices_for_batch(
                points,
                target,
                condition_kind,
                ratio,
                generator,
            )
            device = torch.device("cuda")
            stressed_points = build_forward_points(points, forward_indices, condition_kind).to(device)
            labels = label.long().to(device)
            clean_points = points.to(device)
            clean_logits = classifier(clean_points.transpose(2, 1), to_categorical(labels, 16, device)).detach().cpu()
            if condition_kind == "clean":
                logits = clean_logits
            else:
                logits = classifier(stressed_points.transpose(2, 1), to_categorical(labels, 16, device)).detach().cpu()
            for b, keep_idx in enumerate(keep_indices):
                forward_idx = forward_indices[b]
                target_unique = target[b, keep_idx].long()
                logits_unique = average_logits_by_original_index(logits[b], forward_idx, keep_idx)
                pred, hit2, hit5 = category_restricted_prediction_1d(logits_unique, target_unique)
                clean_pred, clean_hit2, clean_hit5 = category_restricted_prediction_1d(clean_logits[b, keep_idx], target_unique)

                target_np = target_unique.numpy()
                pred_batches.append(pred)
                target_batches.append(target_np)
                top2_hits.append(hit2)
                top5_hits.append(hit5)
                clean_pred_batches.append(clean_pred)
                clean_top2_hits.append(clean_hit2)
                clean_top5_hits.append(clean_hit5)

                before_hist += part_hist(target[b])
                retained_hist += part_hist(target_unique)
                if removed_parts[b] >= 0:
                    removed_hist[removed_parts[b]] += 1
                retained_counts.append(int(keep_idx.numel()))
                forward_counts.append(int(forward_idx.numel()))
                repeated_forward_counts.append(int(forward_idx.numel() - keep_idx.numel()))

    perturbed = summarize_prediction_lists(
        pred_batches=pred_batches,
        target_batches=target_batches,
        top2_hits=top2_hits,
        top5_hits=top5_hits,
    )
    clean_subset = summarize_prediction_lists(
        pred_batches=clean_pred_batches,
        target_batches=target_batches,
        top2_hits=clean_top2_hits,
        top5_hits=clean_top5_hits,
    )
    retained_counts_np = np.asarray(retained_counts, dtype=np.float64)
    forward_counts_np = np.asarray(forward_counts, dtype=np.float64)
    repeated_counts_np = np.asarray(repeated_forward_counts, dtype=np.float64)
    return {
        "condition": condition_name,
        **perturbed,
        "clean_subset_point_top1": clean_subset["point_top1"],
        "clean_subset_point_top2_hit": clean_subset["point_top2_hit"],
        "clean_subset_point_top5_hit": clean_subset["point_top5_hit"],
        "clean_subset_class_avg_miou": clean_subset["class_avg_miou"],
        "clean_subset_instance_avg_miou": clean_subset["instance_avg_miou"],
        "damage_pp": float(clean_subset["instance_avg_miou"] - perturbed["instance_avg_miou"]),
        "metric_scope": "unique_retained_original_points",
        "logit_aggregation": "mean_by_original_index",
        "n_forward_points": int(forward_counts_np.sum()),
        "support_provenance": {
            "mean_retained_unique_points": float(retained_counts_np.mean()),
            "min_retained_unique_points": int(retained_counts_np.min()),
            "max_retained_unique_points": int(retained_counts_np.max()),
            "mean_forward_points": float(forward_counts_np.mean()),
            "mean_repeated_forward_points": float(repeated_counts_np.mean()),
            "mean_forward_duplication_factor": float(np.mean(forward_counts_np / retained_counts_np)),
            "part_histogram_before": before_hist.tolist(),
            "part_histogram_retained": retained_hist.tolist(),
            "largest_removed_part_histogram": removed_hist.tolist(),
            "part_entropy_before": label_entropy(before_hist),
            "part_entropy_retained": label_entropy(retained_hist),
        },
    }


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    repo_root = Path(args.repo_root).resolve()
    dataset_cls, model_mod = import_repo(repo_root)
    dataset = dataset_cls(
        root=args.data_root,
        npoints=args.npoints,
        split="test",
        normal_channel=args.normal,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    classifier = model_mod.get_model(50).cuda()
    checkpoint_note = load_seg_checkpoint(classifier, args.checkpoint)
    patched_groups = patch_eval_grouping(classifier, args.grouping_mode, args.seed)

    wanted = set(args.conditions)
    condition_specs = []
    for idx, (name, kind, ratio) in enumerate(PART_CONDITIONS):
        if name not in wanted:
            continue
        condition_specs.append((name, kind, ratio, args.seed + idx))
    rows = eval_conditions(
        classifier=classifier,
        loader=loader,
        conditions=condition_specs,
        max_batches=args.max_batches,
    )

    payload = {
        "metadata": {
            "model": args.model,
            "task": "shapenetpart",
            "split": "test",
            "selection_protocol": args.selection_protocol,
            "grouping_mode": args.grouping_mode,
            "patched_group_modules": patched_groups,
            "repo_root": str(repo_root),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "data_path": str(Path(args.data_root).resolve()),
            "n_samples": len(dataset),
            "seed": args.seed,
            "script": str(Path(__file__).resolve()),
            "git_commit": git_commit(root),
            "npoints": args.npoints,
            "notes": (
                "ShapeNetPart top-k is computed under the standard known-category part-label restriction "
                "used by the original evaluation. grouping_mode is an eval-time patchization perturbation "
                "with checkpoint/readout fixed; fps_knn is the unmodified path. Support perturbation "
                "metrics are computed on unique retained original point indices. When retained support "
                "is resampled for the fixed-size forward pass, logits are averaged back to each original "
                f"retained point before mIoU/top-k computation; {checkpoint_note}."
            ),
        },
        "conditions": rows,
    }
    write_json(args.output_json, jsonable(payload))
    print(f"[done] wrote {args.output_json}")


if __name__ == "__main__":
    main()
