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
    PART_CONDITIONS,
    SEG_CLASSES,
    SEG_LABEL_TO_CAT,
    file_sha256,
    git_commit,
    jsonable,
    repo_root_from_script,
    shape_iou_metrics,
    stress_points_and_labels,
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
    oracle2_batches = []
    oracle5_batches = []
    top2_hits = []
    top5_hits = []

    classifier.eval()
    with torch.no_grad():
        for batch_idx, (points, label, target) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            points = points.float()
            target = target.long()
            stressed_points, stressed_target = stress_points_and_labels(
                points,
                target,
                condition_kind,
                ratio,
                generator,
            )
            device = torch.device("cuda")
            stressed_points = stressed_points.to(device)
            labels = label.long().to(device)
            logits = classifier(stressed_points.transpose(2, 1), to_categorical(labels, 16, device))
            pred, hit2, hit5 = category_restricted_predictions(logits, stressed_target)
            target_np = stressed_target.numpy()
            pred_batches.append(pred)
            target_batches.append(target_np)
            top2_hits.append(hit2)
            top5_hits.append(hit5)
            oracle2_batches.append(oracle_predictions(pred, hit2, target_np))
            oracle5_batches.append(oracle_predictions(pred, hit5, target_np))

    pred_all = np.concatenate(pred_batches, axis=0)
    target_all = np.concatenate(target_batches, axis=0)
    hit2_all = np.concatenate(top2_hits, axis=0)
    hit5_all = np.concatenate(top5_hits, axis=0)
    oracle2_all = np.concatenate(oracle2_batches, axis=0)
    oracle5_all = np.concatenate(oracle5_batches, axis=0)

    base_iou = shape_iou_metrics(pred_all, target_all)
    oracle2_iou = shape_iou_metrics(oracle2_all, target_all)
    oracle5_iou = shape_iou_metrics(oracle5_all, target_all)
    point_top1 = float((pred_all == target_all).mean() * 100.0)
    return {
        "condition": condition_name,
        "point_top1": point_top1,
        "point_top2_hit": float(hit2_all.mean() * 100.0),
        "point_top5_hit": float(hit5_all.mean() * 100.0),
        "class_avg_miou": base_iou["class_avg_miou"],
        "instance_avg_miou": base_iou["instance_avg_miou"],
        "oracle2_class_avg_miou": oracle2_iou["class_avg_miou"],
        "oracle2_instance_avg_miou": oracle2_iou["instance_avg_miou"],
        "oracle5_class_avg_miou": oracle5_iou["class_avg_miou"],
        "oracle5_instance_avg_miou": oracle5_iou["instance_avg_miou"],
        "n_shapes": int(target_all.shape[0]),
        "n_points": int(target_all.size),
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

    wanted = set(args.conditions)
    rows = []
    for idx, (name, kind, ratio) in enumerate(PART_CONDITIONS):
        if name not in wanted:
            continue
        rows.append(
            eval_condition(
                classifier=classifier,
                loader=loader,
                condition_name=name,
                condition_kind=kind,
                ratio=ratio,
                seed=args.seed + idx,
                max_batches=args.max_batches,
            )
        )

    clean_miou = next((r["instance_avg_miou"] for r in rows if r["condition"] == "clean"), None)
    for row in rows:
        row["damage_pp"] = float(clean_miou - row["instance_avg_miou"]) if clean_miou is not None else None

    payload = {
        "metadata": {
            "model": args.model,
            "task": "shapenetpart",
            "split": "test",
            "selection_protocol": args.selection_protocol,
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
                f"used by the original evaluation; {checkpoint_note}."
            ),
        },
        "conditions": rows,
    }
    write_json(args.output_json, jsonable(payload))
    print(f"[done] wrote {args.output_json}")


if __name__ == "__main__":
    main()
