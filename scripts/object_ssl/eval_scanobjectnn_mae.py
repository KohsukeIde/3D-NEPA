#!/usr/bin/env python
"""ScanObjectNN Q3/Q4 diagnostics for Point-MAE-family checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from object_ssl_common import (
    GROUPING_MODES,
    SCANOBJECTNN_CLASS_NAMES,
    SCAN_CONDITIONS,
    confusion_matrix,
    file_sha256,
    git_commit,
    hardest_pair,
    jsonable,
    patch_eval_grouping,
    repo_root_from_script,
    stress_points,
    topk_metrics,
    write_json,
)


VARIANT_H5 = {
    "obj_bg": ("main_split", "test_objectdataset.h5"),
    "obj_only": ("main_split_nobg", "test_objectdataset.h5"),
    "pb_t50_rs": ("main_split", "test_objectdataset_augmentedrot_scale75.h5"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Point-MAE / PCP-MAE ScanObjectNN diagnostics")
    p.add_argument("--model", required=True, choices=["pointmae", "pcpmae"])
    p.add_argument("--repo-root", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variant", required=True, choices=sorted(VARIANT_H5))
    p.add_argument("--data-root", default="")
    p.add_argument("--h5", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)  # kept for provenance
    p.add_argument("--npoints", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--conditions", nargs="*", default=[x[0] for x in SCAN_CONDITIONS])
    p.add_argument("--grouping-mode", default="fps_knn", choices=GROUPING_MODES)
    p.add_argument("--selection-protocol", default="official_checkpoint")
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def import_repo(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    from tools import builder  # type: ignore
    from utils import misc  # type: ignore
    from utils.config import cfg_from_yaml_file  # type: ignore

    return builder, misc, cfg_from_yaml_file


def load_h5(args: argparse.Namespace, root: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    if args.h5:
        h5_path = Path(args.h5)
    else:
        data_root = Path(args.data_root) if args.data_root else root / "data" / "ScanObjectNN" / "h5_files"
        subdir, name = VARIANT_H5[args.variant]
        h5_path = data_root / subdir / name
    if not h5_path.is_file():
        raise FileNotFoundError(f"ScanObjectNN h5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        points = np.asarray(f["data"], dtype=np.float32)
        labels = np.asarray(f["label"], dtype=np.int64).reshape(-1)
    return points, labels, h5_path


def build_model(args: argparse.Namespace, builder, cfg_from_yaml_file):
    cfg = cfg_from_yaml_file(args.config)
    model = builder.model_builder(cfg.model)
    builder.load_model(model, args.checkpoint, logger=None)
    model = model.cuda().eval()
    return model, cfg


def eval_condition(
    *,
    model,
    misc,
    points_np: np.ndarray,
    labels_np: np.ndarray,
    condition_name: str,
    condition_kind: str,
    ratio: float,
    batch_size: int,
    npoints: int,
    seed: int,
    max_batches: int,
) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    logits_all = []
    labels_all = []

    with torch.no_grad():
        n_batches = int(np.ceil(points_np.shape[0] / batch_size))
        for batch_idx in range(n_batches):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            start = batch_idx * batch_size
            end = min(points_np.shape[0], start + batch_size)
            points = torch.from_numpy(points_np[start:end]).float()
            labels = torch.from_numpy(labels_np[start:end]).long()
            if points.shape[1] != npoints:
                points = misc.fps(points.cuda(), npoints).cpu()
            stressed = stress_points(points, condition_kind, ratio, generator).cuda()
            logits = model(stressed)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            logits_all.append(logits.detach().cpu())
            labels_all.append(labels)

    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    pred = logits_cat.argmax(dim=-1)
    top = topk_metrics(logits_cat, labels_cat, ks=(1, 2, 5))
    conf = confusion_matrix(pred.numpy(), labels_cat.numpy(), len(SCANOBJECTNN_CLASS_NAMES))
    clean = {
        "condition": condition_name,
        "top1": top["top1"],
        "top2_hit": top["top2_hit"],
        "top5_hit": top["top5_hit"],
        "oracle2_score": top["top2_hit"],
        "oracle5_score": top["top5_hit"],
        "n_samples": int(labels_cat.numel()),
        "confusion_matrix": conf.tolist(),
        "hardest_pair": hardest_pair(conf, SCANOBJECTNN_CLASS_NAMES),
        "per_class_acc": {},
    }
    for idx, name in enumerate(SCANOBJECTNN_CLASS_NAMES):
        mask = labels_cat == idx
        clean["per_class_acc"][name] = (
            float((pred[mask] == labels_cat[mask]).float().mean().item() * 100.0)
            if int(mask.sum().item()) > 0
            else None
        )
    return clean


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    repo_root = Path(args.repo_root).resolve()
    builder, misc, cfg_from_yaml_file = import_repo(repo_root)
    points, labels, h5_path = load_h5(args, root)
    model, cfg = build_model(args, builder, cfg_from_yaml_file)
    patched_groups = patch_eval_grouping(model, args.grouping_mode, args.seed)

    wanted = set(args.conditions)
    rows = []
    for idx, (name, kind, ratio) in enumerate(SCAN_CONDITIONS):
        if name not in wanted:
            continue
        rows.append(
            eval_condition(
                model=model,
                misc=misc,
                points_np=points,
                labels_np=labels,
                condition_name=name,
                condition_kind=kind,
                ratio=ratio,
                batch_size=args.batch_size,
                npoints=args.npoints,
                seed=args.seed + idx,
                max_batches=args.max_batches,
            )
        )

    clean_top1 = next((r["top1"] for r in rows if r["condition"] == "clean"), None)
    for row in rows:
        row["damage_pp"] = float(clean_top1 - row["top1"]) if clean_top1 is not None else None

    payload = {
        "metadata": {
            "model": args.model,
            "task": "scanobjectnn",
            "split": args.variant,
            "selection_protocol": args.selection_protocol,
            "grouping_mode": args.grouping_mode,
            "patched_group_modules": patched_groups,
            "repo_root": str(repo_root),
            "config": str(Path(args.config).resolve()),
            "checkpoint_path": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "data_path": str(h5_path.resolve()),
            "n_samples": int(labels.shape[0]),
            "seed": args.seed,
            "script": str(Path(__file__).resolve()),
            "git_commit": git_commit(root),
            "npoints": getattr(cfg, "npoints", args.npoints),
            "notes": (
                "structured_keep drops a local anchor neighborhood by retaining farthest points, "
                "matching existing PointGPT support-stress protocol; input-level FPS is skipped "
                "when the H5 point count already equals npoints. grouping_mode is an eval-time "
                "patchization perturbation with checkpoint/readout fixed; fps_knn is the unmodified path."
            ),
        },
        "conditions": rows,
    }
    write_json(args.output_json, jsonable(payload))
    print(f"[done] wrote {args.output_json}")


if __name__ == "__main__":
    main()
