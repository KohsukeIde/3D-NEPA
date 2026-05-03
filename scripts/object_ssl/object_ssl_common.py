#!/usr/bin/env python
"""Shared utilities for Point-MAE / PCP-MAE object SSL diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

import numpy as np
import torch


SCANOBJECTNN_CLASS_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]


SEG_CLASSES = {
    "Earphone": [16, 17, 18],
    "Motorbike": [30, 31, 32, 33, 34, 35],
    "Rocket": [41, 42, 43],
    "Car": [8, 9, 10, 11],
    "Laptop": [28, 29],
    "Cap": [6, 7],
    "Skateboard": [44, 45, 46],
    "Mug": [36, 37],
    "Guitar": [19, 20, 21],
    "Bag": [4, 5],
    "Lamp": [24, 25, 26, 27],
    "Table": [47, 48, 49],
    "Airplane": [0, 1, 2, 3],
    "Pistol": [38, 39, 40],
    "Chair": [12, 13, 14, 15],
    "Knife": [22, 23],
}

SEG_LABEL_TO_CAT = {label: cat for cat, labels in SEG_CLASSES.items() for label in labels}

SCAN_CONDITIONS = [
    ("clean", "clean", 1.0),
    ("random_keep80", "random_drop", 0.8),
    ("random_keep50", "random_drop", 0.5),
    ("random_keep20", "random_drop", 0.2),
    ("random_keep10", "random_drop", 0.1),
    ("structured_keep80", "structured_drop", 0.8),
    ("structured_keep50", "structured_drop", 0.5),
    ("structured_keep20", "structured_drop", 0.2),
    ("structured_keep10", "structured_drop", 0.1),
    ("xyz_zero", "xyz_zero", 0.0),
]

PART_CONDITIONS = [
    ("clean", "clean", 1.0),
    ("random_keep80", "random_drop", 0.8),
    ("random_keep50", "random_drop", 0.5),
    ("random_keep20", "random_drop", 0.2),
    ("random_keep10", "random_drop", 0.1),
    ("structured_keep80", "structured_drop", 0.8),
    ("structured_keep50", "structured_drop", 0.5),
    ("structured_keep20", "structured_drop", 0.2),
    ("structured_keep10", "structured_drop", 0.1),
    ("largest_part_removed", "largest_part_removed", 0.0),
    ("part_keep20_per_part", "part_keep", 0.2),
    ("xyz_zero", "xyz_zero", 0.0),
]

GROUPING_MODES = ["fps_knn", "random_center_knn", "random_group"]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)


def git_commit(root: str | Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def topk_metrics(logits: torch.Tensor, labels: torch.Tensor, ks: Iterable[int] = (1, 2, 5)) -> dict[str, float]:
    labels = labels.view(-1)
    max_k = min(max(ks), logits.shape[-1])
    top = logits.topk(max_k, dim=-1).indices
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, logits.shape[-1])
        hit = (top[:, :kk] == labels[:, None]).any(dim=1).float().mean().item() * 100.0
        out[f"top{k}_hit"] = hit
    out["top1"] = out.get("top1_hit", float("nan"))
    return out


def confusion_matrix(pred: np.ndarray, labels: np.ndarray, n_cls: int) -> np.ndarray:
    mat = np.zeros((n_cls, n_cls), dtype=np.int64)
    for gt, pr in zip(labels.reshape(-1), pred.reshape(-1)):
        if 0 <= int(gt) < n_cls and 0 <= int(pr) < n_cls:
            mat[int(gt), int(pr)] += 1
    return mat


def hardest_pair(confusion: np.ndarray, names: list[str]) -> dict[str, Any]:
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    if off.size == 0 or off.max() == 0:
        return {"true": "", "pred": "", "count": 0}
    true_idx, pred_idx = np.unravel_index(int(off.argmax()), off.shape)
    return {
        "true": names[true_idx] if true_idx < len(names) else str(true_idx),
        "pred": names[pred_idx] if pred_idx < len(names) else str(pred_idx),
        "count": int(off[true_idx, pred_idx]),
    }


def _batched_gather_points(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size, num_points, channels = xyz.shape
    idx_base = torch.arange(batch_size, device=xyz.device).view(batch_size, 1, 1) * num_points
    flat_idx = (idx + idx_base).reshape(-1)
    gathered = xyz.reshape(batch_size * num_points, channels)[flat_idx]
    return gathered.reshape(batch_size, idx.shape[1], idx.shape[2], channels)


def _select_random_centers(
    xyz: torch.Tensor,
    num_group: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, num_points, channels = xyz.shape
    rows = []
    for _ in range(batch_size):
        if num_points >= num_group:
            idx = torch.randperm(num_points, device=xyz.device, generator=generator)[:num_group]
        else:
            idx = torch.randint(0, num_points, (num_group,), device=xyz.device, generator=generator)
        rows.append(idx)
    center_idx = torch.stack(rows, dim=0)
    return xyz.gather(1, center_idx.unsqueeze(-1).expand(-1, -1, channels))


def _knn_indices(xyz: torch.Tensor, center: torch.Tensor, group_size: int) -> torch.Tensor:
    dist = torch.cdist(center.float(), xyz.float())
    return torch.topk(dist, k=group_size, dim=-1, largest=False, sorted=False).indices


def _random_group_indices(
    xyz: torch.Tensor,
    num_group: int,
    group_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size, num_points, _ = xyz.shape
    rows = [
        torch.randint(0, num_points, (num_group, group_size), device=xyz.device, generator=generator)
        for _ in range(batch_size)
    ]
    return torch.stack(rows, dim=0)


def make_eval_group_forward(grouping_mode: str, seed: int):
    if grouping_mode not in GROUPING_MODES:
        raise ValueError(f"unsupported grouping mode: {grouping_mode}")

    def forward(self, xyz):
        call_idx = int(getattr(self, "_object_ssl_grouping_calls", 0))
        self._object_ssl_grouping_calls = call_idx + 1
        generator = torch.Generator(device=xyz.device)
        generator.manual_seed(int(seed) + call_idx)

        if grouping_mode == "random_center_knn":
            center = _select_random_centers(xyz, self.num_group, generator)
            idx = _knn_indices(xyz, center, self.group_size)
        elif grouping_mode == "random_group":
            center = _select_random_centers(xyz, self.num_group, generator)
            idx = _random_group_indices(xyz, self.num_group, self.group_size, generator)
        else:
            raise ValueError("fps_knn uses the model's original Group.forward and should not be patched")

        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        neighborhood = _batched_gather_points(xyz, idx).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

    return forward


def patch_eval_grouping(model: torch.nn.Module, grouping_mode: str, seed: int) -> int:
    """Patch Point-MAE-family Group modules for inference-time grouping probes.

    `fps_knn` is the unmodified trained path. The random modes are eval-only
    perturbations matching the PointGPT grouping audit: they do not retrain the
    representation or the readout.
    """
    if grouping_mode == "fps_knn":
        return 0
    count = 0
    for module in model.modules():
        if hasattr(module, "num_group") and hasattr(module, "group_size") and type(module).__name__ == "Group":
            module.forward = MethodType(make_eval_group_forward(grouping_mode, seed), module)
            count += 1
    if count == 0:
        raise RuntimeError(f"no Group modules patched for grouping_mode={grouping_mode}")
    return count


def resample_indices(keep_idx: torch.Tensor, npoints: int, generator: torch.Generator) -> torch.Tensor:
    if keep_idx.numel() >= npoints:
        perm = torch.randperm(keep_idx.numel(), generator=generator)[:npoints]
        return keep_idx[perm]
    extra = torch.randint(0, keep_idx.numel(), (npoints - keep_idx.numel(),), generator=generator)
    return torch.cat([keep_idx, keep_idx[extra]], dim=0)


def stress_points(points: torch.Tensor, condition: str, ratio: float, generator: torch.Generator) -> torch.Tensor:
    """Apply PointGPT-matched ScanObjectNN support stresses.

    structured_drop follows the existing PointGPT audit implementation: choose a random
    anchor, drop its local neighborhood, and keep the farthest ratio of points.
    """
    if condition == "clean":
        return points
    if condition == "xyz_zero":
        return torch.zeros_like(points)
    bsz, npoints, _ = points.shape
    keep_n = max(1, int(round(npoints * ratio)))
    out = []
    for b in range(bsz):
        pts = points[b]
        if condition == "random_drop":
            keep_idx = torch.randperm(npoints, generator=generator)[:keep_n]
        elif condition == "structured_drop":
            anchor = int(torch.randint(0, npoints, (1,), generator=generator).item())
            dist = torch.sum((pts[:, :3] - pts[anchor : anchor + 1, :3]) ** 2, dim=-1)
            keep_idx = torch.topk(dist, k=keep_n, largest=True).indices
        else:
            raise ValueError(f"Unsupported point condition: {condition}")
        idx = resample_indices(keep_idx, npoints, generator)
        out.append(pts[idx])
    return torch.stack(out, dim=0)


def stress_points_and_labels(
    points: torch.Tensor,
    labels: torch.Tensor,
    condition: str,
    ratio: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "clean":
        return points, labels
    if condition == "xyz_zero":
        out = points.clone()
        out[:, :, :3] = 0
        return out, labels

    bsz, npoints, _ = points.shape
    out_points = []
    out_labels = []
    keep_n = max(1, int(round(npoints * ratio)))
    for b in range(bsz):
        pts = points[b]
        tgt = labels[b]
        if condition == "random_drop":
            keep_idx = torch.randperm(npoints, generator=generator)[:keep_n]
        elif condition == "structured_drop":
            anchor = int(torch.randint(0, npoints, (1,), generator=generator).item())
            dist = torch.sum((pts[:, :3] - pts[anchor : anchor + 1, :3]) ** 2, dim=-1)
            keep_idx = torch.topk(dist, k=keep_n, largest=True).indices
        elif condition == "largest_part_removed":
            uniq, counts = torch.unique(tgt, return_counts=True)
            largest = uniq[counts.argmax()]
            keep_idx = torch.nonzero(tgt != largest, as_tuple=False).view(-1)
            if keep_idx.numel() == 0:
                keep_idx = torch.arange(npoints)
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
        idx = resample_indices(keep_idx, npoints, generator)
        out_points.append(pts[idx])
        out_labels.append(tgt[idx])
    return torch.stack(out_points, dim=0), torch.stack(out_labels, dim=0)


def _iter_shape_arrays(pred: Any, target: Any):
    if isinstance(target, np.ndarray) and target.dtype != object and target.ndim >= 2:
        for i in range(target.shape[0]):
            yield np.asarray(pred[i]), np.asarray(target[i])
    else:
        for pr, gt in zip(pred, target):
            yield np.asarray(pr), np.asarray(gt)


def shape_iou_metrics(pred: Any, target: Any) -> dict[str, float]:
    shape_ious = {cat: [] for cat in SEG_CLASSES}
    for pr, gt in _iter_shape_arrays(pred, target):
        cat = SEG_LABEL_TO_CAT[int(gt[0])]
        part_ious = []
        for part in SEG_CLASSES[cat]:
            gt_mask = gt == part
            pr_mask = pr == part
            union = np.logical_or(gt_mask, pr_mask).sum()
            if union == 0:
                part_ious.append(1.0)
            else:
                part_ious.append(float(np.logical_and(gt_mask, pr_mask).sum() / union))
        shape_ious[cat].append(float(np.mean(part_ious)))

    class_vals = [float(np.mean(v)) for v in shape_ious.values() if v]
    inst_vals = [x for vals in shape_ious.values() for x in vals]
    return {
        "class_avg_miou": float(np.mean(class_vals) * 100.0) if class_vals else float("nan"),
        "instance_avg_miou": float(np.mean(inst_vals) * 100.0) if inst_vals else float("nan"),
    }


def one_hot(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.eye(num_classes, device=device)[labels.detach().cpu().numpy()]
