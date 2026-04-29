#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = [
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


CONDITIONS = [
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
    "xyz_zero",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Reader-facing multi-view 2D recognizability audit for ScanObjectNN")
    p.add_argument("--root", default="3D-NEPA/PointGPT/data/ScanObjectNN/h5_files/main_split")
    p.add_argument("--train_file", default="training_objectdataset.h5")
    p.add_argument("--test_file", default="test_objectdataset.h5")
    p.add_argument("--image_size", type=int, default=96)
    p.add_argument("--views", type=int, default=10, choices=[6, 10])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_noise_sigma", type=float, default=0.08)
    p.add_argument(
        "--train_conditions",
        default="clean",
        help="Comma-separated stress conditions used for training renders, or 'all'. Default: clean.",
    )
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)
    p.add_argument("--num_examples", type=int, default=15)
    p.add_argument("--save_individual_views", type=int, default=1)
    p.add_argument("--output_dir", default="3D-NEPA/results/multiview_2d/scanobjectnn_recognizability")
    return p.parse_args()


def parse_condition_list(spec: str) -> list[str]:
    if spec.strip().lower() == "all":
        return list(CONDITIONS)
    conditions = [item.strip() for item in spec.split(",") if item.strip()]
    if not conditions:
        raise ValueError("--train_conditions produced an empty condition list")
    unknown = [cond for cond in conditions if cond not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown train condition(s): {unknown}; valid={CONDITIONS}")
    return conditions


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        points = np.asarray(h5["data"], dtype=np.float32)
        labels = np.asarray(h5["label"], dtype=np.int64).reshape(-1)
    return points[:, :, :3], labels


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32, copy=True)
    pts -= pts.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(pts, axis=1).max())
    if scale > 1e-6:
        pts /= scale
    return pts


def resample_to_n(points: np.ndarray, keep_idx: np.ndarray, npoints: int, rng: np.random.RandomState) -> np.ndarray:
    kept = points[keep_idx]
    if kept.shape[0] >= npoints:
        idx = rng.choice(kept.shape[0], npoints, replace=False)
    else:
        idx = rng.choice(kept.shape[0], npoints, replace=True)
    return kept[idx]


def local_patch_indices(points: np.ndarray, ratio: float, rng: np.random.RandomState) -> np.ndarray:
    npoints = points.shape[0]
    patch_n = max(1, int(round(npoints * ratio)))
    anchor = int(rng.randint(0, npoints))
    dist = np.sum((points - points[anchor : anchor + 1]) ** 2, axis=1)
    return np.argsort(dist)[:patch_n]


def apply_stress(points: np.ndarray, condition: str, rng: np.random.RandomState, local_noise_sigma: float) -> np.ndarray:
    pts = points.astype(np.float32, copy=True)
    npoints = pts.shape[0]
    if condition == "clean":
        return pts
    if condition == "xyz_zero":
        return np.zeros_like(pts)

    for prefix in ("random_keep", "structured_keep", "local_jitter", "local_replace"):
        if not condition.startswith(prefix):
            continue
        ratio = int(condition[len(prefix) :]) / 100.0
        if prefix == "random_keep":
            keep_n = max(1, int(round(npoints * ratio)))
            keep_idx = rng.choice(npoints, keep_n, replace=False)
            return resample_to_n(pts, keep_idx, npoints, rng)
        if prefix == "structured_keep":
            # Match the 3D stress evaluator: remove a contiguous anchor-neighborhood
            # and keep a fixed budget of farthest points, then resample.
            keep_n = max(1, int(round(npoints * ratio)))
            anchor = int(rng.randint(0, npoints))
            dist = np.sum((pts - pts[anchor : anchor + 1]) ** 2, axis=1)
            keep_idx = np.argsort(dist)[-keep_n:]
            return resample_to_n(pts, keep_idx, npoints, rng)
        patch_idx = local_patch_indices(pts, ratio, rng)
        if prefix == "local_jitter":
            diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            sigma = max(1e-6, local_noise_sigma * diag)
            pts[patch_idx] += rng.normal(0.0, sigma, size=(patch_idx.size, 3)).astype(np.float32)
            return pts
        if prefix == "local_replace":
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            pts[patch_idx] = mins + rng.rand(patch_idx.size, 3).astype(np.float32) * (maxs - mins)
            return pts
    raise ValueError(f"unknown condition: {condition}")


def rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    rz = np.array(
        [
            [np.cos(az), -np.sin(az), 0.0],
            [np.sin(az), np.cos(az), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(el), -np.sin(el)],
            [0.0, np.sin(el), np.cos(el)],
        ],
        dtype=np.float32,
    )
    return rx @ rz


def view_matrices(num_views: int) -> list[np.ndarray]:
    if num_views == 6:
        return [rotation_matrix(a, 15.0) for a in (0, 60, 120, 180, 240, 300)]
    mats = [rotation_matrix(a, 18.0) for a in (0, 45, 90, 135, 180, 225, 270, 315)]
    mats.append(rotation_matrix(0, 90.0))
    mats.append(rotation_matrix(0, -90.0))
    return mats


def render_multiview(points: np.ndarray, image_size: int, num_views: int) -> np.ndarray:
    pts0 = normalize_points(points)
    s = image_size
    out = np.zeros((num_views, 2, s, s), dtype=np.uint8)
    for view_idx, rot in enumerate(view_matrices(num_views)):
        pts = pts0 @ rot.T
        xy = np.clip((pts[:, :2] + 1.05) / 2.10, 0.0, 1.0)
        depth = np.clip((pts[:, 2] + 1.05) / 2.10, 0.0, 1.0)
        x = np.minimum(s - 1, np.maximum(0, (xy[:, 0] * (s - 1)).astype(np.int64)))
        y = np.minimum(s - 1, np.maximum(0, ((1.0 - xy[:, 1]) * (s - 1)).astype(np.int64)))
        occ = np.zeros((s, s), dtype=np.uint8)
        dep = np.zeros((s, s), dtype=np.float32)
        occ[y, x] = 255
        np.maximum.at(dep, (y, x), depth)
        out[view_idx, 0] = occ
        out[view_idx, 1] = np.asarray(np.round(dep * 255.0), dtype=np.uint8)
    return out


def render_dataset(
    points: np.ndarray,
    condition: str,
    image_size: int,
    num_views: int,
    seed: int,
    local_noise_sigma: float,
    max_items: int = 0,
) -> np.ndarray:
    if max_items > 0:
        points = points[:max_items]
    x = np.empty((points.shape[0], num_views, 2, image_size, image_size), dtype=np.uint8)
    for idx in range(points.shape[0]):
        rng = np.random.RandomState(seed + idx * 9973)
        x[idx] = render_multiview(apply_stress(points[idx], condition, rng, local_noise_sigma), image_size, num_views)
    return x


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx]).float().div_(255.0)
        return x, int(self.y[idx])


class SharedViewCNN(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, views, channels, height, width = x.shape
        h = self.encoder(x.reshape(bsz * views, channels, height, width)).flatten(1)
        h = h.reshape(bsz, views, -1).amax(dim=1)
        return self.head(h)


def train_model(x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = SharedViewCNN(num_classes=len(CLASS_NAMES)).to(device)
    loader = DataLoader(
        ArrayDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        total = 0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * yb.numel()
            total += int(yb.numel())
            correct += int((logits.argmax(1) == yb).sum().item())
        sched.step()
        if epoch == 0 or epoch + 1 == args.epochs or (epoch + 1) % 10 == 0:
            print(f"[train] epoch={epoch + 1}/{args.epochs} loss={loss_sum/max(1,total):.4f} acc={correct/max(1,total):.4f}", flush=True)
    return model


def render_training_conditions(
    points: np.ndarray,
    labels: np.ndarray,
    conditions: list[str],
    image_size: int,
    num_views: int,
    seed: int,
    local_noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for cond_idx, cond in enumerate(conditions):
        print(f"[render] train condition {cond}", flush=True)
        xs.append(
            render_dataset(
                points,
                cond,
                image_size,
                num_views,
                seed + 20000 * cond_idx,
                local_noise_sigma,
            )
        )
        ys.append(labels)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def eval_model(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> dict:
    device = next(model.parameters()).device
    loader = DataLoader(
        ArrayDataset(x, y),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    total = 0
    correct = 0
    per_total = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    per_correct = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            yy = yb.numpy()
            total += int(yy.size)
            correct += int((pred == yy).sum())
            for cls in np.unique(yy):
                mask = yy == cls
                per_total[cls] += int(mask.sum())
                per_correct[cls] += int((pred[mask] == yy[mask]).sum())
    return {
        "acc": float(correct / max(1, total)),
        "per_class_acc": {
            name: float(per_correct[i] / per_total[i]) if per_total[i] > 0 else float("nan")
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def choose_class_diverse(labels: np.ndarray, n: int) -> list[int]:
    chosen = []
    seen = set()
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int in seen:
            continue
        seen.add(label_int)
        chosen.append(idx)
        if len(chosen) >= n:
            return chosen
    for idx in range(labels.shape[0]):
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= n:
            break
    return chosen


def save_visual_panels(
    points: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    image_size: int,
    num_views: int,
    seed: int,
    local_noise_sigma: float,
    num_examples: int,
    save_individual: bool,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    view_dir = output_dir / "views"
    if save_individual:
        view_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    label_w = 150
    header_h = 18
    cell = image_size
    chosen = choose_class_diverse(labels, num_examples)
    for idx in chosen:
        label = int(labels[idx])
        cls_name = CLASS_NAMES[label]
        panel = Image.new("L", (label_w + num_views * cell, header_h + len(CONDITIONS) * cell), 255)
        draw = ImageDraw.Draw(panel)
        for v in range(num_views):
            draw.text((label_w + v * cell + 3, 2), f"view{v:02d}", fill=0, font=font)
        for row, cond in enumerate(CONDITIONS):
            y0 = header_h + row * cell
            draw.text((2, y0 + 3), cond, fill=0, font=font)
            rng = np.random.RandomState(seed + idx * 9973)
            stressed = apply_stress(points[idx], cond, rng, local_noise_sigma)
            rendered = render_multiview(stressed, image_size, num_views)
            if save_individual:
                cond_dir = view_dir / f"example_{idx:04d}_{cls_name}" / cond
                cond_dir.mkdir(parents=True, exist_ok=True)
            for v in range(num_views):
                img = Image.fromarray(rendered[v, 1], mode="L")
                panel.paste(img, (label_w + v * cell, y0))
                if save_individual:
                    img.save(cond_dir / f"view_{v:02d}.png")
        panel.save(panel_dir / f"example_{idx:04d}_{cls_name}_all_conditions.png")


def write_summary(output_dir: Path, args: argparse.Namespace, rows: list[dict], train_file: Path, test_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "acc", "delta_vs_clean"])
        writer.writeheader()
        clean = next(row["acc"] for row in rows if row["condition"] == "clean")
        for row in rows:
            writer.writerow(
                {
                    "condition": row["condition"],
                    "acc": f"{row['acc']:.6f}",
                    "delta_vs_clean": f"{row['acc'] - clean:.6f}",
                }
            )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_file": str(train_file),
                "test_file": str(test_file),
                "image_size": args.image_size,
                "views": args.views,
                "epochs": args.epochs,
                "local_noise_sigma": args.local_noise_sigma,
                "train_conditions": parse_condition_list(args.train_conditions),
                "results": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    clean = next(row["acc"] for row in rows if row["condition"] == "clean")
    lines = [
        "# ScanObjectNN Multi-view 2D Recognizability Audit",
        "",
        "This audit renders each object as separate grayscale orthographic depth/occupancy views.",
        "It does not use real RGB and does not merge different views into pseudo-RGB.",
        "",
        f"- train file: `{train_file}`",
        f"- test file: `{test_file}`",
        f"- image size: `{args.image_size}`",
        f"- views per object: `{args.views}`",
        f"- epochs: `{args.epochs}`",
        f"- train conditions: `{', '.join(parse_condition_list(args.train_conditions))}`",
        "",
        "| condition | acc | delta vs clean |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['condition']}` | `{row['acc']:.4f}` | `{row['acc'] - clean:.4f}` |")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "- This is a rendered-view recognizability calibration, not a VLM or human study.",
        "- Rows with high 2D recognizability indicate that object-level visual evidence remains in orthographic projections.",
        "- Rows with low 2D recognizability should not be used to claim that a 3D model missed human-obvious evidence.",
        "- Reader-facing panels are saved under `panels/`; individual grayscale views are saved under `views/`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    train_file = Path(args.root) / args.train_file
    test_file = Path(args.root) / args.test_file
    output_dir = Path(args.output_dir)
    train_points, train_labels = load_h5(train_file)
    test_points, test_labels = load_h5(test_file)
    if args.max_train > 0:
        train_points = train_points[: args.max_train]
        train_labels = train_labels[: args.max_train]
    if args.max_test > 0:
        test_points = test_points[: args.max_test]
        test_labels = test_labels[: args.max_test]

    train_conditions = parse_condition_list(args.train_conditions)
    x_train, train_labels_aug = render_training_conditions(
        train_points,
        train_labels,
        train_conditions,
        args.image_size,
        args.views,
        args.seed,
        args.local_noise_sigma,
    )
    print(f"[render] train tensor {x_train.shape} {x_train.dtype}", flush=True)
    model = train_model(x_train, train_labels_aug, args)

    rows: list[dict] = []
    for cond_idx, cond in enumerate(CONDITIONS):
        print(f"[render/eval] {cond}", flush=True)
        x_test = render_dataset(
            test_points,
            cond,
            args.image_size,
            args.views,
            args.seed + 10000 * (cond_idx + 1),
            args.local_noise_sigma,
        )
        result = eval_model(model, x_test, test_labels, args.batch_size)
        rows.append({"condition": cond, **result})

    output_dir.mkdir(parents=True, exist_ok=True)
    save_visual_panels(
        test_points,
        test_labels,
        output_dir,
        args.image_size,
        args.views,
        args.seed,
        args.local_noise_sigma,
        args.num_examples,
        bool(args.save_individual_views),
    )
    write_summary(output_dir, args, rows, train_file, test_file)
    print(f"[done] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
