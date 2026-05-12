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
from torch.utils.data import DataLoader, TensorDataset


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


def parse_args():
    p = argparse.ArgumentParser("Multi-view 2D support calibration for ScanObjectNN")
    p.add_argument("--root", default="3D-NEPA/PointGPT/data/ScanObjectNN/h5_files/main_split")
    p.add_argument("--train_file", default="training_objectdataset.h5")
    p.add_argument("--test_file", default="test_objectdataset.h5")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_noise_sigma", type=float, default=0.08)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_test", type=int, default=0)
    p.add_argument("--output_dir", default="3D-NEPA/results/multiview_2d/scanobjectnn_objbg")
    p.add_argument("--save_examples", type=int, default=1)
    p.add_argument("--num_examples", type=int, default=12)
    return p.parse_args()


def load_h5(path: Path):
    with h5py.File(path, "r") as h5:
        points = np.asarray(h5["data"], dtype=np.float32)
        labels = np.asarray(h5["label"], dtype=np.int64).reshape(-1)
    return points, labels


def normalize_points(points: np.ndarray) -> np.ndarray:
    pts = points[:, :3].astype(np.float32, copy=True)
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts = pts / scale
    return pts


def resample_to_n(points: np.ndarray, keep_idx: np.ndarray, npoints: int, rng: np.random.RandomState):
    kept = points[keep_idx]
    if kept.shape[0] >= npoints:
        idx = rng.choice(kept.shape[0], npoints, replace=False)
    else:
        idx = rng.choice(kept.shape[0], npoints, replace=True)
    return kept[idx]


def local_patch_indices(points: np.ndarray, ratio: float, rng: np.random.RandomState):
    npoints = points.shape[0]
    patch_n = max(1, int(round(npoints * ratio)))
    anchor = int(rng.randint(0, npoints))
    dist = np.sum((points - points[anchor : anchor + 1]) ** 2, axis=1)
    return np.argsort(dist)[:patch_n]


def apply_stress(points: np.ndarray, condition: str, rng: np.random.RandomState, local_noise_sigma: float):
    pts = points.astype(np.float32, copy=True)
    npoints = pts.shape[0]
    if condition == "clean":
        return pts
    if condition == "xyz_zero":
        return np.zeros_like(pts)

    for prefix, mode in [
        ("random_keep", "random"),
        ("structured_keep", "structured"),
        ("local_jitter", "local_jitter"),
        ("local_replace", "local_replace"),
    ]:
        if condition.startswith(prefix):
            ratio = int(condition[len(prefix) :]) / 100.0
            if mode == "random":
                keep_n = max(1, int(round(npoints * ratio)))
                keep_idx = rng.choice(npoints, keep_n, replace=False)
                return resample_to_n(pts, keep_idx, npoints, rng)
            if mode == "structured":
                keep_n = max(1, int(round(npoints * ratio)))
                anchor = int(rng.randint(0, npoints))
                dist = np.sum((pts - pts[anchor : anchor + 1]) ** 2, axis=1)
                keep_idx = np.argsort(dist)[-keep_n:]
                return resample_to_n(pts, keep_idx, npoints, rng)
            patch_idx = local_patch_indices(pts, ratio, rng)
            if mode == "local_jitter":
                diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                sigma = max(1e-6, local_noise_sigma * diag)
                pts[patch_idx] = pts[patch_idx] + rng.normal(0.0, sigma, size=(patch_idx.size, 3)).astype(np.float32)
                return pts
            if mode == "local_replace":
                mins = pts.min(axis=0)
                maxs = pts.max(axis=0)
                pts[patch_idx] = mins + rng.rand(patch_idx.size, 3).astype(np.float32) * (maxs - mins)
                return pts
    raise ValueError(condition)


def render_multiview(points: np.ndarray, image_size: int) -> np.ndarray:
    pts = normalize_points(points)
    pts = np.clip((pts + 1.0) * 0.5, 0.0, 1.0)
    views = [
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (2, 0, 1),
        (1, 2, 0),
        (2, 1, 0),
    ]
    imgs = []
    s = image_size
    for ax0, ax1, depth_ax in views:
        x = np.minimum(s - 1, np.maximum(0, (pts[:, ax0] * (s - 1)).astype(np.int64)))
        y = np.minimum(s - 1, np.maximum(0, ((1.0 - pts[:, ax1]) * (s - 1)).astype(np.int64)))
        d = pts[:, depth_ax]
        img = np.zeros((s, s), dtype=np.float32)
        # Depth-aware occupancy: keep the nearest/brighter point per pixel.
        np.maximum.at(img, (y, x), 0.25 + 0.75 * d)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def render_dataset(points: np.ndarray, condition: str, image_size: int, seed: int, local_noise_sigma: float, max_items: int = 0):
    if max_items > 0:
        points = points[:max_items]
    out = np.empty((points.shape[0], 6, image_size, image_size), dtype=np.float32)
    for i in range(points.shape[0]):
        rng = np.random.RandomState(seed + i * 9973)
        stressed = apply_stress(points[i], condition, rng, local_noise_sigma)
        out[i] = render_multiview(stressed, image_size)
    return out


class MultiViewCNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.net(x).flatten(1)
        return self.head(h)


def train_model(x_train, y_train, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = MultiViewCNN(in_channels=x_train.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.int64)))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
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
            total_loss += float(loss.item()) * yb.numel()
            total += yb.numel()
            correct += int((logits.argmax(dim=1) == yb).sum().item())
        if epoch in {0, args.epochs - 1} or (epoch + 1) % 10 == 0:
            print(f"[train] epoch={epoch + 1}/{args.epochs} loss={total_loss/max(1,total):.4f} acc={correct/max(1,total):.4f}")
    return model


def eval_model(model, x, y, batch_size=128):
    device = next(model.parameters()).device
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    total = 0
    correct = 0
    per_total = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    per_correct = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            yy = yb.numpy()
            total += yy.size
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


def save_examples(points, labels, output_dir: Path, image_size: int, seed: int, local_noise_sigma: float, num_examples: int):
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        print(f"[warn] PIL unavailable, skip examples: {exc}")
        return
    conditions = ["clean", "random_keep20", "structured_keep20", "local_jitter20", "local_replace20"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prefer class-diverse examples over the first N rows, because ScanObjectNN
    # h5 files are class-ordered in this split.
    chosen = []
    seen = set()
    for idx, label in enumerate(labels):
        label_int = int(label)
        if label_int in seen:
            continue
        seen.add(label_int)
        chosen.append(idx)
        if len(chosen) >= num_examples:
            break
    if len(chosen) < min(num_examples, points.shape[0]):
        for idx in range(points.shape[0]):
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= min(num_examples, points.shape[0]):
                break
    for idx in chosen:
        tiles = []
        for cond in conditions:
            rng = np.random.RandomState(seed + idx * 9973)
            img = render_multiview(apply_stress(points[idx], cond, rng, local_noise_sigma), image_size)
            # Use three representative views as RGB for compact inspection.
            rgb = np.stack([img[0], img[2], img[4]], axis=-1)
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            pil = Image.fromarray(rgb).resize((128, 128))
            draw = ImageDraw.Draw(pil)
            draw.text((4, 4), cond, fill=(255, 255, 255))
            tiles.append(pil)
        canvas = Image.new("RGB", (128 * len(tiles), 128), (0, 0, 0))
        for j, tile in enumerate(tiles):
            canvas.paste(tile, (128 * j, 0))
        label = CLASS_NAMES[int(labels[idx])]
        canvas.save(output_dir / f"example_{idx:03d}_{label}.png")


def to_md(summary):
    lines = [
        "# ScanObjectNN Multi-view 2D Support Calibration",
        "",
        f"- train file: `{summary['train_file']}`",
        f"- test file: `{summary['test_file']}`",
        f"- image size: `{summary['image_size']}`",
        f"- epochs: `{summary['epochs']}`",
        "",
        "| condition | acc | delta vs clean |",
        "|---|---:|---:|",
    ]
    clean = summary["conditions"][0]["acc"]
    for row in summary["conditions"]:
        lines.append(f"| `{row['condition']}` | `{row['acc']:.4f}` | `{row['acc'] - clean:.4f}` |")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "- This is a 2D rendered-view calibration model, not a human study.",
        "- Conditions that remain high here preserve object-level visual evidence in rendered views.",
        "- Conditions that fail here should not be used to claim that the 3D model missed human-obvious evidence.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_points, train_labels = load_h5(Path(args.root) / args.train_file)
    test_points, test_labels = load_h5(Path(args.root) / args.test_file)
    if args.max_train > 0:
        train_points = train_points[: args.max_train]
        train_labels = train_labels[: args.max_train]
    if args.max_test > 0:
        test_points = test_points[: args.max_test]
        test_labels = test_labels[: args.max_test]

    print("[render] train clean")
    x_train = render_dataset(train_points, "clean", args.image_size, args.seed, args.local_noise_sigma)
    model = train_model(x_train, train_labels, args)

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
        "xyz_zero",
    ]
    rows = []
    for offset, cond in enumerate(conditions):
        print(f"[render/eval] {cond}")
        x_test = render_dataset(test_points, cond, args.image_size, args.seed + 10000 * (offset + 1), args.local_noise_sigma)
        result = eval_model(model, x_test, test_labels, batch_size=max(args.batch_size, 128))
        result["condition"] = cond
        rows.append(result)

    if args.save_examples:
        save_examples(test_points, test_labels, out_dir / "examples", args.image_size, args.seed, args.local_noise_sigma, args.num_examples)

    summary = {
        "train_file": str(Path(args.root) / args.train_file),
        "test_file": str(Path(args.root) / args.test_file),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "conditions": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(to_md(summary))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "acc", "delta_vs_clean"])
        writer.writeheader()
        clean = rows[0]["acc"]
        for row in rows:
            writer.writerow({
                "condition": row["condition"],
                "acc": row["acc"],
                "delta_vs_clean": row["acc"] - clean,
            })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
