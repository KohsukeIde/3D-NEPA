#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_ops import normalize_point_cloud
from noise_nepa.eval.common import load_model
from noise_nepa.models.noise_nepa import NoiseNEPA


SCAN_FILES = {
    "obj_bg": (
        "h5_files/main_split/training_objectdataset.h5",
        "h5_files/main_split/test_objectdataset.h5",
    ),
    "obj_only": (
        "h5_files/main_split_nobg/training_objectdataset.h5",
        "h5_files/main_split_nobg/test_objectdataset.h5",
    ),
    "pb_t50_rs": (
        "h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5",
        "h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5",
    ),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        points = np.asarray(f["data"], dtype=np.float32)
        labels = np.asarray(f["label"], dtype=np.int64).reshape(-1)
    return points, labels


def stratified_subset(points: np.ndarray, labels: np.ndarray, max_n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_n <= 0 or max_n >= len(labels):
        return points, labels
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    chosen = []
    base = max_n // len(classes)
    rem = max_n % len(classes)
    class_order = classes.copy()
    rng.shuffle(class_order)
    for ci, cls in enumerate(class_order):
        inds = np.flatnonzero(labels == cls)
        rng.shuffle(inds)
        take = base + (1 if ci < rem else 0)
        if take > 0:
            chosen.extend(inds[: min(take, len(inds))].tolist())
    if len(chosen) < max_n:
        used = set(chosen)
        rest = np.array([i for i in range(len(labels)) if i not in used], dtype=np.int64)
        rng.shuffle(rest)
        chosen.extend(rest[: max_n - len(chosen)].tolist())
    chosen = np.asarray(chosen[:max_n], dtype=np.int64)
    rng.shuffle(chosen)
    return points[chosen], labels[chosen]


def sample_points(points: torch.Tensor, npoints: int, seed: int, sample_ids: torch.Tensor) -> torch.Tensor:
    rows = []
    for i in range(points.shape[0]):
        x = points[i]
        sample_seed = seed + int(sample_ids[i].item()) * 1_000_003
        g = torch.Generator(device="cpu").manual_seed(sample_seed)
        if x.shape[0] == npoints:
            out = x
        elif x.shape[0] > npoints:
            idx = torch.randperm(x.shape[0], generator=g)[:npoints]
            out = x[idx, :]
        else:
            idx = torch.randint(x.shape[0], (npoints,), generator=g)
            out = x[idx, :]
        rows.append(out)
    return normalize_point_cloud(torch.stack(rows, dim=0))


def corrupt(points: torch.Tensor, condition: str, seed: int) -> torch.Tensor:
    if condition == "clean":
        return points
    g = torch.Generator(device="cpu").manual_seed(seed)
    if condition == "jitter":
        noise = torch.randn(points.shape, generator=g, dtype=points.dtype) * 0.02
        return points + noise
    if condition.startswith("drop") or condition.startswith("random_keep"):
        raw = condition.replace("drop", "").replace("random_keep", "")
        keep = float(raw) / 100.0
        keep_n = max(8, int(points.shape[1] * keep))
        idx = torch.randperm(points.shape[1], generator=g)[:keep_n]
        sparse = points[:, idx, :]
        if keep_n < points.shape[1]:
            pad = torch.randint(keep_n, (points.shape[1] - keep_n,), generator=g)
            sparse = torch.cat([sparse, sparse[:, pad, :]], dim=1)
        return sparse
    if condition == "xyz_zero":
        return torch.zeros_like(points)
    raise ValueError(f"unknown condition: {condition}")


def extract_features(
    model: NoiseNEPA,
    points_np: np.ndarray,
    labels_np: np.ndarray,
    npoints: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    condition: str,
    feature_source: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.arange(points_np.shape[0], dtype=torch.long)
    ds = TensorDataset(torch.from_numpy(points_np).float(), torch.from_numpy(labels_np).long(), ids)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labs = [], []
    with torch.no_grad():
        for bi, (points, labels, sample_ids) in enumerate(loader):
            x = sample_points(points, npoints, seed, sample_ids)
            x = corrupt(x, condition, seed + 100_003 + bi).to(device)
            if feature_source == "online":
                z = model.encode_online(x).detach().cpu()
            elif feature_source == "target":
                z = model.encode_target(x).detach().cpu()
            else:
                raise ValueError(f"unknown feature_source: {feature_source}")
            feats.append(z.float())
            labs.append(labels.long())
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


def label_hist(labels: np.ndarray | torch.Tensor) -> dict[str, int]:
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    vals, counts = np.unique(labels, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals, counts)}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_by_condition: dict[str, tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_by_condition = {k: ((x - mean) / std, y) for k, (x, y) in test_by_condition.items()}
    num_classes = int(max([train_y.max().item()] + [y.max().item() for _, y in test_by_condition.values()]) + 1)
    head = nn.Linear(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(epochs):
        head.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
    head.eval()
    out = {}
    with torch.no_grad():
        train_logits = head(train_x.to(device))
        out["train_acc"] = float((train_logits.argmax(1).cpu() == train_y).float().mean())
        for name, (tx, ty) in test_by_condition.items():
            logits = head(tx.to(device))
            out[f"{name}_acc"] = float((logits.argmax(1).cpu() == ty).float().mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--random-encoder", action="store_true")
    ap.add_argument("--data-root", default="/groups/gag51402/datasets/scanobjectnn")
    ap.add_argument("--split", default="pb_t50_rs", choices=sorted(SCAN_FILES))
    ap.add_argument("--npoints", type=int, default=0)
    ap.add_argument("--feature-batch-size", type=int, default=128)
    ap.add_argument("--probe-batch-size", type=int, default=2048)
    ap.add_argument("--probe-epochs", type=int, default=100)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=0.01)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    ap.add_argument("--conditions", default="clean,jitter,drop50")
    ap.add_argument("--feature-source", default="target", choices=["online", "target"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    if not conditions:
        raise SystemExit("[error] --conditions must contain at least one condition, e.g. clean")
    for cond in conditions:
        if cond != "clean" and cond != "jitter" and cond != "xyz_zero" and not cond.startswith("drop") and not cond.startswith("random_keep"):
            raise SystemExit(f"[error] unsupported condition: {cond}")
    set_seed(args.seed)
    device_name = "cuda" if args.device == "auto" else args.device
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")

    if args.random_encoder:
        model = NoiseNEPA().to(device).eval()
        ckpt_args = {}
        ckpt_label = "random_encoder"
    else:
        if not args.ckpt:
            raise SystemExit("[error] set --ckpt or --random-encoder")
        model, ckpt_args = load_model(args.ckpt, device=device)
        ckpt_label = str(Path(args.ckpt).resolve())
    npoints = args.npoints or int(ckpt_args.get("npoints", 1024))

    train_rel, test_rel = SCAN_FILES[args.split]
    root = Path(args.data_root)
    train_path = root / train_rel
    test_path = root / test_rel
    train_points, train_labels = load_h5(root / train_rel)
    test_points, test_labels = load_h5(root / test_rel)
    source_train_shape = tuple(train_points.shape)
    source_test_shape = tuple(test_points.shape)
    source_train_hist = label_hist(train_labels)
    source_test_hist = label_hist(test_labels)
    train_points, train_labels = stratified_subset(train_points, train_labels, args.max_train, args.seed)
    test_points, test_labels = stratified_subset(test_points, test_labels, args.max_test, args.seed + 99_991)

    train_x, train_y = extract_features(
        model, train_points, train_labels, npoints, args.feature_batch_size, device, args.seed, "clean", args.feature_source
    )
    test_by_condition = {}
    for cond in conditions:
        test_by_condition[cond] = extract_features(
            model, test_points, test_labels, npoints, args.feature_batch_size, device, args.seed, cond, args.feature_source
        )
    metrics = train_probe(
        train_x,
        train_y,
        test_by_condition,
        args.probe_epochs,
        args.probe_batch_size,
        args.probe_lr,
        args.probe_weight_decay,
        args.seed,
        device,
    )
    out = {
        "ckpt": ckpt_label,
        "ckpt_sha256": "" if args.random_encoder else file_sha256(args.ckpt),
        "ckpt_args": ckpt_args,
        "split": args.split,
        "train_h5": str(train_path),
        "test_h5": str(test_path),
        "source_train_shape": source_train_shape,
        "source_test_shape": source_test_shape,
        "source_train_label_hist": source_train_hist,
        "source_test_label_hist": source_test_hist,
        "npoints": npoints,
        "source_npoints": int(source_train_shape[1]),
        "downsampled": bool(npoints != int(source_train_shape[1])),
        "feature_source": args.feature_source,
        "feature_dim": int(train_x.shape[1]),
        "train_n": int(train_x.shape[0]),
        "test_n": int(next(iter(test_by_condition.values()))[0].shape[0]),
        "train_label_hist": label_hist(train_y),
        "test_label_hist": label_hist(next(iter(test_by_condition.values()))[1]),
        "conditions": list(test_by_condition),
        "probe_epochs": args.probe_epochs,
        "probe_batch_size": args.probe_batch_size,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "max_train": args.max_train,
        "max_test": args.max_test,
        "seed": args.seed,
        **metrics,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
