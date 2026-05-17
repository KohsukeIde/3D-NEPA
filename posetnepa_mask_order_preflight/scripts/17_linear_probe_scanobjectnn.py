#!/usr/bin/env python3
"""Run frozen-feature linear probes for PointGPT/pointNEPA checkpoints.

This probe is intentionally separate from the existing fine-tune runner. It
loads a pretrain checkpoint, extracts the classification feature from the
frozen PointTransformer, then trains only one linear layer on cached features.
That keeps the question narrow: does the pretext row produce linearly usable
features, or is full fine-tuning washing out the order/filtration difference?
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SCAN_FILES = {
    "objbg": (
        "h5_files/main_split/training_objectdataset.h5",
        "h5_files/main_split/test_objectdataset.h5",
    ),
    "objonly": (
        "h5_files/main_split_nobg/training_objectdataset.h5",
        "h5_files/main_split_nobg/test_objectdataset.h5",
    ),
    "hardest": (
        "h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5",
        "h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5",
    ),
}


@dataclass(frozen=True)
class ManifestSpec:
    path: Path
    run_tag: str
    label: str


class AttrDict(dict):
    """Tiny EasyDict-compatible fallback for YAML-loaded configs."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def to_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return AttrDict({k: to_attr_dict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [to_attr_dict(v) for v in value]
    return value


def parse_manifest_spec(raw: str, repo: Path) -> ManifestSpec:
    parts = raw.split("::")
    if len(parts) not in {2, 3}:
        raise SystemExit(
            "[error] --manifest-spec must be manifest.json::run_tag[::label]"
        )
    path = Path(parts[0])
    if not path.is_absolute():
        path = repo / path
    run_tag = parts[1]
    label = parts[2] if len(parts) == 3 else Path(parts[0]).parent.name
    return ManifestSpec(path=path, run_tag=run_tag, label=label)


def normalize_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_splits(raw: str) -> list[str]:
    aliases = {
        "obj_bg": "objbg",
        "objbg": "objbg",
        "obj_only": "objonly",
        "objonly": "objonly",
        "hardest": "hardest",
        "pb": "hardest",
        "pb_t50_rs": "hardest",
    }
    out: list[str] = []
    for item in normalize_csv(raw.lower()):
        if item not in aliases:
            raise SystemExit(f"[error] unsupported split: {item}")
        split = aliases[item]
        if split not in out:
            out.append(split)
    return out


def read_yaml(path: Path) -> AttrDict:
    import yaml

    with path.open() as f:
        return to_attr_dict(yaml.load(f, Loader=yaml.FullLoader))


def scanobjectnn_root(repo: Path) -> Path:
    root = repo / "data" / "ScanObjectNN"
    if root.exists():
        return root
    root = repo / "PointGPT" / "data" / "ScanObjectNN"
    return root


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        points = np.asarray(f["data"], dtype=np.float32)
        labels = np.asarray(f["label"], dtype=np.int64).reshape(-1)
    return points, labels


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def exp_root(pointgpt: Path, cfg_rel: str) -> Path:
    cfg = Path(cfg_rel)
    return pointgpt / "experiments" / cfg.stem / cfg.parent.name


def find_pretrain_exp_dir(
    pointgpt: Path,
    cfg_rel: str,
    pretrain_exp: str,
    run_tag: str,
) -> Path | None:
    root = exp_root(pointgpt, cfg_rel)
    if not root.exists():
        return None
    candidates: list[Path]
    if run_tag:
        exact = root / f"{pretrain_exp}_{run_tag}"
        candidates = [exact] if exact.exists() else []
        candidates += [p for p in root.glob(pretrain_exp + f"*{run_tag}*") if p.is_dir()]
    else:
        candidates = [p for p in root.glob(pretrain_exp + "*") if p.is_dir()]
    candidates = sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def build_model(pointgpt: Path, config_path: Path, ckpt_path: Path, device: torch.device) -> nn.Module:
    sys.path.insert(0, str(pointgpt))
    from models import build_model_from_cfg  # type: ignore

    cfg = read_yaml(config_path)
    model = build_model_from_cfg(cfg.model)
    model.load_model_from_ckpt(str(ckpt_path))
    model.to(device)
    model.eval()
    return model


def feature_cache_key(
    ckpt_path: Path,
    config_path: Path,
    split: str,
    subset: str,
    npoints: int,
    n_samples: int,
    deterministic_fps: bool,
) -> str:
    raw = "::".join(
        [
            str(ckpt_path.resolve()),
            str(config_path.resolve()),
            split,
            subset,
            str(npoints),
            str(n_samples),
            str(int(deterministic_fps)),
            str(ckpt_path.stat().st_mtime_ns if ckpt_path.exists() else 0),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def extract_features(
    *,
    model: nn.Module,
    points_np: np.ndarray,
    labels_np: np.ndarray,
    npoints: int,
    batch_size: int,
    device: torch.device,
    cache_path: Path | None,
    deterministic_fps: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        return cached["features"].float(), cached["labels"].long()

    from utils import misc  # type: ignore

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    dataset = TensorDataset(torch.from_numpy(points_np).float(), torch.from_numpy(labels_np).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    with torch.no_grad():
        for batch_points, batch_labels in loader:
            pts = batch_points.to(device, non_blocking=True)
            if deterministic_fps:
                pts = misc.fps(pts, npoints)
            elif pts.shape[1] != npoints:
                pts = pts[:, :npoints, :].contiguous()
            _, _, feat = model(pts, compute_recon=False, return_features=True)
            features.append(feat.detach().cpu().float())
            labels.append(batch_labels.detach().cpu().long())

    feats = torch.cat(features, dim=0)
    labs = torch.cat(labels, dim=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": feats, "labels": labs}, cache_path)
    return feats, labs


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean().item() * 100.0


def train_linear_probe(
    *,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    eval_every: int,
    device: torch.device,
) -> dict[str, float | int]:
    set_seed(seed)
    num_classes = int(max(train_y.max().item(), test_y.max().item()) + 1)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    linear = nn.Linear(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_ds = TensorDataset(train_x, train_y)
    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    train_x_dev = train_x.to(device)
    train_y_dev = train_y.to(device)
    test_x_dev = test_x.to(device)
    test_y_dev = test_y.to(device)
    best_test = -1.0
    best_epoch = 0
    last_test = -1.0
    last_train = -1.0
    for epoch in range(1, epochs + 1):
        linear.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(linear(xb), yb)
            loss.backward()
            opt.step()
        if epoch == epochs or epoch % eval_every == 0:
            linear.eval()
            with torch.no_grad():
                last_test = accuracy(linear(test_x_dev), test_y_dev)
                last_train = accuracy(linear(train_x_dev), train_y_dev)
            if last_test > best_test:
                best_test = last_test
                best_epoch = epoch

    return {
        "seed": seed,
        "epochs": epochs,
        "last_test_acc": last_test,
        "best_test_acc": best_test,
        "best_epoch": best_epoch,
        "train_acc": last_train,
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "source",
        "run_id",
        "order",
        "mask_ratio",
        "group_mode",
        "split",
        "seed",
        "last_test_acc",
        "best_test_acc",
        "best_epoch",
        "train_acc",
        "epochs",
        "feature_dim",
        "train_n",
        "test_n",
        "ckpt_path",
        "config_path",
        "error",
    ]
    extra = sorted({k for row in rows for k in row if k not in fields})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)

    with out_md.open("w") as f:
        f.write("# ScanObjectNN Frozen Linear Probe\n\n")
        f.write("Backbone is frozen. Features are extracted from `PointTransformer.forward(..., return_features=True)` and only one linear layer is trained on cached features.\n\n")
        f.write("| status | source | run | order | mask | split | seed | last test acc | best test acc | best epoch | train acc |\n")
        f.write("|---|---|---|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                "| "
                + " | ".join(
                    format_value(row.get(k))
                    for k in [
                        "status",
                        "source",
                        "run_id",
                        "order",
                        "mask_ratio",
                        "split",
                        "seed",
                        "last_test_acc",
                        "best_test_acc",
                        "best_epoch",
                        "train_acc",
                    ]
                )
                + " |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest-spec", action="append", required=True, help="manifest.json::run_tag[::label]")
    ap.add_argument("--splits", default="hardest")
    ap.add_argument("--include-orders", default="", help="Optional comma-separated allow-list.")
    ap.add_argument("--include-run-ids", default="", help="Optional comma-separated allow-list.")
    ap.add_argument("--npoints", type=int, default=2048)
    ap.add_argument("--feature-batch-size", type=int, default=64)
    ap.add_argument("--probe-batch-size", type=int, default=2048)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-train", type=int, default=0, help="Optional smoke-test limit; 0 means full train set.")
    ap.add_argument("--max-test", type=int, default=0, help="Optional smoke-test limit; 0 means full test set.")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-deterministic-fps", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = (repo / args.pointgpt_dir).resolve()
    specs = [parse_manifest_spec(x, repo) for x in args.manifest_spec]
    splits = normalize_splits(args.splits)
    include_orders = set(normalize_csv(args.include_orders))
    include_run_ids = set(normalize_csv(args.include_run_ids))
    seeds = [int(x) for x in normalize_csv(args.seeds)]
    cache_dir = (repo / args.cache_dir).resolve() if args.cache_dir else None
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    deterministic_fps = not args.no_deterministic_fps

    rows: list[dict[str, Any]] = []
    scan_root = scanobjectnn_root(repo)
    loaded_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split in splits:
        train_rel, test_rel = SCAN_FILES[split]
        train_points, train_labels = load_h5(scan_root / train_rel)
        test_points, test_labels = load_h5(scan_root / test_rel)
        if args.max_train > 0:
            train_points = train_points[: args.max_train]
            train_labels = train_labels[: args.max_train]
        if args.max_test > 0:
            test_points = test_points[: args.max_test]
            test_labels = test_labels[: args.max_test]
        loaded_splits[split] = (train_points, train_labels, test_points, test_labels)

    start_all = time.time()
    for spec in specs:
        manifest = json.loads(spec.path.read_text())
        for entry in manifest["runs"]:
            if include_orders and entry["order"] not in include_orders:
                continue
            if include_run_ids and entry["run_id"] not in include_run_ids:
                continue
            exp_dir = find_pretrain_exp_dir(pointgpt, entry["pretrain_config"], entry["pretrain_exp"], spec.run_tag)
            ckpt_path = exp_dir / "ckpt-last.pth" if exp_dir else None
            status_base = {
                "source": spec.label,
                "run_id": entry["run_id"],
                "order": entry["order"],
                "mask_ratio": entry["mask_ratio"],
                "group_mode": entry["group_mode"],
                "ckpt_path": str(ckpt_path) if ckpt_path else "",
                "config_path": "",
            }
            if ckpt_path is None or not ckpt_path.exists():
                for split in splits:
                    rows.append({**status_base, "status": "missing_ckpt", "split": split, "error": "ckpt-last.pth not found"})
                write_outputs(rows, repo / args.out_csv, repo / args.out_md)
                continue

            for split in splits:
                ft_cfg = entry["finetune_configs"][split]
                config_path = pointgpt / ft_cfg
                split_status = {**status_base, "split": split, "config_path": str(config_path)}
                if not config_path.exists():
                    rows.append({**split_status, "status": "missing_config", "error": "finetune config not found"})
                    write_outputs(rows, repo / args.out_csv, repo / args.out_md)
                    continue
                try:
                    model = build_model(pointgpt, config_path, ckpt_path, device)
                    train_points, train_labels, test_points, test_labels = loaded_splits[split]
                    feature_dim = None
                    cache_base = cache_dir / spec.label / entry["run_id"] / split if cache_dir else None
                    train_cache = None
                    test_cache = None
                    if cache_base is not None:
                        train_key = feature_cache_key(
                            ckpt_path,
                            config_path,
                            split,
                            "train",
                            args.npoints,
                            int(train_points.shape[0]),
                            deterministic_fps,
                        )
                        test_key = feature_cache_key(
                            ckpt_path,
                            config_path,
                            split,
                            "test",
                            args.npoints,
                            int(test_points.shape[0]),
                            deterministic_fps,
                        )
                        train_cache = cache_base / f"train_{train_key}.pt"
                        test_cache = cache_base / f"test_{test_key}.pt"
                    train_x, train_y = extract_features(
                        model=model,
                        points_np=train_points,
                        labels_np=train_labels,
                        npoints=args.npoints,
                        batch_size=args.feature_batch_size,
                        device=device,
                        cache_path=train_cache,
                        deterministic_fps=deterministic_fps,
                    )
                    test_x, test_y = extract_features(
                        model=model,
                        points_np=test_points,
                        labels_np=test_labels,
                        npoints=args.npoints,
                        batch_size=args.feature_batch_size,
                        device=device,
                        cache_path=test_cache,
                        deterministic_fps=deterministic_fps,
                    )
                    feature_dim = int(train_x.shape[1])
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    for seed in seeds:
                        metrics = train_linear_probe(
                            train_x=train_x,
                            train_y=train_y,
                            test_x=test_x,
                            test_y=test_y,
                            seed=seed,
                            epochs=args.probe_epochs,
                            batch_size=args.probe_batch_size,
                            lr=args.probe_lr,
                            weight_decay=args.probe_weight_decay,
                            eval_every=args.eval_every,
                            device=device,
                        )
                        rows.append(
                            {
                                **split_status,
                                "status": "ok",
                                **metrics,
                                "feature_dim": feature_dim,
                                "train_n": int(train_x.shape[0]),
                                "test_n": int(test_x.shape[0]),
                                "elapsed_sec_total": round(time.time() - start_all, 2),
                                "error": "",
                            }
                        )
                        write_outputs(rows, repo / args.out_csv, repo / args.out_md)
                except Exception as exc:  # Keep the chain moving for other rows.
                    rows.append({**split_status, "status": "error", "error": repr(exc)})
                    write_outputs(rows, repo / args.out_csv, repo / args.out_md)
                    if args.fail_fast:
                        raise

    write_outputs(rows, repo / args.out_csv, repo / args.out_md)
    print(f"[done] wrote {repo / args.out_csv}")
    print(f"[done] wrote {repo / args.out_md}")


if __name__ == "__main__":
    # Avoid HDF5/OpenMP surprises when this runs after long DDP jobs.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
