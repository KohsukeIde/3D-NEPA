#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.eval.common import load_model
from viewaction_nepa.models.simple_point_encoder import SimplePointEncoder


@dataclass
class ViewSamples:
    points: torch.Tensor
    view_id: np.ndarray
    category_name: list[str]
    shape_id: list[str]
    split: str
    requested_split: str
    num_shapes: int
    fallback_used: bool


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[error] CUDA requested but torch.cuda.is_available() is false")
    return device


def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    return obj


def resample_views(views: torch.Tensor, target_points: int, rng: np.random.Generator) -> torch.Tensor:
    if target_points <= 0 or views.shape[1] == target_points:
        return views
    num_views, num_points, _ = views.shape
    out = torch.empty((num_views, target_points, 3), dtype=views.dtype)
    for vi in range(num_views):
        if num_points > target_points:
            idx = rng.choice(num_points, size=target_points, replace=False)
            idx.sort()
        else:
            extra = rng.choice(num_points, size=target_points - num_points, replace=True)
            idx = np.concatenate([np.arange(num_points), extra])
        out[vi] = views[vi, torch.as_tensor(idx, dtype=torch.long)]
    return out


def collect_view_samples(
    cache_root: str | Path,
    split: str,
    max_shapes: int,
    max_samples: int,
    points_per_sample: int,
    seed: int,
) -> ViewSamples:
    rng = np.random.default_rng(seed)
    ds = ViewActionDataset(cache_root, split=split, mode="all_views", max_shapes=max_shapes, seed=seed)
    fallback_used = False
    actual_split = split
    if len(ds.files) == 0 and split not in {"all", "full"}:
        ds = ViewActionDataset(cache_root, split="all", mode="all_views", max_shapes=max_shapes, seed=seed)
        fallback_used = True
        actual_split = "all"
    if len(ds.files) == 0:
        raise SystemExit(f"[error] empty dataset for cache={cache_root} split={split}")

    point_batches: list[torch.Tensor] = []
    view_ids: list[np.ndarray] = []
    categories: list[str] = []
    shape_ids: list[str] = []
    target_points = points_per_sample
    for item in ds:
        views = item["views"].float()
        if target_points <= 0:
            target_points = int(views.shape[1])
        views = resample_views(views, target_points=target_points, rng=rng)
        num_views = int(views.shape[0])
        point_batches.append(views)
        view_ids.append(np.arange(num_views, dtype=np.int64))
        categories.extend([str(item["category"])] * num_views)
        shape_ids.extend([str(item["shape_id"])] * num_views)

    points = torch.cat(point_batches, dim=0).contiguous()
    view_id = np.concatenate(view_ids, axis=0)
    if max_samples and points.shape[0] > max_samples:
        keep = rng.choice(points.shape[0], size=max_samples, replace=False)
        keep.sort()
        points = points[torch.as_tensor(keep, dtype=torch.long)]
        view_id = view_id[keep]
        categories = [categories[int(i)] for i in keep]
        shape_ids = [shape_ids[int(i)] for i in keep]

    return ViewSamples(
        points=points,
        view_id=view_id,
        category_name=categories,
        shape_id=shape_ids,
        split=actual_split,
        requested_split=split,
        num_shapes=len(ds.files),
        fallback_used=fallback_used,
    )


def build_category_labels(train: ViewSamples, eval_samples: ViewSamples) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    cats = sorted(set(train.category_name))
    cat_to_idx = {cat: i for i, cat in enumerate(cats)}
    y_train = np.asarray([cat_to_idx[c] for c in train.category_name], dtype=np.int64)
    y_eval = np.asarray([cat_to_idx.get(c, -1) for c in eval_samples.category_name], dtype=np.int64)
    return y_train, y_eval, cat_to_idx


def geometric_descriptors(points: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            x = points[start:start + batch_size].float()
            mean = x.mean(dim=1)
            centered = x - mean[:, None, :]
            std = x.std(dim=1, unbiased=False)
            mn = x.amin(dim=1)
            mx = x.amax(dim=1)
            extent = mx - mn
            radial = torch.linalg.norm(centered, dim=-1)
            radial_stats = torch.stack(
                [
                    radial.mean(dim=1),
                    radial.std(dim=1, unbiased=False),
                    radial.amin(dim=1),
                    radial.amax(dim=1),
                    torch.quantile(radial, 0.25, dim=1),
                    torch.quantile(radial, 0.50, dim=1),
                    torch.quantile(radial, 0.75, dim=1),
                ],
                dim=1,
            )
            denom = max(int(x.shape[1]) - 1, 1)
            cov = centered.transpose(1, 2) @ centered / float(denom)
            eig = torch.linalg.eigvalsh(cov).sort(dim=1, descending=True).values
            eig_ratio = eig / eig.sum(dim=1, keepdim=True).clamp_min(1e-8)
            feats.append(torch.cat([mean, std, mn, mx, extent, radial_stats, eig, eig_ratio], dim=1).cpu())
    return torch.cat(feats, dim=0).numpy()


def encode_with_module(
    encoder: nn.Module,
    points: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    encoder.to(device).eval()
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            pts = points[start:start + batch_size].to(device, non_blocking=device.type == "cuda")
            feats.append(encoder(pts).detach().cpu())
    return torch.cat(feats, dim=0).numpy()


def encode_with_nepa(
    model: nn.Module,
    points: torch.Tensor,
    batch_size: int,
    device: torch.device,
    encoder_name: str,
) -> np.ndarray:
    model.to(device).eval()
    feats: list[torch.Tensor] = []
    encode = model.encode_online if encoder_name == "online" else model.encode_target
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            pts = points[start:start + batch_size].to(device, non_blocking=device.type == "cuda")
            feats.append(encode(pts).detach().cpu())
    return torch.cat(feats, dim=0).numpy()


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in np.unique(y_true):
        mask = y_true == cls
        if mask.any():
            recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    max_iter: int,
    seed: int,
) -> dict[str, Any]:
    train_mask = y_train >= 0
    eval_mask = y_eval >= 0
    y_tr = y_train[train_mask]
    y_ev = y_eval[eval_mask]
    classes = np.unique(y_tr)
    if len(classes) < 2:
        return {
            "status": "skipped",
            "reason": "fewer_than_two_train_classes",
            "n_train": int(train_mask.sum()),
            "n_eval": int(eval_mask.sum()),
            "num_train_classes": int(len(classes)),
        }
    if int(eval_mask.sum()) == 0:
        return {
            "status": "skipped",
            "reason": "no_eval_samples_with_train_label",
            "n_train": int(train_mask.sum()),
            "n_eval": 0,
            "num_train_classes": int(len(classes)),
        }
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed, class_weight="balanced"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_train[train_mask], y_tr)
    pred = clf.predict(x_eval[eval_mask])
    return {
        "status": "ok",
        "n_train": int(train_mask.sum()),
        "n_eval": int(eval_mask.sum()),
        "num_train_classes": int(len(classes)),
        "num_eval_classes": int(len(np.unique(y_ev))),
        "accuracy": float((pred == y_ev).mean()),
        "balanced_accuracy": balanced_accuracy(y_ev, pred),
        "chance": float(1.0 / max(len(classes), 1)),
    }


def evaluate_feature_source(
    name: str,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    train: ViewSamples,
    eval_samples: ViewSamples,
    train_category: np.ndarray,
    eval_category: np.ndarray,
    probe_max_iter: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "feature_dim": int(x_train.shape[1]),
        "view_id": fit_probe(
            x_train,
            train.view_id,
            x_eval,
            eval_samples.view_id,
            max_iter=probe_max_iter,
            seed=seed,
        ),
        "category": fit_probe(
            x_train,
            train_category,
            x_eval,
            eval_category,
            max_iter=probe_max_iter,
            seed=seed,
        ),
    }


class TinyPointNetClassifier(nn.Module):
    def __init__(self, num_views: int, num_categories: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
        )
        self.view_head = nn.Linear(256, num_views)
        self.category_head = nn.Linear(256, num_categories) if num_categories >= 2 else None

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.net(points.float()).max(dim=1).values
        cat_logits = self.category_head(h) if self.category_head is not None else None
        return self.view_head(h), cat_logits


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_train_classes: int) -> dict[str, Any]:
    mask = y_true >= 0
    if int(mask.sum()) == 0:
        return {
            "status": "skipped",
            "reason": "no_eval_samples_with_train_label",
            "n_eval": 0,
            "num_train_classes": int(num_train_classes),
        }
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "status": "ok",
        "n_eval": int(mask.sum()),
        "num_train_classes": int(num_train_classes),
        "num_eval_classes": int(len(np.unique(yt))),
        "accuracy": float((yp == yt).mean()),
        "balanced_accuracy": balanced_accuracy(yt, yp),
        "chance": float(1.0 / max(num_train_classes, 1)),
    }


def train_pointnet_classifier(
    train: ViewSamples,
    eval_samples: ViewSamples,
    train_category: np.ndarray,
    eval_category: np.ndarray,
    num_views: int,
    num_categories: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    category_weight: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if epochs <= 0:
        return {"status": "skipped", "reason": "epochs_le_0"}

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_ds = TensorDataset(
        train.points,
        torch.as_tensor(train.view_id, dtype=torch.long),
        torch.as_tensor(train_category, dtype=torch.long),
    )
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    model = TinyPointNetClassifier(num_views=num_views, num_categories=num_categories).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    last_loss = float("nan")
    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        for pts, view_y, cat_y in dl:
            pts = pts.to(device, non_blocking=device.type == "cuda")
            view_y = view_y.to(device, non_blocking=device.type == "cuda")
            cat_y = cat_y.to(device, non_blocking=device.type == "cuda")
            opt.zero_grad(set_to_none=True)
            view_logits, cat_logits = model(pts)
            loss = F.cross_entropy(view_logits, view_y)
            if cat_logits is not None and category_weight > 0:
                valid = cat_y >= 0
                if bool(valid.any()):
                    loss = loss + category_weight * F.cross_entropy(cat_logits[valid], cat_y[valid])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach().cpu()) * int(pts.shape[0])
            total += int(pts.shape[0])
        last_loss = total_loss / max(total, 1)

    model.eval()
    view_preds: list[torch.Tensor] = []
    cat_preds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, eval_samples.points.shape[0], batch_size):
            pts = eval_samples.points[start:start + batch_size].to(device, non_blocking=device.type == "cuda")
            view_logits, cat_logits = model(pts)
            view_preds.append(view_logits.argmax(dim=1).cpu())
            if cat_logits is not None:
                cat_preds.append(cat_logits.argmax(dim=1).cpu())
    view_pred = torch.cat(view_preds, dim=0).numpy()
    cat_pred = torch.cat(cat_preds, dim=0).numpy() if cat_preds else np.full_like(eval_category, -1)
    return {
        "status": "ok",
        "epochs": int(epochs),
        "category_weight": float(category_weight),
        "last_train_loss": float(last_loss),
        "view_id": classification_metrics(eval_samples.view_id, view_pred, num_train_classes=num_views),
        "category": (
            classification_metrics(eval_category, cat_pred, num_train_classes=num_categories)
            if num_categories >= 2 and category_weight > 0
            else {
                "status": "skipped",
                "reason": "category_aux_disabled" if num_categories >= 2 else "fewer_than_two_train_classes",
                "n_eval": int((eval_category >= 0).sum()),
                "num_train_classes": int(num_categories),
            }
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# View Discriminability",
        "",
        f"- cache: `{result['config']['cache_root']}`",
        f"- train/eval shapes: `{result['data']['train_shapes']}` / `{result['data']['eval_shapes']}`",
        f"- train/eval samples: `{result['data']['train_samples']}` / `{result['data']['eval_samples']}`",
        f"- num views: `{result['data']['num_views']}`",
        "",
        "| probe | task | acc | balanced_acc | chance | status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for name, probe in result["feature_probes"].items():
        for task in ["view_id", "category"]:
            row = probe[task]
            lines.append(
                f"| {name} | {task} | "
                f"{float(row.get('accuracy', float('nan'))):.4f} | "
                f"{float(row.get('balanced_accuracy', float('nan'))):.4f} | "
                f"{float(row.get('chance', float('nan'))):.4f} | "
                f"{row.get('status', 'unknown')} |"
            )
    pointnet = result["pointnet_classifier"]
    if pointnet.get("status") == "ok":
        for task in ["view_id", "category"]:
            row = pointnet[task]
            lines.append(
                f"| pointnet_classifier | {task} | "
                f"{float(row.get('accuracy', float('nan'))):.4f} | "
                f"{float(row.get('balanced_accuracy', float('nan'))):.4f} | "
                f"{float(row.get('chance', float('nan'))):.4f} | "
                f"{row.get('status', 'unknown')} |"
            )
    else:
        lines.append(f"| pointnet_classifier | view_id | nan | nan | nan | {pointnet.get('status', 'skipped')} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate view/category discriminability on a ViewAction multiview cache.")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--ckpt", default="", help="Optional ViewActionNEPA checkpoint for trained encoder probes.")
    ap.add_argument("--split", default="", help="Compatibility alias for --eval-split.")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--max-shapes", type=int, default=0,
                    help="Compatibility alias applied to train/eval max shapes when nonzero.")
    ap.add_argument("--max-train-shapes", type=int, default=200)
    ap.add_argument("--max-eval-shapes", type=int, default=200)
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--max-eval-samples", type=int, default=0)
    ap.add_argument("--points-per-sample", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latent-dim", type=int, default=384)
    ap.add_argument("--probe-max-iter", type=int, default=200)
    ap.add_argument("--pointnet-epochs", type=int, default=5)
    ap.add_argument("--pointnet-lr", type=float, default=1e-3)
    ap.add_argument("--pointnet-weight-decay", type=float, default=1e-4)
    ap.add_argument("--pointnet-category-weight", type=float, default=0.0)
    ap.add_argument("--ckpt-encoder", default="target", choices=["target", "online"])
    args = ap.parse_args()

    if args.split:
        args.eval_split = args.split
    if args.max_shapes:
        args.max_train_shapes = args.max_shapes
        args.max_eval_shapes = args.max_shapes

    if args.batch_size <= 0:
        raise SystemExit("[error] --batch-size must be positive")
    if args.points_per_sample < 0:
        raise SystemExit("[error] --points-per-sample must be non-negative")

    seed_all(args.seed)
    device = resolve_device(args.device)

    train = collect_view_samples(
        args.cache_root,
        split=args.train_split,
        max_shapes=args.max_train_shapes,
        max_samples=args.max_train_samples,
        points_per_sample=args.points_per_sample,
        seed=args.seed,
    )
    eval_samples = collect_view_samples(
        args.cache_root,
        split=args.eval_split,
        max_shapes=args.max_eval_shapes,
        max_samples=args.max_eval_samples,
        points_per_sample=args.points_per_sample,
        seed=args.seed + 17,
    )
    train_category, eval_category, cat_to_idx = build_category_labels(train, eval_samples)
    num_views = int(max(train.view_id.max(), eval_samples.view_id.max())) + 1

    feature_results: dict[str, Any] = {}
    raw_train = geometric_descriptors(train.points, batch_size=max(args.batch_size, 1))
    raw_eval = geometric_descriptors(eval_samples.points, batch_size=max(args.batch_size, 1))
    feature_results["raw_geometric"] = evaluate_feature_source(
        "raw_geometric",
        raw_train,
        raw_eval,
        train,
        eval_samples,
        train_category,
        eval_category,
        probe_max_iter=args.probe_max_iter,
        seed=args.seed,
    )

    random_encoder = SimplePointEncoder(args.latent_dim)
    random_train = encode_with_module(random_encoder, train.points, args.batch_size, device)
    random_eval = encode_with_module(random_encoder, eval_samples.points, args.batch_size, device)
    feature_results["random_simple_point_encoder"] = evaluate_feature_source(
        "random_simple_point_encoder",
        random_train,
        random_eval,
        train,
        eval_samples,
        train_category,
        eval_category,
        probe_max_iter=args.probe_max_iter,
        seed=args.seed,
    )

    if args.ckpt:
        model = load_model(args.ckpt, args.cache_root, device=str(device))
        ckpt_train = encode_with_nepa(model, train.points, args.batch_size, device, args.ckpt_encoder)
        ckpt_eval = encode_with_nepa(model, eval_samples.points, args.batch_size, device, args.ckpt_encoder)
        feature_results[f"ckpt_{args.ckpt_encoder}_encoder"] = evaluate_feature_source(
            f"ckpt_{args.ckpt_encoder}_encoder",
            ckpt_train,
            ckpt_eval,
            train,
            eval_samples,
            train_category,
            eval_category,
            probe_max_iter=args.probe_max_iter,
            seed=args.seed,
        )

    pointnet = train_pointnet_classifier(
        train,
        eval_samples,
        train_category,
        eval_category,
        num_views=num_views,
        num_categories=len(cat_to_idx),
        epochs=args.pointnet_epochs,
        batch_size=args.batch_size,
        lr=args.pointnet_lr,
        weight_decay=args.pointnet_weight_decay,
        category_weight=args.pointnet_category_weight,
        device=device,
        seed=args.seed,
    )

    result = {
        "config": {
            "cache_root": str(args.cache_root),
            "ckpt": str(args.ckpt) if args.ckpt else None,
            "ckpt_encoder": args.ckpt_encoder if args.ckpt else None,
            "device": str(device),
            "seed": int(args.seed),
            "latent_dim": int(args.latent_dim),
            "probe_max_iter": int(args.probe_max_iter),
            "pointnet_epochs": int(args.pointnet_epochs),
            "pointnet_category_weight": float(args.pointnet_category_weight),
            "batch_size": int(args.batch_size),
            "points_per_sample": int(args.points_per_sample),
        },
        "data": {
            "train_split": train.split,
            "train_requested_split": train.requested_split,
            "train_fallback_used": train.fallback_used,
            "eval_split": eval_samples.split,
            "eval_requested_split": eval_samples.requested_split,
            "eval_fallback_used": eval_samples.fallback_used,
            "train_shapes": int(train.num_shapes),
            "eval_shapes": int(eval_samples.num_shapes),
            "train_samples": int(train.points.shape[0]),
            "eval_samples": int(eval_samples.points.shape[0]),
            "num_views": int(num_views),
            "points_per_sample_actual": int(train.points.shape[1]),
            "train_categories": int(len(cat_to_idx)),
            "eval_categories_total": int(len(set(eval_samples.category_name))),
            "eval_category_samples_known_to_train": int((eval_category >= 0).sum()),
            "category_to_index": cat_to_idx,
        },
        "feature_probes": feature_results,
        "pointnet_classifier": pointnet,
    }

    text = json.dumps(as_jsonable(result), indent=2, sort_keys=True)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    if args.out_md:
        write_markdown(Path(args.out_md), result)
    print(text)


if __name__ == "__main__":
    main()
