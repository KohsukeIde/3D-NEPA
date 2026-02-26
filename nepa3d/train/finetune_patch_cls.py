"""Patchified Transformer classification training (scratch or finetune).

This script is intentionally *separate* from `finetune_cls.py` so we don't
accidentally break existing NEPA-token pipelines.

Primary purpose (Step 1 baseline):
- Train a patch-token transformer from random init ("Transformer (rand)" style)
- Verify ScanObjectNN / ModelNet40 reach typical ~80%+ ranges under matched settings

We reuse the cached datasets produced by the existing preprocess scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

from nepa3d.data.cls_patch_dataset import PatchClsPointDataset, PointAugConfig
from nepa3d.data.modelnet40_index import (
    build_label_map,
    list_npz,
    scanobjectnn_group_key,
    stratified_train_val_split,
)
from nepa3d.models.patch_classifier import PatchTransformerClassifier


def add_args(p: argparse.ArgumentParser) -> None:
    # IO
    p.add_argument("--ckpt", type=str, default="", help="Optional init checkpoint (can be empty for scratch).")
    p.add_argument("--save_dir", type=str, default="runs/patchcls", help="Directory to save outputs.")
    p.add_argument("--run_name", type=str, default="", help="Optional run name subfolder.")

    # Dataset
    p.add_argument("--cache_root", type=str, required=True, help="Cache root (preprocessed npz tree).")
    p.add_argument("--split_train", type=str, default="train")
    p.add_argument("--split_test", type=str, default="test")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument(
        "--val_split_mode",
        type=str,
        default="group_auto",
        choices=["file", "group_auto", "group_scanobjectnn"],
        help=(
            "Validation split mode from TRAIN. "
            "group_auto resolves to ScanObjectNN group split when cache_root contains 'scanobjectnn'."
        ),
    )

    p.add_argument("--n_point", type=int, default=1024)
    p.add_argument("--pt_sample_mode_train", type=str, default="random", choices=["random", "fps"])
    p.add_argument("--pt_sample_mode_eval", type=str, default="fps", choices=["random", "fps"])
    p.add_argument("--use_normals", type=int, default=0)

    # Aug
    p.add_argument("--aug_preset", type=str, default="default", choices=["none", "default", "strong"])
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--aug_scale_min", type=float, default=0.9)
    p.add_argument("--aug_scale_max", type=float, default=1.1)
    p.add_argument("--aug_shift_std", type=float, default=0.02)
    p.add_argument("--aug_jitter_std", type=float, default=0.005)
    p.add_argument("--aug_jitter_clip", type=float, default=0.02)
    p.add_argument("--aug_rot_axis", type=str, default="y", choices=["x", "y", "z"])
    p.add_argument("--aug_rot_deg", type=float, default=180.0)
    p.add_argument("--aug_dropout_ratio", type=float, default=0.0)
    p.add_argument("--aug_dropout_prob", type=float, default=0.0)
    p.add_argument("--aug_eval", type=int, default=0, help="Apply augmentation during eval (TTA).")
    p.add_argument("--mc_eval_k_test", type=int, default=1, help="MC crops at test time (K>1 averages logits).")

    # Model
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
    p.add_argument("--is_causal", type=int, default=0)

    # Optim
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=float, default=10.0)
    p.add_argument("--warmup_start_factor", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--allow_scan_uniscale_v2",
        type=int,
        default=0,
        choices=[0, 1],
        help="Safety guard: disallow scanobjectnn_*_v2 caches unless explicitly set to 1.",
    )


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: PatchTransformerClassifier,
    loader: DataLoader,
    accelerator: Accelerator,
    use_normals: bool,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        xyz = batch["xyz"].to(accelerator.device)
        label = batch["label"].to(accelerator.device)
        normal = batch.get("normal", None)
        if use_normals and normal is not None:
            normal = normal.to(accelerator.device)

        # MC eval: xyz is (B,K,N,3)
        if xyz.dim() == 4:
            B, K, N, C = xyz.shape
            xyz2 = xyz.reshape(B * K, N, C)
            if use_normals and normal is not None and normal.dim() == 4:
                normal2 = normal.reshape(B * K, N, 3)
            else:
                normal2 = None
            logits = model(xyz2, normal2)
            logits = logits.reshape(B, K, -1).mean(dim=1)
        else:
            logits = model(xyz, normal)

        loss = F.cross_entropy(logits, label)
        preds = logits.argmax(dim=-1)
        correct += (preds == label).sum().item()
        total += label.numel()
        loss_sum += loss.item() * label.size(0)

    # gather across processes
    correct = accelerator.gather_for_metrics(torch.tensor(correct, device=accelerator.device)).sum().item()
    total = accelerator.gather_for_metrics(torch.tensor(total, device=accelerator.device)).sum().item()
    loss_sum = accelerator.gather_for_metrics(torch.tensor(loss_sum, device=accelerator.device)).sum().item()

    acc = float(correct) / float(max(1, total))
    loss_avg = float(loss_sum) / float(max(1, total))
    return {"acc": acc, "loss": loss_avg}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    _set_seed(args.seed)
    accelerator = Accelerator()

    # Resolve save path
    save_dir = Path(args.save_dir)
    if args.run_name:
        save_dir = save_dir / args.run_name
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "checkpoints").mkdir(exist_ok=True)
        with open(save_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    # Dataset lists
    cache_root = args.cache_root
    cache_root_abs_l = os.path.abspath(cache_root).lower()
    if ("scanobjectnn_" in cache_root_abs_l) and ("_v2" in cache_root_abs_l) and (int(args.allow_scan_uniscale_v2) != 1):
        raise ValueError(
            f"cache_root={cache_root} is a uniscale v2 cache and is disallowed by policy. "
            "Use scanobjectnn_*_v3_nonorm, or set --allow_scan_uniscale_v2 1 for intentional legacy reruns."
        )
    if "scanobjectnn_main_split_v2" in cache_root_abs_l:
        raise ValueError(
            f"cache_root={cache_root} is disallowed for benchmark runs (main_split deprecated). "
            "Use variant cache roots: scanobjectnn_obj_bg_v3_nonorm | scanobjectnn_obj_only_v3_nonorm | scanobjectnn_pb_t50_rs_v3_nonorm."
        )
    train_paths = list_npz(cache_root, split=args.split_train)
    test_paths = list_npz(cache_root, split=args.split_test)

    # Build label map from the available class folders.
    label_map = build_label_map(train_paths + test_paths)
    num_classes = max(label_map.values()) + 1

    val_group_key_fn = None
    resolved_val_split_mode = str(args.val_split_mode)
    if args.val_split_mode == "group_scanobjectnn":
        val_group_key_fn = scanobjectnn_group_key
    elif args.val_split_mode == "group_auto":
        if "scanobjectnn" in cache_root_abs_l:
            val_group_key_fn = scanobjectnn_group_key
            resolved_val_split_mode = "group_scanobjectnn(auto)"
        else:
            resolved_val_split_mode = "file(auto-fallback)"

    train_paths, val_paths = stratified_train_val_split(
        train_paths,
        val_ratio=args.val_ratio,
        seed=args.val_seed,
        group_key_fn=val_group_key_fn,
    )

    # Aug preset
    if args.aug_preset == "none":
        aug_cfg = PointAugConfig(prob=0.0)
        aug_train = False
    else:
        # default / strong
        prob = args.aug_prob if args.aug_preset == "default" else max(0.8, args.aug_prob)
        aug_cfg = PointAugConfig(
            prob=prob,
            scale_min=args.aug_scale_min,
            scale_max=args.aug_scale_max,
            shift_std=args.aug_shift_std,
            jitter_std=args.aug_jitter_std,
            jitter_clip=args.aug_jitter_clip,
            rot_axis=args.aug_rot_axis,
            rot_deg=args.aug_rot_deg,
            dropout_ratio=args.aug_dropout_ratio,
            dropout_prob=args.aug_dropout_prob,
        )
        aug_train = True

    use_normals = bool(args.use_normals)
    train_set = PatchClsPointDataset(
        train_paths,
        cache_root=cache_root,
        label_map=label_map,
        n_point=args.n_point,
        sample_mode=args.pt_sample_mode_train,
        use_normals=use_normals,
        aug=aug_train,
        aug_cfg=aug_cfg,
        rng_seed=args.seed,
        mc_eval_k=1,
        aug_eval=False,
    )
    val_set = PatchClsPointDataset(
        val_paths,
        cache_root=cache_root,
        label_map=label_map,
        n_point=args.n_point,
        sample_mode=args.pt_sample_mode_eval,
        use_normals=use_normals,
        aug=False,
        aug_cfg=aug_cfg,
        rng_seed=args.seed + 123,
        mc_eval_k=1,
        aug_eval=False,
    )
    test_set = PatchClsPointDataset(
        test_paths,
        cache_root=cache_root,
        label_map=label_map,
        n_point=args.n_point,
        sample_mode=args.pt_sample_mode_eval,
        use_normals=use_normals,
        aug=False,
        aug_cfg=aug_cfg,
        rng_seed=args.seed + 456,
        mc_eval_k=max(1, args.mc_eval_k_test),
        aug_eval=bool(args.aug_eval),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = PatchTransformerClassifier(
        num_classes=num_classes,
        num_groups=args.num_groups,
        group_size=args.group_size,
        use_normals=use_normals,
        center_mode=args.center_mode,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        pooling=args.pooling,
        is_causal=bool(args.is_causal),
    )

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        # allow head mismatch
        missing, unexpected = model.load_state_dict(state, strict=False)
        if accelerator.is_main_process:
            print(f"Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        # warmup+cosine in epoch units
        warmup_epochs = max(0.0, float(args.warmup_epochs))
        if warmup_epochs > 0:
            schedulers = [
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=float(args.warmup_start_factor),
                    total_iters=int(round(warmup_epochs)),
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, args.epochs - int(round(warmup_epochs))),
                    eta_min=0.0,
                ),
            ]
            lr_scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=[int(round(warmup_epochs))],
            )
        else:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=0.0)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    best_val = -1.0
    best_path = save_dir / "checkpoints" / "best.pt"

    if accelerator.is_main_process:
        print(
            f"PatchCls: classes={num_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}\n"
            f"  n_point={args.n_point} groups={args.num_groups} group_size={args.group_size} "
            f"d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
            f"pooling={args.pooling} is_causal={bool(args.is_causal)}\n"
            f"  val_split_mode={resolved_val_split_mode}\n"
            f"  train_sample={args.pt_sample_mode_train} eval_sample={args.pt_sample_mode_eval} "
            f"aug_train={aug_train} aug_preset={args.aug_preset} aug_eval={bool(args.aug_eval)} mc_test={args.mc_eval_k_test}"
        )

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            xyz = batch["xyz"].to(accelerator.device)
            label = batch["label"].to(accelerator.device)
            normal = batch.get("normal", None)
            if use_normals and normal is not None:
                normal = normal.to(accelerator.device)

            logits = model(xyz, normal)
            loss = F.cross_entropy(logits, label)

            accelerator.backward(loss)
            if args.grad_clip and args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # val
        val_metrics = evaluate(model, val_loader, accelerator, use_normals=use_normals)
        if accelerator.is_main_process:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[ep {epoch+1:03d}/{args.epochs}] lr={lr_now:.2e} "
                f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}"
            )

        if val_metrics["acc"] > best_val and accelerator.is_main_process:
            best_val = val_metrics["acc"]
            torch.save({"model": accelerator.unwrap_model(model).state_dict(), "args": vars(args)}, best_path)
            print(f"  saved best -> {best_path} (val_acc={best_val:.4f})")

    # Load best on all processes for a consistent final test.
    accelerator.wait_for_everyone()
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    accelerator.wait_for_everyone()
    test_metrics = evaluate(model, test_loader, accelerator, use_normals=use_normals)
    if accelerator.is_main_process:
        print(f"TEST acc={test_metrics['acc']:.4f} loss={test_metrics['loss']:.4f}")
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
