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
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from torch import optim
from torch.utils.data import DataLoader

from nepa3d.data.cls_patch_dataset import (
    PatchClsArrayDataset,
    PatchClsPointDataset,
    PointAugConfig,
    load_scanobjectnn_h5_arrays,
)
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
    p.add_argument("--cache_root", type=str, default="", help="Cache root (preprocessed npz tree).")
    p.add_argument(
        "--data_format",
        type=str,
        default="npz",
        choices=["npz", "scan_h5"],
        help="Input backend: npz cache or ScanObjectNN h5 direct.",
    )
    p.add_argument("--scan_h5_root", type=str, default="", help="ScanObjectNN h5 root for --data_format=scan_h5.")
    p.add_argument(
        "--scan_variant",
        type=str,
        default="auto",
        choices=["auto", "obj_bg", "obj_only", "pb_t50_rs"],
        help="Variant selector for scan_h5 route.",
    )
    p.add_argument("--split_train", type=str, default="train")
    p.add_argument("--split_test", type=str, default="test")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_seed", type=int, default=0)
    p.add_argument(
        "--val_split_mode",
        type=str,
        default="file",
        choices=["file", "group_auto", "group_scanobjectnn", "pointmae"],
        help=(
            "Validation split mode from TRAIN. "
            "group_auto resolves to ScanObjectNN group split when cache_root contains 'scanobjectnn'. "
            "pointmae uses official train for training and official test for validation (Point-MAE legacy test-as-val). "
            "Default=file to match current Point-MAE strict policy (train->train/val split)."
        ),
    )

    p.add_argument("--n_point", type=int, default=1024)
    p.add_argument("--pt_sample_mode_train", type=str, default="random", choices=["random", "fps"])
    p.add_argument("--pt_sample_mode_eval", type=str, default="fps", choices=["random", "fps"])
    p.add_argument("--use_normals", type=int, default=0)

    # Aug
    p.add_argument("--aug_preset", type=str, default="pointmae", choices=["none", "default", "strong", "pointmae"])
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
    p.add_argument(
        "--patch_embed",
        type=str,
        default="fps_knn",
        choices=["fps_knn", "serial"],
        help="Patch grouping backend: fps_knn (default) or serial (Morton/chunk).",
    )
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument(
        "--serial_order",
        type=str,
        default="morton",
        choices=["morton", "morton_trans", "z", "z-trans", "random", "identity"],
    )
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla"])
    p.add_argument("--qk_norm", type=int, default=1, choices=[0, 1])
    p.add_argument("--qk_norm_affine", type=int, default=0, choices=[0, 1])
    p.add_argument("--qk_norm_bias", type=int, default=0, choices=[0, 1])
    p.add_argument("--layerscale_value", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=100.0, help="<=0 disables RoPE in nepa2d blocks.")
    p.add_argument("--rope_prefix_tokens", type=int, default=1)
    p.add_argument("--use_gated_mlp", type=int, default=0, choices=[0, 1], help="Enable gated MLP path.")
    p.add_argument("--hidden_act", type=str, default="gelu", choices=["gelu", "silu"], help="MLP activation.")
    p.add_argument("--pooling", type=str, default="cls_max", choices=["mean", "cls", "cls_max"])
    p.add_argument("--pos_mode", type=str, default="center_mlp", choices=["learned", "center_mlp"])
    p.add_argument("--head_mode", type=str, default="auto", choices=["auto", "linear", "pointmae_mlp"])
    p.add_argument("--head_hidden_dim", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.5)
    p.add_argument("--init_mode", type=str, default="default", choices=["default", "pointmae"])
    p.add_argument("--is_causal", type=int, default=0)

    # Optim
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument(
        "--batch_mode",
        type=str,
        default="global",
        choices=["global", "per_proc"],
        help=(
            "Batch interpretation. "
            "'global' keeps total batch fixed across DDP world-size (per-proc=batch/world_size). "
            "'per_proc' uses --batch as per-process batch."
        ),
    )
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=float, default=10.0)
    p.add_argument("--warmup_start_factor", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=10.0)
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


def _stratified_split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    labels = labels.astype(np.int64, copy=False)
    train_idx = []
    val_idx = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = idx.copy()
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(n * float(val_ratio))
        # Match modelnet40_index.stratified_train_val_split behavior.
        if n_val <= 0 and n >= 2:
            n_val = 1
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    train_idx = np.concatenate(train_idx, axis=0) if train_idx else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_idx, axis=0) if val_idx else np.zeros((0,), dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _resolve_scan_h5_files(scan_h5_root: str, scan_variant: str) -> Tuple[Path, Path]:
    root = Path(scan_h5_root)
    if not root.exists():
        raise FileNotFoundError(f"scan_h5_root not found: {scan_h5_root}")

    variant = scan_variant
    if variant == "auto":
        name_l = str(root).lower()
        if "nobg" in name_l or "obj_only" in name_l:
            variant = "obj_only"
        elif "pb_t50_rs" in name_l or "scale75" in name_l:
            variant = "pb_t50_rs"
        else:
            variant = "obj_bg"

    if variant == "pb_t50_rs":
        tr = root / "training_objectdataset_augmentedrot_scale75.h5"
        te = root / "test_objectdataset_augmentedrot_scale75.h5"
    else:
        tr = root / "training_objectdataset.h5"
        te = root / "test_objectdataset.h5"

    if not tr.exists() or not te.exists():
        raise FileNotFoundError(f"missing h5 files: train={tr} test={te}")
    return tr, te


@torch.no_grad()
def evaluate_local(
    model: PatchTransformerClassifier,
    loader: DataLoader,
    device: torch.device,
    use_normals: bool,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for batch in loader:
        xyz = batch["xyz"].to(device)
        label = batch["label"].to(device)
        normal = batch.get("normal", None)
        if use_normals and normal is not None:
            normal = normal.to(device)

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

    if args.data_format == "npz":
        if not args.cache_root:
            raise ValueError("--cache_root is required when --data_format=npz")
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
        train_paths_full = list_npz(cache_root, split=args.split_train)
        test_paths = list_npz(cache_root, split=args.split_test)

        # Build label map from the available class folders.
        label_map = build_label_map(train_paths_full + test_paths)
        num_classes = max(label_map.values()) + 1

        resolved_val_split_mode = str(args.val_split_mode)
        if args.val_split_mode == "pointmae":
            # Point-MAE style: train on official train split, select/check by official test split.
            train_paths = train_paths_full
            val_paths = test_paths
            resolved_val_split_mode = "pointmae(test-as-val)"
        else:
            val_group_key_fn = None
            if args.val_split_mode == "group_scanobjectnn":
                val_group_key_fn = scanobjectnn_group_key
            elif args.val_split_mode == "group_auto":
                if "scanobjectnn" in cache_root_abs_l:
                    val_group_key_fn = scanobjectnn_group_key
                    resolved_val_split_mode = "group_scanobjectnn(auto)"
                else:
                    resolved_val_split_mode = "file(auto-fallback)"

            train_paths, val_paths = stratified_train_val_split(
                train_paths_full,
                val_ratio=args.val_ratio,
                seed=args.val_seed,
                group_key_fn=val_group_key_fn,
            )
    else:
        if not args.scan_h5_root:
            raise ValueError("--scan_h5_root is required when --data_format=scan_h5")
        h5_train, h5_test = _resolve_scan_h5_files(args.scan_h5_root, args.scan_variant)
        tr_points, tr_labels = load_scanobjectnn_h5_arrays(str(h5_train))
        te_points, te_labels = load_scanobjectnn_h5_arrays(str(h5_test))
        if args.val_split_mode == "pointmae":
            tr_idx = np.arange(tr_labels.shape[0], dtype=np.int64)
            va_points, va_labels = te_points, te_labels
            resolved_val_split_mode = "pointmae(test-as-val)"
        else:
            tr_idx, va_idx = _stratified_split_indices(tr_labels, args.val_ratio, args.val_seed)
            va_points, va_labels = tr_points[va_idx], tr_labels[va_idx]
            resolved_val_split_mode = "stratified_label(h5)"
        num_classes = int(max(tr_labels.max(initial=0), te_labels.max(initial=0))) + 1

    # Aug preset
    if args.aug_preset == "none":
        aug_cfg = PointAugConfig(prob=0.0)
        aug_train = False
    elif args.aug_preset == "pointmae":
        # Point-MAE ScanObjectNN transform:
        # PointcloudScaleAndTranslate(scale=[2/3, 3/2], translate=0.2)
        # No jitter/rotation/dropout in this preset.
        aug_cfg = PointAugConfig(
            prob=1.0,
            scale_min=2.0 / 3.0,
            scale_max=3.0 / 2.0,
            shift_std=0.2,
            jitter_std=0.0,
            jitter_clip=0.0,
            rot_axis="y",
            rot_deg=0.0,
            dropout_ratio=0.0,
            dropout_prob=0.0,
        )
        aug_train = True
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
    if args.data_format == "npz":
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
            deterministic_eval_sampling=False,
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
            deterministic_eval_sampling=True,
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
            deterministic_eval_sampling=True,
        )
    else:
        train_set = PatchClsArrayDataset(
            tr_points[tr_idx],
            tr_labels[tr_idx],
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_train,
            aug=aug_train,
            aug_cfg=aug_cfg,
            rng_seed=args.seed,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=False,
        )
        val_set = PatchClsArrayDataset(
            va_points,
            va_labels,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            aug=False,
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 123,
            mc_eval_k=1,
            aug_eval=False,
            deterministic_eval_sampling=True,
        )
        test_set = PatchClsArrayDataset(
            te_points,
            te_labels,
            n_point=args.n_point,
            sample_mode=args.pt_sample_mode_eval,
            aug=False,
            aug_cfg=aug_cfg,
            rng_seed=args.seed + 456,
            mc_eval_k=max(1, args.mc_eval_k_test),
            aug_eval=bool(args.aug_eval),
            deterministic_eval_sampling=True,
        )

    world_size = max(1, int(accelerator.num_processes))
    eff_batch = int(args.batch)
    if args.batch_mode == "global" and world_size > 1:
        if eff_batch % world_size != 0:
            raise ValueError(
                f"--batch {eff_batch} must be divisible by world_size={world_size} when --batch_mode=global"
            )
        eff_batch = eff_batch // world_size

    train_loader = DataLoader(
        train_set,
        batch_size=eff_batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=eff_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=eff_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = PatchTransformerClassifier(
        num_classes=num_classes,
        patch_embed=args.patch_embed,
        num_groups=args.num_groups,
        group_size=args.group_size,
        use_normals=use_normals,
        center_mode=args.center_mode,
        serial_order=args.serial_order,
        serial_bits=args.serial_bits,
        serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        backbone_mode=args.backbone_mode,
        qk_norm=bool(args.qk_norm),
        qk_norm_affine=bool(args.qk_norm_affine),
        qk_norm_bias=bool(args.qk_norm_bias),
        layerscale_value=float(args.layerscale_value),
        rope_theta=float(args.rope_theta),
        rope_prefix_tokens=int(args.rope_prefix_tokens),
        use_gated_mlp=bool(args.use_gated_mlp),
        hidden_act=str(args.hidden_act),
        pooling=args.pooling,
        pos_mode=args.pos_mode,
        head_mode=args.head_mode,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        init_mode=args.init_mode,
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

    # Only train loader is distributed. Val/test are evaluated on main process only
    # to keep single-GPU and DDP metrics strictly comparable.
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    best_val = -1.0
    best_path = save_dir / "checkpoints" / "best.pt"

    if accelerator.is_main_process:
        pos_inject_note = (
            "per_layer_explicit" if args.backbone_mode == "vanilla" else "internal_rope_only"
        )
        print(
            f"PatchCls: classes={num_classes} train={len(train_set)} val={len(val_set)} test={len(test_set)}\n"
            f"  patch_embed={args.patch_embed} n_point={args.n_point} groups={args.num_groups} group_size={args.group_size} "
            f"serial_order={args.serial_order} serial_bits={args.serial_bits} serial_shuffle={int(args.serial_shuffle_within_patch)} "
            f"d_model={args.d_model} layers={args.n_layers} heads={args.n_heads} "
            f"backbone_mode={args.backbone_mode} qk_norm={int(args.qk_norm)} qk_norm_affine={int(args.qk_norm_affine)} "
            f"qk_norm_bias={int(args.qk_norm_bias)} layerscale_value={float(args.layerscale_value):g} "
            f"rope_theta={float(args.rope_theta):g} rope_prefix_tokens={int(args.rope_prefix_tokens)} "
            f"use_gated_mlp={int(args.use_gated_mlp)} hidden_act={str(args.hidden_act)} "
            f"pooling={args.pooling} pos_mode={args.pos_mode} pos_inject={pos_inject_note} head_mode={args.head_mode} "
            f"init_mode={args.init_mode} "
            f"is_causal={bool(args.is_causal)}\n"
            f"  world_size={world_size} batch_mode={args.batch_mode} batch_arg={args.batch} batch_effective={eff_batch}\n"
            f"  data_format={args.data_format} input_root={args.cache_root if args.data_format == 'npz' else args.scan_h5_root}\n"
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

        # val (main process only, full dataset)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # Rank-0-only eval must run on the unwrapped module. Running forward on
            # DDP-wrapped model only on rank0 causes collective mismatch.
            val_metrics = evaluate_local(
                accelerator.unwrap_model(model), val_loader, accelerator.device, use_normals=use_normals
            )
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[ep {epoch+1:03d}/{args.epochs}] lr={lr_now:.2e} "
                f"val_acc={val_metrics['acc']:.4f} val_loss={val_metrics['loss']:.4f}"
            )

            if val_metrics["acc"] > best_val:
                best_val = val_metrics["acc"]
                torch.save({"model": accelerator.unwrap_model(model).state_dict(), "args": vars(args)}, best_path)
                print(f"  saved best -> {best_path} (val_acc={best_val:.4f})")
        accelerator.wait_for_everyone()

    # Load best on all processes for a consistent final test.
    accelerator.wait_for_everyone()
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=True)

    accelerator.wait_for_everyone()
    test_metrics = None
    if accelerator.is_main_process:
        test_metrics = evaluate_local(
            accelerator.unwrap_model(model), test_loader, accelerator.device, use_normals=use_normals
        )
    if accelerator.is_main_process:
        print(f"TEST acc={test_metrics['acc']:.4f} loss={test_metrics['loss']:.4f}")
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if hasattr(accelerator, "end_training"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
