import argparse
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader

from ..data.dataset import ModelNet40QueryDataset, collate
from ..data.modelnet40_index import (
    build_label_map,
    label_from_path,
    list_npz,
    scanobjectnn_group_key,
    stratified_train_val_split,
)
from ..models.query_nepa import QueryNepa
from ..token.tokenizer import (
    TYPE_BOS,
    TYPE_EOS,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_A_POINT,
    TYPE_A_RAY,
)
from ..utils.seed import set_seed
from ..utils.ckpt_utils import load_state_dict_flexible, maybe_resize_pos_emb_in_state_dict


def stratified_kshot(paths, k, seed=0):
    """Select up to K samples per class from a list of paths (.../<split>/<class>/<id>.npz)."""
    from collections import defaultdict
    from ..data.modelnet40_index import label_from_path

    paths = list(paths)
    if k <= 0:
        return sorted(paths)
    groups = defaultdict(list)
    for p in paths:
        groups[label_from_path(p)].append(p)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    out = []
    for cls, cls_paths in groups.items():
        cls_paths = sorted(cls_paths)
        rng.shuffle(cls_paths)
        out.extend(cls_paths[: min(len(cls_paths), int(k))])
    return sorted(out)


def stratified_nway(paths, n_way, seed=0):
    """Select N classes and keep all samples from those classes."""
    from collections import defaultdict

    paths = list(paths)
    if n_way <= 0:
        classes = sorted({label_from_path(p) for p in paths})
        return sorted(paths), classes

    groups = defaultdict(list)
    for p in paths:
        groups[label_from_path(p)].append(p)
    classes = sorted(groups.keys())
    if n_way > len(classes):
        raise ValueError(f"n_way={n_way} exceeds available classes={len(classes)}")

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    picked_idx = rng.choice(len(classes), size=int(n_way), replace=False)
    picked = sorted(classes[i] for i in picked_idx.tolist())
    picked_set = set(picked)
    out = [p for p in paths if label_from_path(p) in picked_set]
    return sorted(out), picked


class ClsWrapper(nn.Module):
    def __init__(
        self,
        backbone,
        n_classes,
        cls_is_causal=False,
        cls_pooling="auto",
        qa_tokens=False,
        use_fc_norm=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc_norm = nn.LayerNorm(backbone.d_model) if bool(use_fc_norm) else nn.Identity()
        self.head = nn.Linear(backbone.d_model, n_classes)
        self.cls_is_causal = bool(cls_is_causal)
        self.cls_pooling = str(cls_pooling)
        self.qa_tokens = bool(qa_tokens)

    def _resolved_pooling(self):
        if self.cls_pooling == "auto":
            return "mean_a" if self.qa_tokens else "eos"
        return self.cls_pooling

    def forward(self, feat, type_id):
        _, _, h = self.backbone(feat, type_id, is_causal=self.cls_is_causal)
        pool = self._resolved_pooling()

        def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            """Mean over token dim with boolean mask.

            x: (B,T,C), mask: (B,T)
            """
            if mask.dtype != torch.bool:
                mask = mask.bool()
            w = mask.float().unsqueeze(-1)
            denom = w.sum(dim=1).clamp(min=1.0)
            return (x * w).sum(dim=1) / denom

        if pool == "mean":
            pooled = h.mean(dim=1)
        elif pool == "bos":
            pooled = h[:, 0, :]
        elif pool == "mean_no_special":
            ns_mask = (type_id != TYPE_BOS) & (type_id != TYPE_EOS)
            pooled = _masked_mean(h, ns_mask) if ns_mask.any() else h[:, -1, :]
        elif pool == "mean_pts":
            pt_mask = (type_id == TYPE_POINT) | (type_id == TYPE_Q_POINT) | (type_id == TYPE_A_POINT)
            pooled = _masked_mean(h, pt_mask) if pt_mask.any() else h[:, -1, :]
        elif pool == "mean_q":
            q_mask = (type_id == TYPE_POINT) | (type_id == TYPE_Q_POINT)
            pooled = _masked_mean(h, q_mask) if q_mask.any() else h[:, -1, :]
        elif pool == "mean_a":
            a_mask = (type_id == TYPE_A_POINT) | (type_id == TYPE_A_RAY)
            pooled = _masked_mean(h, a_mask) if a_mask.any() else h[:, -1, :]
        else:
            # EOS/last-token pooling (legacy behavior).
            pooled = h[:, -1, :]
        pooled = self.fc_norm(pooled)
        return self.head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="data/modelnet40_cache")
    ap.add_argument(
        "--backend",
        type=str,
        default="mesh",
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="shorthand: sets both --train_backend and --eval_backend unless they are explicitly provided",
    )
    ap.add_argument(
        "--train_backend",
        type=str,
        default=None,
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="backend for TRAIN split (fine-tuning / probe training)",
    )
    ap.add_argument(
        "--eval_backend",
        type=str,
        default=None,
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
        help="backend for VAL/TEST split (evaluation). Use this for cross-backend probing.",
    )
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--save_dir", type=str, default="", help="optional: save best/last checkpoints here")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument(
        "--weight_decay_norm",
        type=float,
        default=0.0,
        help="Weight decay for norm/bias/1D params (usually 0.0).",
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="CrossEntropy label smoothing in [0, 1).",
    )
    ap.add_argument("--n_point", type=int, default=None)
    ap.add_argument("--n_ray", type=int, default=None)
    ap.add_argument(
        "--allow_scale_up",
        type=int,
        default=0,
        help=(
            "Allow n_point/n_ray to exceed the pretrain checkpoint settings (requires resizing pos_emb). "
            "Default keeps legacy behavior (cap to pretrain sizes)."
        ),
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=-1,
        help=(
            "Override model max_len (pos-emb length). If <0, auto = max(ckpt_max_len, required_seq_len). "
            "If set and differs from checkpoint, pos_emb is resized by 1D interpolation."
        ),
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_ratio", type=float, default=0.1, help="stratified val split ratio from TRAIN")
    ap.add_argument("--val_seed", type=int, default=0, help="seed for stratified train/val split")
    ap.add_argument(
        "--val_split_mode",
        type=str,
        default="file",
        choices=["file", "group_auto", "group_scanobjectnn"],
        help=(
            "Validation split mode from TRAIN. "
            "file: legacy file-level split; "
            "group_auto: group-aware split for ScanObjectNN caches only; "
            "group_scanobjectnn: always use ScanObjectNN group key."
        ),
    )
    ap.add_argument("--fewshot_k", type=int, default=0, help="K-shot fine-tuning: number of training samples per class (0=full train)")
    ap.add_argument("--fewshot_seed", type=int, default=0, help="seed for K-shot subset selection")
    ap.add_argument(
        "--fewshot_n_way",
        type=int,
        default=0,
        help="N-way episodic setting: keep only N classes for train/val/test (0=all classes)",
    )
    ap.add_argument(
        "--fewshot_way_seed",
        type=int,
        default=-1,
        help="seed for N-way class selection (-1: use --fewshot_seed)",
    )

    ap.add_argument("--eval_seed", type=int, default=0, help="deterministic eval seed (per-sample)")
    ap.add_argument("--mc_eval_k", type=int, default=1, help="MC eval: number of query resamples per test sample")
    ap.add_argument(
        "--mc_eval_k_val",
        type=int,
        default=-1,
        help="MC eval resamples for validation (-1: use --mc_eval_k)",
    )
    ap.add_argument(
        "--mc_eval_k_test",
        type=int,
        default=-1,
        help="MC eval resamples for final test (-1: use --mc_eval_k)",
    )
    ap.add_argument("--drop_ray_prob_train", type=float, default=0.0)
    ap.add_argument("--force_missing_ray", action="store_true")
    ap.add_argument("--add_eos", type=int, default=-1, help="1/0 to override, -1 to infer from checkpoint")
    ap.add_argument("--qa_tokens", type=int, default=-1, help="Use Q/A separated tokenization (0/1). -1: infer from checkpoint")
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="linear probe mode: freeze backbone, train classifier head only",
    )
    ap.add_argument(
        "--cls_is_causal",
        type=int,
        default=0,
        help="classification attention mode: 1=causal (AR-style), 0=bidirectional",
    )
    ap.add_argument(
        "--cls_pooling",
        type=str,
        default="mean_a",
        choices=["auto", "eos", "bos", "mean", "mean_no_special", "mean_pts", "mean_q", "mean_a"],
        help="classification pooling (default=mean_a): auto(mean_a for qa_tokens else eos), eos, bos, mean, mean_no_special, mean_pts, mean_q, mean_a",
    )
    ap.add_argument(
        "--use_fc_norm",
        type=int,
        default=0,
        choices=[0, 1],
        help="Apply LayerNorm on pooled feature before linear classifier head.",
    )
    ap.add_argument(
        "--ablate_point_dist",
        action="store_true",
        help="ablation: zero out point-distance channel before tokenization",
    )
    ap.add_argument(
        "--pt_xyz_key",
        type=str,
        default="pt_xyz_pool",
        help="npz key for point xyz pool (default: pt_xyz_pool; use pc_xyz to match surface-point protocols)",
    )
    ap.add_argument(
        "--pt_dist_key",
        type=str,
        default="pt_dist_pool",
        help="npz key for point dist pool (default: pt_dist_pool). If missing or length-mismatch, zeros are used.",
    )
    ap.add_argument(
        "--pt_sample_mode_train",
        type=str,
        default="random",
        choices=["random", "fps", "rfps", "grid", "fixed_grid"],
        help="Point sampling for TRAIN sequences.",
    )
    ap.add_argument(
        "--pt_sample_mode_eval",
        type=str,
        default="fps",
        choices=["random", "fps", "rfps", "grid", "fixed_grid"],
        help="Point sampling for EVAL sequences.",
    )
    ap.add_argument(
        "--point_order_mode",
        type=str,
        default="morton",
        choices=["morton", "fps", "random"],
        help="Point ordering after sampling: morton=space-filling sort, fps=keep sampled order, random=shuffle.",
    )
    ap.add_argument(
        "--pt_fps_key",
        type=str,
        default="pt_fps_order",
        help="npz key for precomputed FPS order.",
    )
    ap.add_argument(
        "--pt_rfps_m",
        type=int,
        default=4096,
        help="RFPS candidate count when fps key is unavailable.",
    )
    ap.add_argument(
        "--ddp_find_unused_parameters",
        type=int,
        default=1,
        help="DDP setting for finetune (1=enable find_unused_parameters, safer for partial-token pooling).",
    )
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch * world_size * grad_accum_steps).",
    )
    ap.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Global grad norm clipping (<=0 disables clipping).",
    )
    ap.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine"],
        help="Learning-rate schedule for fine-tuning.",
    )
    ap.add_argument(
        "--warmup_epochs",
        type=float,
        default=10.0,
        help="Warmup duration in epochs (used when --lr_scheduler cosine).",
    )
    ap.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Warmup start LR factor for LinearLR (used when --lr_scheduler cosine).",
    )
    ap.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum LR floor for cosine scheduler.",
    )
    ap.add_argument(
        "--llrd",
        type=float,
        default=1.0,
        help="Layer-wise learning-rate decay factor in (0,1]. 1.0 disables LLRD.",
    )
    ap.add_argument(
        "--llrd_mode",
        type=str,
        default="exp",
        choices=["exp", "linear"],
        help=(
            "LLRD schedule mode: "
            "exp uses factor^(depth), "
            "linear maps shallowest=args.llrd to deepest=1.0."
        ),
    )
    ap.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        help="Backbone drop-path (stochastic depth) rate during fine-tuning.",
    )

    # --- Data augmentation (classification protocol alignment) ---
    ap.add_argument(
        "--aug_preset",
        type=str,
        default="none",
        choices=[
            "none",
            "modelnet40",
            "modelnet40_legacy",
            "scanobjectnn",
            "scanobjectnn_rot_only",
        ],
        help=(
            "Augmentation preset (train/eval when --aug_eval): "
            "none | modelnet40(strong) | modelnet40_legacy | "
            "scanobjectnn(strong) | scanobjectnn_rot_only."
        ),
    )
    ap.add_argument("--aug_rotate_z", action="store_true", help="Train-time: random rotation around Z axis")
    ap.add_argument("--aug_scale_min", type=float, default=1.0, help="Train-time: random scale min")
    ap.add_argument("--aug_scale_max", type=float, default=1.0, help="Train-time: random scale max")
    ap.add_argument("--aug_translate", type=float, default=0.0, help="Train-time: random translation range (uniform in [-t,t])")
    ap.add_argument("--aug_jitter_sigma", type=float, default=0.0, help="Train-time: jitter sigma")
    ap.add_argument("--aug_jitter_clip", type=float, default=0.0, help="Train-time: jitter clip")
    ap.add_argument(
        "--aug_recompute_dist",
        type=int,
        default=1,
        choices=[0, 1],
        help=(
            "If 1, recompute point distance-to-surface after jitter augmentation "
            "for strict xyz/dist consistency (slower)."
        ),
    )
    ap.add_argument(
        "--aug_eval",
        action="store_true",
        help="Apply the same augmentation pipeline during eval (useful for vote-style TTA). Default off.",
    )

    args = ap.parse_args()
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(int(args.ddp_find_unused_parameters))
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    log = accelerator.print

    # Apply augmentation presets (override individual aug_* defaults).
    if args.aug_preset == "modelnet40":
        # Strong ModelNet40 protocol for regularization + TTA.
        args.aug_rotate_z = False
        args.aug_scale_min = 0.8
        args.aug_scale_max = 1.25
        args.aug_translate = 0.1
        args.aug_jitter_sigma = 0.01
        args.aug_jitter_clip = 0.05
    elif args.aug_preset == "modelnet40_legacy":
        args.aug_rotate_z = False
        args.aug_scale_min = 0.8
        args.aug_scale_max = 1.25
        args.aug_translate = 0.1
        args.aug_jitter_sigma = 0.0
        args.aug_jitter_clip = 0.0
    elif args.aug_preset == "scanobjectnn":
        # Strong ScanObjectNN protocol to mitigate overfitting on small train sets.
        args.aug_rotate_z = True
        args.aug_scale_min = 0.67
        args.aug_scale_max = 1.5
        args.aug_translate = 0.2
        args.aug_jitter_sigma = 0.01
        args.aug_jitter_clip = 0.05
    elif args.aug_preset == "scanobjectnn_rot_only":
        # Legacy preset kept for historical reproducibility.
        args.aug_rotate_z = True
        args.aug_scale_min = 1.0
        args.aug_scale_max = 1.0
        args.aug_translate = 0.0
        args.aug_jitter_sigma = 0.0
        args.aug_jitter_clip = 0.0

    set_seed(args.seed)
    device = accelerator.device

    train_backend = args.train_backend or args.backend
    eval_backend = args.eval_backend or args.backend
    mc_eval_k_val = args.mc_eval_k if args.mc_eval_k_val < 0 else int(args.mc_eval_k_val)
    mc_eval_k_test = args.mc_eval_k if args.mc_eval_k_test < 0 else int(args.mc_eval_k_test)
    if not (0.0 < float(args.llrd) <= 1.0):
        raise ValueError(f"--llrd must be in (0,1], got {args.llrd}")
    if not (0.0 <= float(args.drop_path) < 1.0):
        raise ValueError(f"--drop_path must be in [0,1), got {args.drop_path}")

    train_paths_full = list_npz(args.cache_root, "train")
    test_paths = list_npz(args.cache_root, "test")
    way_seed = args.fewshot_seed if args.fewshot_way_seed < 0 else args.fewshot_way_seed
    picked_classes = None
    if args.fewshot_n_way and args.fewshot_n_way > 0:
        train_paths_full, picked_classes = stratified_nway(
            train_paths_full,
            n_way=args.fewshot_n_way,
            seed=way_seed,
        )
        picked_set = set(picked_classes)
        test_paths = sorted([p for p in test_paths if label_from_path(p) in picked_set])
        log(
            f"[fewshot-nway] n_way={args.fewshot_n_way} way_seed={way_seed} "
            f"picked_classes={picked_classes}"
        )
        log(
            f"[fewshot-nway] filtered_train={len(train_paths_full)} "
            f"filtered_test={len(test_paths)}"
        )

    val_group_key_fn = None
    resolved_val_split_mode = str(args.val_split_mode)
    if args.val_split_mode == "group_scanobjectnn":
        val_group_key_fn = scanobjectnn_group_key
    elif args.val_split_mode == "group_auto":
        cache_root_l = os.path.abspath(args.cache_root).lower()
        if "scanobjectnn" in cache_root_l:
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
    if args.fewshot_k and args.fewshot_k > 0:
        train_paths = stratified_kshot(train_paths, k=args.fewshot_k, seed=args.fewshot_seed)
        log(f"[fewshot] k={args.fewshot_k} selected_train={len(train_paths)} (from split-train)")

    label_map = build_label_map(train_paths_full)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    pre_args = ckpt["args"]
    ckpt_n_types = ckpt["model"]["type_emb.weight"].shape[0]
    if args.add_eos < 0:
        add_eos = bool(pre_args.get("add_eos", ckpt_n_types >= 5))
    else:
        add_eos = bool(args.add_eos)

    if args.qa_tokens >= 0:
        qa_tokens = bool(args.qa_tokens)
    else:
        qa_tokens = bool(pre_args.get("qa_tokens", ckpt_n_types >= 9))

    # Safety guard: with qa_tokens=1 + point-dist ablation, A_POINT carries near-empty info.
    # Averaging Q/A point tokens together (mean_pts) can dilute useful signal.
    if qa_tokens and args.ablate_point_dist and args.cls_pooling == "mean_pts":
        log(
            "[warn] qa_tokens=1 + --ablate_point_dist + cls_pooling=mean_pts can dilute Q-point signal; "
            "override cls_pooling to mean_q"
        )
        args.cls_pooling = "mean_q"

    pre_n_point = int(pre_args["n_point"])
    pre_n_ray = int(pre_args["n_ray"])

    allow_scale_up = bool(int(args.allow_scale_up))

    if args.n_point is None:
        args.n_point = pre_n_point
    elif args.n_point > pre_n_point and (not allow_scale_up):
        log(
            f"[sizes] requested n_point={args.n_point} > pretrain n_point={pre_n_point}; capping (set --allow_scale_up 1 to override)"
        )
        args.n_point = pre_n_point
    elif args.n_point > pre_n_point and allow_scale_up:
        log(f"[sizes] scale-up enabled: n_point {pre_n_point} -> {args.n_point}")
    elif args.n_point <= 0:
        raise ValueError(f"--n_point must be positive, got {args.n_point}")

    if args.n_ray is None:
        args.n_ray = pre_n_ray
    elif args.n_ray > pre_n_ray and (not allow_scale_up):
        log(
            f"[sizes] requested n_ray={args.n_ray} > pretrain n_ray={pre_n_ray}; capping (set --allow_scale_up 1 to override)"
        )
        args.n_ray = pre_n_ray
    elif args.n_ray > pre_n_ray and allow_scale_up:
        log(f"[sizes] scale-up enabled: n_ray {pre_n_ray} -> {args.n_ray}")
    elif args.n_ray < 0:
        raise ValueError(f"--n_ray must be >= 0, got {args.n_ray}")

    train_ds = ModelNet40QueryDataset(
        train_paths,
        backend=train_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        drop_ray_prob=args.drop_ray_prob_train,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        pt_xyz_key=args.pt_xyz_key,
        pt_dist_key=args.pt_dist_key,
        ablate_point_dist=args.ablate_point_dist,
        pt_sample_mode=args.pt_sample_mode_train,
        pt_fps_key=args.pt_fps_key,
        pt_rfps_m=args.pt_rfps_m,
        aug_rotate_z=args.aug_rotate_z,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_translate=args.aug_translate,
        aug_jitter_sigma=args.aug_jitter_sigma,
        aug_jitter_clip=args.aug_jitter_clip,
        aug_recompute_dist=bool(int(args.aug_recompute_dist)),
        aug_eval=args.aug_eval,
        point_order_mode=args.point_order_mode,
        return_label=True,
        label_map=label_map,
    )

    val_ds = ModelNet40QueryDataset(
        val_paths,
        backend=eval_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        mode="eval",
        eval_seed=args.eval_seed,
        mc_eval_k=mc_eval_k_val,
        drop_ray_prob=0.0,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        pt_xyz_key=args.pt_xyz_key,
        pt_dist_key=args.pt_dist_key,
        ablate_point_dist=args.ablate_point_dist,
        pt_sample_mode=args.pt_sample_mode_eval,
        pt_fps_key=args.pt_fps_key,
        pt_rfps_m=args.pt_rfps_m,
        aug_rotate_z=args.aug_rotate_z,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_translate=args.aug_translate,
        aug_jitter_sigma=args.aug_jitter_sigma,
        aug_jitter_clip=args.aug_jitter_clip,
        aug_recompute_dist=bool(int(args.aug_recompute_dist)),
        aug_eval=args.aug_eval,
        point_order_mode=args.point_order_mode,
        return_label=True,
        label_map=label_map,
    )

    test_ds = ModelNet40QueryDataset(
        test_paths,
        backend=eval_backend,
        n_point=args.n_point,
        n_ray=args.n_ray,
        mode="eval",
        eval_seed=args.eval_seed,
        mc_eval_k=mc_eval_k_test,
        drop_ray_prob=0.0,
        force_missing_ray=args.force_missing_ray,
        add_eos=add_eos,
        qa_tokens=qa_tokens,
        voxel_grid=args.voxel_grid,
        voxel_dilate=args.voxel_dilate,
        voxel_max_steps=args.voxel_max_steps,
        pt_xyz_key=args.pt_xyz_key,
        pt_dist_key=args.pt_dist_key,
        ablate_point_dist=args.ablate_point_dist,
        pt_sample_mode=args.pt_sample_mode_eval,
        pt_fps_key=args.pt_fps_key,
        pt_rfps_m=args.pt_rfps_m,
        aug_rotate_z=args.aug_rotate_z,
        aug_scale_min=args.aug_scale_min,
        aug_scale_max=args.aug_scale_max,
        aug_translate=args.aug_translate,
        aug_jitter_sigma=args.aug_jitter_sigma,
        aug_jitter_clip=args.aug_jitter_clip,
        aug_recompute_dist=bool(int(args.aug_recompute_dist)),
        aug_eval=args.aug_eval,
        point_order_mode=args.point_order_mode,
        return_label=True,
        label_map=label_map,
    )

    def _worker_init_fn(worker_id: int):
        # Ensure numpy RNG differs across dataloader workers.
        import random

        base = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(base)
        random.seed(base)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        worker_init_fn=_worker_init_fn,
    )

    # Allow scale-up by resizing learned positional embeddings.
    if "pos_emb" not in ckpt["model"]:
        raise KeyError("checkpoint missing pos_emb; cannot infer/resize max_len")

    ckpt_max_len = int(ckpt["model"]["pos_emb"].shape[1])
    required_len = (
        1 + 2 * int(args.n_point) + 2 * int(args.n_ray) + (1 if add_eos else 0)
        if qa_tokens
        else 1 + int(args.n_point) + int(args.n_ray) + (1 if add_eos else 0)
    )

    if int(args.max_len) >= 0:
        model_max_len = int(args.max_len)
        if model_max_len < required_len:
            raise ValueError(
                f"--max_len too small: max_len={model_max_len} < required_seq_len={required_len} "
                f"(qa_tokens={qa_tokens}, add_eos={add_eos}, n_point={args.n_point}, n_ray={args.n_ray})."
            )
    else:
        # Auto: keep ckpt length unless we need to scale up.
        model_max_len = max(ckpt_max_len, required_len)

    state = ckpt["model"]
    if model_max_len != ckpt_max_len:
        log(f"[ckpt] resizing pos_emb: ckpt_len={ckpt_max_len} -> max_len={model_max_len}")
        state = maybe_resize_pos_emb_in_state_dict(dict(state), model_max_len)

    backbone = QueryNepa(
        feat_dim=15,
        d_model=pre_args["d_model"],
        n_types=ckpt_n_types,
        nhead=pre_args["heads"],
        num_layers=pre_args["layers"],
        drop_path=float(args.drop_path),
        max_len=model_max_len,
    )
    load_state_dict_flexible(backbone, state, strict=True)
    backbone.to(device)

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()

    model = ClsWrapper(
        backbone,
        n_classes=len(label_map),
        cls_is_causal=bool(int(args.cls_is_causal)),
        cls_pooling=args.cls_pooling,
        qa_tokens=qa_tokens,
        use_fc_norm=bool(int(args.use_fc_norm)),
    ).to(device)
    if args.freeze_backbone:
        # In linear-probe mode, only train head (and optional fc_norm).
        trainable_named_params = [
            (n, p)
            for n, p in model.named_parameters()
            if p.requires_grad and (n.startswith("head.") or n.startswith("fc_norm."))
        ]
    else:
        trainable_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    arch_name = str(pre_args.get("arch", "causal")).lower()
    n_layers = int(pre_args.get("layers", 0))
    use_llrd = (not args.freeze_backbone) and (float(args.llrd) < 0.999999)
    max_layer_idx = (2 * n_layers if arch_name == "encdec" else n_layers) + 1

    def _layer_idx_from_name(param_name: str) -> int:
        # Wrapper head is deepest: no decay in LR scale.
        if param_name.startswith("head.") or param_name.startswith("fc_norm."):
            return max_layer_idx
        m = re.search(r"\.(?:enc|encoder)\.layers\.(\d+)\.", param_name)
        if m:
            return int(m.group(1)) + 1
        m = re.search(r"\.decoder\.layers\.(\d+)\.", param_name)
        if m:
            return n_layers + int(m.group(1)) + 1
        # Token/type/pos embeddings and misc params are treated as shallowest.
        return 0

    grouped = {}
    llrd_scales = []
    for name, param in trainable_named_params:
        name_l = name.lower()
        is_no_decay = bool(param.ndim <= 1 or name_l.endswith(".bias") or ("norm" in name_l))
        wd = float(args.weight_decay_norm if is_no_decay else args.weight_decay)

        lr_scale = 1.0
        if use_llrd:
            layer_idx = _layer_idx_from_name(name)
            if args.llrd_mode == "linear":
                llrd_min = float(args.llrd)
                lr_scale = llrd_min + (1.0 - llrd_min) * (
                    float(layer_idx) / float(max(1, max_layer_idx))
                )
            else:
                lr_scale = float(args.llrd) ** float(max_layer_idx - layer_idx)
            llrd_scales.append(lr_scale)

        key = (round(lr_scale, 12), wd)
        if key not in grouped:
            grouped[key] = {
                "params": [],
                "weight_decay": wd,
                "lr": float(args.lr) * lr_scale,
            }
        grouped[key]["params"].append(param)

    opt_groups = list(grouped.values())
    if not opt_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

    opt = optim.AdamW(opt_groups, lr=args.lr)
    smoothing = float(args.label_smoothing)
    smoothing = min(max(smoothing, 0.0), 0.999)
    ce = nn.CrossEntropyLoss(label_smoothing=smoothing)
    model, opt, train_dl, val_dl, test_dl = accelerator.prepare(model, opt, train_dl, val_dl, test_dl)

    # Build scheduler on optimizer after accelerator.prepare so the wrapped optimizer is controlled.
    steps_per_epoch = max(1, math.ceil(len(train_dl) / args.grad_accum_steps))
    total_update_steps = max(1, int(args.epochs) * steps_per_epoch)
    warmup_steps = 0
    lr_scheduler = None
    if args.lr_scheduler == "cosine":
        warmup_steps = int(round(float(args.warmup_epochs) * steps_per_epoch))
        if total_update_steps > 1:
            warmup_steps = max(0, min(warmup_steps, total_update_steps - 1))
        else:
            warmup_steps = 0

        schedulers = []
        milestones = []
        if warmup_steps > 0:
            start_factor = float(args.warmup_start_factor)
            start_factor = max(1e-8, min(start_factor, 1.0))
            schedulers.append(
                optim.lr_scheduler.LinearLR(
                    opt,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                )
            )
            milestones.append(warmup_steps)

        cosine_steps = max(1, total_update_steps - warmup_steps)
        schedulers.append(
            optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=cosine_steps,
                eta_min=float(args.min_lr),
            )
        )
        if len(schedulers) == 1:
            lr_scheduler = schedulers[0]
        else:
            lr_scheduler = optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=schedulers,
                milestones=milestones,
            )

    def eval_acc(dl):
        model.eval()
        correct = torch.zeros((), device=device, dtype=torch.long)
        total = torch.zeros((), device=device, dtype=torch.long)
        with torch.no_grad():
            for batch in dl:
                feat = batch["feat"].to(device).float()
                type_id = batch["type_id"].to(device).long()
                y = batch["label"].to(device).long()

                # MC eval: if feat is (B,K,T,F), average logits over K resamples
                if feat.dim() == 4:
                    B, K, T, F = feat.shape
                    feat_ = feat.view(B * K, T, F)
                    type_ = type_id.view(B * K, T)
                    logits_ = model(feat_, type_)
                    logits = logits_.view(B, K, -1).mean(dim=1)
                else:
                    logits = model(feat, type_id)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum()
                total += y.numel()
        correct = accelerator.reduce(correct, reduction="sum")
        total = accelerator.reduce(total, reduction="sum")
        return float((correct.float() / total.clamp(min=1).float()).item())

    if args.save_dir:
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    best_val = -1.0
    best_ep = -1
    best_state = None

    model_for_log = accelerator.unwrap_model(model)
    log(
        f"train_backend={train_backend} eval_backend={eval_backend} "
        f"val_ratio={args.val_ratio} val_seed={args.val_seed} "
        f"val_split_mode={resolved_val_split_mode} "
        f"mc_eval_k_val={mc_eval_k_val} mc_eval_k_test={mc_eval_k_test} "
        f"freeze_backbone={bool(args.freeze_backbone)} "
        f"cls_is_causal={bool(int(args.cls_is_causal))} "
        f"cls_pooling={args.cls_pooling}->{model_for_log._resolved_pooling()} "
        f"pt_xyz_key={args.pt_xyz_key} pt_dist_key={args.pt_dist_key} "
        f"pt_sample_mode_train={args.pt_sample_mode_train} pt_sample_mode_eval={args.pt_sample_mode_eval} "
        f"point_order_mode={args.point_order_mode} "
        f"pt_fps_key={args.pt_fps_key} pt_rfps_m={args.pt_rfps_m} "
        f"aug_recompute_dist={bool(int(args.aug_recompute_dist))} "
        f"ablate_point_dist={bool(args.ablate_point_dist)} "
        f"n_point={args.n_point}/{pre_n_point} n_ray={args.n_ray}/{pre_n_ray} "
        f"model_max_len={model_max_len} world_size={accelerator.num_processes} "
        f"use_fc_norm={bool(int(args.use_fc_norm))} "
        f"label_smoothing={smoothing:.4f} "
        f"weight_decay={args.weight_decay} weight_decay_norm={args.weight_decay_norm} "
        f"grad_accum_steps={args.grad_accum_steps} max_grad_norm={args.max_grad_norm} "
        f"llrd={args.llrd:.4f} llrd_mode={args.llrd_mode} "
        f"drop_path={args.drop_path:.4f} "
        f"lr_scheduler={args.lr_scheduler} warmup_steps={warmup_steps} total_update_steps={total_update_steps} "
        f"min_lr={args.min_lr}"
    )
    log(f"num_train={len(train_paths)} num_val={len(val_paths)} num_test={len(test_paths)}")
    if use_llrd and len(llrd_scales) > 0:
        log(
            f"[llrd] enabled: mode={args.llrd_mode} factor={args.llrd:.4f} "
            f"min_scale={min(llrd_scales):.6f} max_scale={max(llrd_scales):.6f} "
            f"param_groups={len(opt_groups)}"
        )

    global_update_steps = 0
    for ep in range(args.epochs):
        if hasattr(train_dl, "set_epoch"):
            train_dl.set_epoch(ep)
        model.train()
        if args.freeze_backbone:
            # keep frozen backbone deterministic even in train loop
            accelerator.unwrap_model(model).backbone.eval()

        train_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        train_correct = torch.zeros((), device=device, dtype=torch.long)
        train_total = torch.zeros((), device=device, dtype=torch.long)
        train_steps = torch.zeros((), device=device, dtype=torch.long)
        train_update_steps = torch.zeros((), device=device, dtype=torch.long)

        opt.zero_grad(set_to_none=True)

        for batch in train_dl:
            feat = batch["feat"].to(device).float()
            type_id = batch["type_id"].to(device).long()
            y = batch["label"].to(device).long()

            with accelerator.accumulate(model):
                logits = model(feat, type_id)
                loss = ce(logits, y)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.max_grad_norm > 0:
                        accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                    opt.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    opt.zero_grad(set_to_none=True)
                    train_update_steps += 1
                    global_update_steps += 1

            train_loss_sum += loss.detach()
            train_steps += 1
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                train_correct += (pred == y).sum()
                train_total += y.numel()

        val_acc = eval_acc(val_dl)
        train_loss_sum = accelerator.reduce(train_loss_sum, reduction="sum")
        train_steps = accelerator.reduce(train_steps, reduction="sum")
        train_correct = accelerator.reduce(train_correct, reduction="sum")
        train_total = accelerator.reduce(train_total, reduction="sum")
        train_loss = float((train_loss_sum / train_steps.clamp(min=1).to(train_loss_sum.dtype)).item())
        train_acc = float((train_correct.float() / train_total.clamp(min=1).float()).item())
        cur_lr = float(opt.param_groups[0]["lr"])
        improved = val_acc > best_val + 1e-12
        if improved:
            best_val = val_acc
            best_ep = ep
            # Keep best weights.
            best_state = {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()}
            if args.save_dir and accelerator.is_main_process:
                torch.save(
                    {
                        "model": best_state,
                        "epoch": best_ep,
                        "val_acc": best_val,
                        "args": vars(args),
                        "pretrain_ckpt": args.ckpt,
                    },
                    os.path.join(args.save_dir, "best.pt"),
                )

        log(
            f"ep={ep} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} best_val={best_val:.4f} best_ep={best_ep} "
            f"lr={cur_lr:.6e} updates_ep={int(train_update_steps.item())} updates_total={global_update_steps}"
        )

    # Final test evaluation on the best VAL checkpoint.
    if best_state is not None:
        accelerator.unwrap_model(model).load_state_dict(best_state, strict=True)
    test_acc = eval_acc(test_dl)
    log(f"best_val={best_val:.4f} best_ep={best_ep} test_acc={test_acc:.4f}")

    if args.save_dir and accelerator.is_main_process:
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in accelerator.unwrap_model(model).state_dict().items()},
                "epoch": args.epochs - 1,
                "val_acc": best_val,
                "test_acc": test_acc,
                "args": vars(args),
                "pretrain_ckpt": args.ckpt,
            },
            os.path.join(args.save_dir, "last.pt"),
        )


if __name__ == "__main__":
    main()
