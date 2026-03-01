"""Patch-token NEPA pretraining entrypoint."""

from __future__ import annotations

import argparse
import re
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random

from nepa3d.data.mixed_pretrain import build_mixed_pretrain
from nepa3d.models.patch_nepa import PatchTransformerNepa
from nepa3d.token.tokenizer import (
    TYPE_A_POINT,
    TYPE_A_RAY,
    TYPE_BOS,
    TYPE_EOS,
    TYPE_MISSING_RAY,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_Q_RAY,
    TYPE_RAY,
    TYPE_SEP,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_patch_nepa")

    # IO
    p.add_argument("--save_dir", type=str, default="runs_patch_nepa")
    p.add_argument("--run_name", type=str, default="debug")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--auto_resume", type=int, default=1, choices=[0, 1])
    p.add_argument("--resume_optimizer", type=int, default=1, choices=[0, 1])
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-pretrain")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=50)

    # Data
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--n_point", type=int, default=1024)
    p.add_argument("--n_ray", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pt_xyz_key", type=str, default="pt_xyz_pool")
    p.add_argument("--pt_dist_key", type=str, default="pt_dist_pool")
    p.add_argument("--ablate_point_dist", type=int, default=0, choices=[0, 1])
    p.add_argument("--pt_sample_mode", type=str, default="random", choices=["random", "fps", "rfps", "rfps_cached", "grid"])
    p.add_argument("--pt_fps_key", type=str, default="auto")
    p.add_argument("--pt_rfps_key", type=str, default="auto")
    p.add_argument("--pt_rfps_m", type=int, default=4096)
    p.add_argument("--point_order_mode", type=str, default="morton", choices=["morton", "fps", "random"])

    # Model (patching)
    p.add_argument("--patch_embed", type=str, default="fps_knn", choices=["serial", "pointgpt", "fps_knn"])
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--serial_order", type=str, default="morton",
                   choices=["morton", "morton_trans", "z", "z-trans", "random", "identity"])
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--morton_bits", type=int, default=10, help="Backward-compat alias; used when --serial_bits is omitted.")
    p.add_argument("--scale_morton", type=float, default=1.0, help="Compatibility arg (unused).")
    p.add_argument("--use_normals", type=int, default=0, choices=[0, 1])

    # Model (transformer)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
    p.add_argument("--qk_norm", type=int, default=1)
    p.add_argument("--qk_norm_affine", type=int, default=0)
    p.add_argument("--qk_norm_bias", type=int, default=0)
    p.add_argument("--layerscale_value", type=float, default=1e-5)
    p.add_argument("--rope_theta", type=float, default=100.0)
    p.add_argument("--use_gated_mlp", type=int, default=0)
    p.add_argument("--hidden_act", type=str, default="gelu")
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla"])

    # Positional embedding
    p.add_argument("--pos_mode", type=str, default="center_mlp", choices=["center_mlp", "none"])
    p.add_argument("--nepa2d_pos", type=int, default=1, choices=[0, 1])
    p.add_argument("--type_specific_pos", type=int, default=0, choices=[0, 1])
    p.add_argument("--type_pos_max_len", type=int, default=4096)

    # QueryNEPA-parity Q/A controls
    p.add_argument("--qa_tokens", type=int, default=1, choices=[0, 1])
    p.add_argument("--qa_layout", type=str, default="split_sep", choices=["interleave", "split", "split_sep"])
    p.add_argument("--qa_sep_token", type=int, default=1, choices=[0, 1])
    p.add_argument("--qa_fuse", type=str, default="add", choices=["add", "concat"])
    p.add_argument("--encdec_arch", type=int, default=0, choices=[0, 1])
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--use_pt_dist", type=int, default=1, choices=[0, 1])
    p.add_argument("--use_pt_grad", type=int, default=0, choices=[0, 1])
    p.add_argument("--answer_mlp_layers", type=int, default=2)
    p.add_argument("--answer_pool", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--nepa_skip_k", type=int, default=1)
    p.add_argument("--nepa_multi_k", type=str, default="")

    # Ray binding
    p.add_argument("--use_ray_patch", type=int, default=0)
    p.add_argument("--include_ray_unc", type=int, default=0, choices=[0, 1])
    p.add_argument("--ray_hit_threshold", type=float, default=0.5)
    p.add_argument("--ray_miss_t", type=float, default=4.0)
    p.add_argument("--ray_pool_k_max", type=int, default=32)
    p.add_argument("--ray_pool_mode", type=str, default="amax", choices=["mean", "max", "amax"])
    p.add_argument("--ray_fuse", type=str, default="add", choices=["add", "concat"])
    p.add_argument("--ray_assign_mode",
        type=str,
        default="proxy_sphere",
        choices=["proxy_sphere", "x_anchor", "independent_fps_knn"],
    )
    p.add_argument("--ray_use_origin", type=int, default=0, choices=[0, 1])
    p.add_argument("--ray_proxy_radius_scale", type=float, default=1.05)
    p.add_argument("--ray_num_groups", type=int, default=32)
    p.add_argument("--ray_group_size", type=int, default=32)

    # Train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=-1.0,
        help="Warmup epochs. If <0, resolve from warmup_ratio * epochs.",
    )
    p.add_argument("--warmup_ratio", type=float, default=0.025)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    # Data augmentation parity knobs (same semantics as Query-NEPA pretrain).
    p.add_argument("--aug_rotate_z", type=int, default=0, choices=[0, 1])
    p.add_argument("--aug_scale_min", type=float, default=1.0)
    p.add_argument("--aug_scale_max", type=float, default=1.0)
    p.add_argument("--aug_translate", type=float, default=0.0)
    p.add_argument("--aug_jitter_sigma", type=float, default=0.0)
    p.add_argument("--aug_jitter_clip", type=float, default=0.0)
    p.add_argument("--aug_recompute_dist", type=int, default=0, choices=[0, 1])
    # Dual-mask (QueryNEPA parity)
    p.add_argument("--dual_mask_near", type=float, default=0.0)
    p.add_argument("--dual_mask_far", type=float, default=0.0)
    p.add_argument("--dual_mask_window", type=int, default=32)
    p.add_argument("--dual_mask_type_aware", type=int, default=0, choices=[0, 1])
    p.add_argument("--dual_mask_warmup_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_ema", type=int, default=0, choices=[0, 1])
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--diag_copy", type=int, default=1, choices=[0, 1])
    p.add_argument("--diag_every", type=int, default=50)
    p.add_argument("--diag_k", type=int, default=1, help="diagnostic k shift; <=0 uses first skip_k")
    return p.parse_args()


def save_ckpt(
    save_path: Path,
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    step: int,
    args: argparse.Namespace,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "args": vars(args),
    }
    if ema_model is not None:
        payload["model_ema"] = ema_model.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, str(save_path))


def _init_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_model = ema_model.float()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


@torch.no_grad()
def _update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    if ema_model is None:
        return
    decay = float(min(max(decay, 0.0), 1.0))
    one_minus = 1.0 - decay
    msd = model.state_dict()
    for k, v_ema in ema_model.state_dict().items():
        v = msd.get(k, None)
        if v is None:
            continue
        v_det = v.detach()
        if torch.is_floating_point(v_ema):
            v_ema.mul_(decay).add_(v_det.to(device=v_ema.device, dtype=v_ema.dtype), alpha=one_minus)
        else:
            v_ema.copy_(v_det.to(device=v_ema.device))


def _resolved_warmup_epochs(args: argparse.Namespace) -> float:
    warmup_epochs = float(args.warmup_epochs)
    if warmup_epochs >= 0.0:
        return warmup_epochs
    warmup_ratio = max(0.0, float(args.warmup_ratio))
    if warmup_ratio <= 0.0:
        return 0.0
    return float(args.epochs) * warmup_ratio


def _scheduler_scale(epoch: int, args: argparse.Namespace) -> float:
    if int(args.epochs) <= 1:
        return 1.0
    warmup_epochs = _resolved_warmup_epochs(args)
    if epoch < warmup_epochs:
        return min(1.0, float(epoch + 1) / float(max(1e-8, warmup_epochs)))
    t = float(epoch - warmup_epochs) / float(max(1e-8, float(args.epochs) - warmup_epochs))
    t = min(max(t, 0.0), 1.0)
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * t))).item()
    min_scale = float(args.min_lr) / float(args.lr)
    return min_scale + (1.0 - min_scale) * cosine


def _parse_skip_k_list(args: argparse.Namespace) -> list[int]:
    raw_multi = str(getattr(args, "nepa_multi_k", "")).strip()
    ks: list[int] = []
    if raw_multi:
        for tok in re.split(r"[\s,;:+|]+", raw_multi):
            tok = tok.strip()
            if not tok:
                continue
            k = int(tok)
            if k < 1:
                raise ValueError(f"all nepa_multi_k values must be >=1, got {k}")
            ks.append(k)
    else:
        k = int(getattr(args, "nepa_skip_k", 1))
        if k < 1:
            raise ValueError(f"nepa_skip_k must be >=1, got {k}")
        ks.append(k)

    ks = sorted(set(int(k) for k in ks))
    if not ks:
        ks = [1]
    return ks


def _nepa_target_mask(type_id: torch.Tensor | None, k: int) -> torch.Tensor | None:
    if type_id is None:
        return None
    tgt_ty = type_id[:, k:]
    has_answer = bool((tgt_ty == int(TYPE_A_POINT)).any() or (tgt_ty == int(TYPE_A_RAY)).any())
    if has_answer:
        return (
            ((tgt_ty == int(TYPE_A_POINT)) | (tgt_ty == int(TYPE_A_RAY)))
            & (tgt_ty != int(TYPE_MISSING_RAY))
        )
    return (
        (tgt_ty != int(TYPE_BOS))
        & (tgt_ty != int(TYPE_SEP))
        & (tgt_ty != int(TYPE_EOS))
        & (tgt_ty != int(TYPE_MISSING_RAY))
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> float:
    if mask is None:
        return float(values.mean().item())
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean().item())


def _compute_copy_diag(
    z: torch.Tensor,
    z_hat: torch.Tensor,
    type_id: torch.Tensor | None,
    k: int,
) -> dict[str, float] | None:
    k = max(1, int(k))
    if z.size(1) <= k or z_hat.size(1) <= k:
        return None

    pred = z_hat[:, :-k, :].detach()
    tgt = z[:, k:, :].detach()
    prev = z[:, :-k, :].detach()

    mask = _nepa_target_mask(type_id, k)
    cos_tgt = F.cosine_similarity(pred, tgt, dim=-1)
    cos_prev = F.cosine_similarity(pred, prev, dim=-1)

    cos_tgt_m = _masked_mean(cos_tgt, mask)
    cos_prev_m = _masked_mean(cos_prev, mask)

    if mask is None:
        cmp = cos_prev >= cos_tgt
    else:
        if not bool(mask.any()):
            cmp = None
        else:
            cmp = (cos_prev >= cos_tgt)[mask]
    copy_win = float(cmp.float().mean().item()) if cmp is not None else float("nan")

    return {
        "k": float(k),
        "cos_tgt": cos_tgt_m,
        "cos_prev": cos_prev_m,
        "gap": float(cos_tgt_m - cos_prev_m),
        "copy_win": copy_win,
    }


def _fmt_diag(x: float) -> str:
    return "nan" if (x != x) else f"{x:.4f}"


def _init_wandb(args: argparse.Namespace, accelerator: Accelerator) -> Any:
    if int(args.use_wandb) != 1:
        return None
    if not accelerator.is_main_process:
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        accelerator.print(f"[wandb] disabled: import failed ({e})")
        return None

    tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    run_name = str(args.wandb_run_name).strip() or str(args.run_name)
    group = str(args.wandb_group).strip() or None
    entity = str(args.wandb_entity).strip() or None
    mode = str(args.wandb_mode).strip()

    cfg = dict(vars(args))
    cfg["save_root"] = str(Path(args.save_dir) / args.run_name)
    try:
        run = wandb.init(
            project=str(args.wandb_project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags if tags else None,
            mode=mode,
            config=cfg,
        )
        accelerator.print(
            f"[wandb] enabled project={args.wandb_project} run={run_name} "
            f"group={group or '-'} mode={mode}"
        )
        return run
    except Exception as e:
        accelerator.print(f"[wandb] disabled: init failed ({e})")
        return None


def main() -> None:
    args = parse_args()
    if bool(int(args.use_ray_patch)) and str(args.ray_assign_mode) == "independent_fps_knn":
        # Missing-ray handling in fixed-length independent ray patching relies on
        # type-aware masking to keep missing tokens out of effective K/V context.
        if int(args.dual_mask_type_aware) != 1:
            raise ValueError(
                "independent_fps_knn requires dual_mask_type_aware=1 "
                "(missing-ray safe mode)."
            )
    accelerator = Accelerator(gradient_accumulation_steps=int(args.grad_accum))
    set_seed(int(args.seed))
    mprint = accelerator.print

    serial_bits = int(args.serial_bits) if "serial_bits" in vars(args) else int(args.morton_bits)
    save_root = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    resume_path = str(args.resume).strip()
    if (not resume_path) and bool(int(args.auto_resume)):
        candidate = save_root / "ckpt_latest.pt"
        if candidate.is_file():
            resume_path = str(candidate)

    def _worker_init_fn(worker_id: int):
        # Keep numpy/python RNG streams diverged per worker for reproducibility.
        base = (torch.initial_seed() + int(worker_id)) % (2**32)
        np.random.seed(base)
        random.seed(base)

    dataset, sampler, _info = build_mixed_pretrain(
        mix_config_path=args.mix_config_path,
        n_point=int(args.n_point),
        n_ray=int(args.n_ray),
        num_workers=int(args.num_workers),
        mode="train",
        return_raw=True,
        pt_xyz_key=str(args.pt_xyz_key),
        pt_dist_key=str(args.pt_dist_key),
        ablate_point_dist=bool(args.ablate_point_dist),
        pt_sample_mode=str(args.pt_sample_mode),
        pt_fps_key=str(args.pt_fps_key),
        pt_rfps_key=str(args.pt_rfps_key),
        pt_rfps_m=int(args.pt_rfps_m),
        point_order_mode=str(args.point_order_mode),
        include_pt_grad=bool(int(args.use_pt_grad)),
        include_ray_unc=bool(int(args.include_ray_unc)),
        aug_rotate_z=bool(int(args.aug_rotate_z)),
        aug_scale_min=float(args.aug_scale_min),
        aug_scale_max=float(args.aug_scale_max),
        aug_translate=float(args.aug_translate),
        aug_jitter_sigma=float(args.aug_jitter_sigma),
        aug_jitter_clip=float(args.aug_jitter_clip),
        aug_recompute_dist=bool(int(args.aug_recompute_dist)),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
    )

    model = PatchTransformerNepa(
        patch_embed=args.patch_embed,
        n_point=int(args.n_point),
        group_size=int(args.group_size),
        num_groups=args.num_groups,
        serial_order=str(args.serial_order),
        serial_bits=serial_bits,
        serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
        use_normals=bool(args.use_normals),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path_rate=float(args.drop_path_rate),
        qk_norm=int(args.qk_norm),
        qk_norm_affine=int(args.qk_norm_affine),
        qk_norm_bias=int(args.qk_norm_bias),
        layerscale_value=float(args.layerscale_value),
        rope_theta=float(args.rope_theta),
        use_gated_mlp=int(args.use_gated_mlp),
        hidden_act=str(args.hidden_act),
        backbone_mode=str(args.backbone_mode),
        qa_tokens=int(args.qa_tokens),
        qa_layout=str(args.qa_layout),
        qa_sep_token=bool(int(args.qa_sep_token)),
        qa_fuse=str(args.qa_fuse),
        use_pt_dist=bool(int(args.use_pt_dist)),
        use_pt_grad=bool(int(args.use_pt_grad)),
        answer_mlp_layers=int(args.answer_mlp_layers),
        answer_pool=str(args.answer_pool),
        max_len=int(args.max_len),
        nepa2d_pos=bool(int(args.nepa2d_pos)),
        type_specific_pos=bool(int(args.type_specific_pos)),
        type_pos_max_len=int(args.type_pos_max_len),
        pos_mode=str(args.pos_mode),
        encdec_arch=bool(int(args.encdec_arch)),
        use_ray_patch=bool(args.use_ray_patch),
        include_ray_unc=bool(int(args.include_ray_unc)),
        ray_assign_mode=str(args.ray_assign_mode),
        use_ray_origin=bool(int(args.ray_use_origin)),
        ray_proxy_radius_scale=float(args.ray_proxy_radius_scale),
        ray_pool_mode=("amax" if str(args.ray_pool_mode) == "max" else str(args.ray_pool_mode)),
        ray_num_groups=int(args.ray_num_groups),
        ray_group_size=int(args.ray_group_size),
    )

    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    # Important: do not pass scheduler into accelerator.prepare().
    # Accelerate can step wrapped schedulers with optimizer.step(), which would
    # make this epoch-based schedule advance per-step and break LR behavior.
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    scheduler = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: _scheduler_scale(ep, args))

    model_raw = accelerator.unwrap_model(model)
    ema_model: torch.nn.Module | None = None
    if bool(int(args.use_ema)):
        ema_model = _init_ema_model(model_raw)

    warmup_epochs_resolved = _resolved_warmup_epochs(args)
    skip_k_list = _parse_skip_k_list(args)
    diag_enabled = bool(int(args.diag_copy))
    diag_every = max(1, int(args.diag_every))
    wandb_log_every = max(1, int(args.wandb_log_every))
    wandb_run = _init_wandb(args, accelerator)
    mprint(
        "[patch_nepa_pretrain] "
        f"num_processes={accelerator.num_processes} "
        f"mixed_precision={accelerator.mixed_precision} "
        f"patch_embed={args.patch_embed} "
        f"group_size={int(args.group_size)} num_groups={int(args.num_groups)} "
        f"n_point={int(args.n_point)} n_ray={int(args.n_ray)} use_ray_patch={int(bool(args.use_ray_patch))} "
        f"ray_assign={str(args.ray_assign_mode)} ray_pool={str(args.ray_pool_mode)} ray_groups={int(args.ray_num_groups)} ray_group_size={int(args.ray_group_size)} "
        f"ray_use_origin={int(args.ray_use_origin)} include_ray_unc={int(args.include_ray_unc)} "
        f"pt_sample_mode={str(args.pt_sample_mode)} pt_fps_key={str(args.pt_fps_key)} pt_rfps_m={int(args.pt_rfps_m)} "
        f"pt_rfps_key={str(args.pt_rfps_key)} "
        f"point_order_mode={str(args.point_order_mode)} "
        f"qa_tokens={int(args.qa_tokens)} qa_layout={str(args.qa_layout)} qa_sep={int(args.qa_sep_token)} "
        f"qa_fuse={str(args.qa_fuse)} use_pt_dist={int(args.use_pt_dist)} use_pt_grad={int(args.use_pt_grad)} skip_k={skip_k_list} "
        f"type_specific_pos={int(args.type_specific_pos)} "
        f"dual_mask=({float(args.dual_mask_near):.2f},{float(args.dual_mask_far):.2f},w={int(args.dual_mask_window)},type_aware={int(args.dual_mask_type_aware)}) "
        f"batch_per_proc={int(args.batch_size)} grad_accum={int(args.grad_accum)} "
        f"global_batch={int(args.batch_size) * int(accelerator.num_processes) * int(args.grad_accum)} "
        f"lr={float(args.lr):.3e} scheduler={str(args.lr_scheduler)} "
        f"use_ema={int(args.use_ema)} ema_decay={float(args.ema_decay):.6f} "
        f"warmup_epochs={warmup_epochs_resolved:.3f} warmup_ratio={float(args.warmup_ratio):.4f} "
        f"qk_norm={int(args.qk_norm)} qk_norm_affine={int(args.qk_norm_affine)} qk_norm_bias={int(args.qk_norm_bias)} "
        f"layerscale={float(args.layerscale_value):.2e} rope_theta={float(args.rope_theta):g} "
        f"aug_rotate_z={int(args.aug_rotate_z)} aug_scale=[{float(args.aug_scale_min):.3f},{float(args.aug_scale_max):.3f}] "
        f"aug_translate={float(args.aug_translate):.3f} aug_jitter_sigma={float(args.aug_jitter_sigma):.4f} "
        f"aug_jitter_clip={float(args.aug_jitter_clip):.4f} aug_recompute_dist={int(args.aug_recompute_dist)} "
        f"diag_copy={int(args.diag_copy)} diag_every={int(args.diag_every)} diag_k={int(args.diag_k)}"
    )

    start_epoch = 0
    global_step = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model_raw.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if bool(int(args.use_ema)) and ema_model is not None and isinstance(ckpt, dict):
            ema_state = ckpt.get("model_ema", None)
            if isinstance(ema_state, dict):
                ema_model.load_state_dict(ema_state, strict=False)
            else:
                ema_model.load_state_dict(model_raw.state_dict(), strict=False)
        if bool(int(args.resume_optimizer)) and ("optimizer" in ckpt):
            optimizer.load_state_dict(ckpt["optimizer"])
        if bool(int(args.resume_optimizer)) and (scheduler is not None) and ("scheduler" in ckpt):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", start_epoch * max(1, len(train_loader))))
        mprint(f"[resume] loaded={resume_path} start_epoch={start_epoch} step={global_step}")
    else:
        mprint("[resume] disabled (no checkpoint found/requested)")

    total_steps = max(1, int(args.epochs) * max(1, len(train_loader)))
    warmup_steps = max(1, int(float(args.dual_mask_warmup_frac) * float(total_steps)))
    printed_token_sanity = False

    try:
        for epoch in range(start_epoch, int(args.epochs)):
            model.train()
            sampler_obj = getattr(train_loader, "sampler", None)
            if sampler_obj is not None and hasattr(sampler_obj, "set_epoch"):
                sampler_obj.set_epoch(epoch)

            for batch in train_loader:
                with accelerator.accumulate(model):
                    ramp = 1.0
                    if warmup_steps > 0:
                        ramp = min(float(global_step) / float(warmup_steps), 1.0)
                    dm_near = float(args.dual_mask_near) * ramp
                    dm_far = float(args.dual_mask_far) * ramp

                    out = model(
                        pt_xyz=batch["pt_xyz"],
                        pt_n=None,
                        pt_dist=batch.get("pt_dist", None),
                        pt_grad=batch.get("pt_grad", None),
                        ray_o=batch.get("ray_o", None),
                        ray_d=batch.get("ray_d", None),
                        ray_t=batch.get("ray_t", None),
                        ray_hit=batch.get("ray_hit", None),
                        ray_n=batch.get("ray_n", None),
                        ray_unc=batch.get("ray_unc", None),
                        ray_available=batch.get("ray_available", None),
                        is_causal=True,
                        dual_mask_near=dm_near,
                        dual_mask_far=dm_far,
                        dual_mask_window=int(args.dual_mask_window),
                        dual_mask_type_aware=int(args.dual_mask_type_aware),
                    )
                    if (not printed_token_sanity) and (global_step == 0) and accelerator.is_main_process:
                        c = Counter(out.type_id[0].detach().cpu().tolist())
                        sanity_items = [
                            ("BOS", TYPE_BOS),
                            ("EOS", TYPE_EOS),
                            ("POINT", TYPE_POINT),
                            ("RAY", TYPE_RAY),
                            ("Q_POINT", TYPE_Q_POINT),
                            ("A_POINT", TYPE_A_POINT),
                            ("Q_RAY", TYPE_Q_RAY),
                            ("A_RAY", TYPE_A_RAY),
                            ("SEP", TYPE_SEP),
                            ("MISSING_RAY", TYPE_MISSING_RAY),
                        ]
                        print("\n" + "=" * 56)
                        print("[Sanity Check] step=0 token counts (sample 0)")
                        for name, tid in sanity_items:
                            print(f"  {name:12s} ({int(tid):2d}): {int(c.get(int(tid), 0))}")
                        print("=" * 56 + "\n")
                        printed_token_sanity = True
                    loss_terms = [
                        PatchTransformerNepa.nepa_loss(out.z, out.z_hat, out.type_id, skip_k=int(k))
                        for k in skip_k_list
                    ]
                    loss = torch.stack(loss_terms).mean()
                    accelerator.backward(loss)
                    if float(args.max_grad_norm) > 0:
                        accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
                    optimizer.step()
                    if bool(int(args.use_ema)) and ema_model is not None:
                        _update_ema(ema_model, model_raw, float(args.ema_decay))
                    optimizer.zero_grad(set_to_none=True)

                if accelerator.is_main_process and (global_step % 50 == 0):
                    lr = optimizer.param_groups[0]["lr"]
                    diag_msg = ""
                    diag = None
                    if diag_enabled and (global_step % diag_every == 0):
                        k_diag = int(args.diag_k) if int(args.diag_k) > 0 else int(skip_k_list[0])
                        with torch.no_grad():
                            diag = _compute_copy_diag(out.z, out.z_hat, out.type_id, k_diag)
                        if diag is not None:
                            diag_msg = (
                                f" kdiag={int(diag['k'])}"
                                f" cos_tgt={_fmt_diag(float(diag['cos_tgt']))}"
                                f" cos_prev={_fmt_diag(float(diag['cos_prev']))}"
                                f" gap={_fmt_diag(float(diag['gap']))}"
                                f" copy_win={_fmt_diag(float(diag['copy_win']))}"
                            )
                    print(
                        f"[epoch {epoch:03d} step {global_step:06d}] "
                        f"loss={loss.item():.8e} lr={lr:.2e} "
                        f"dm=({dm_near:.2f},{dm_far:.2f},ramp={ramp:.2f})"
                        f"{diag_msg}"
                    )
                    if wandb_run is not None and (global_step % wandb_log_every == 0):
                        wb = {
                            "train/loss": float(loss.item()),
                            "train/lr": float(lr),
                            "train/epoch": float(epoch),
                            "train/global_step": float(global_step),
                            "train/dm_near": float(dm_near),
                            "train/dm_far": float(dm_far),
                            "train/dm_ramp": float(ramp),
                        }
                        if diag is not None:
                            wb["diag/cos_tgt"] = float(diag["cos_tgt"])
                            wb["diag/cos_prev"] = float(diag["cos_prev"])
                            wb["diag/gap"] = float(diag["gap"])
                            wb["diag/copy_win"] = float(diag["copy_win"])
                        wandb_run.log(wb, step=int(global_step))
                global_step += 1

            if scheduler is not None:
                scheduler.step()
            if accelerator.is_main_process:
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch_end": float(epoch + 1),
                            "train/lr_epoch_end": float(optimizer.param_groups[0]["lr"]),
                        },
                        step=int(global_step),
                    )
                if ((epoch + 1) % int(args.save_every)) == 0:
                    save_ckpt(
                        save_root / f"ckpt_epoch_{epoch:04d}.pt",
                        model_raw,
                        ema_model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        args,
                    )
                    save_ckpt(
                        save_root / "ckpt_latest.pt",
                        model_raw,
                        ema_model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        args,
                    )
    finally:
        if accelerator.is_main_process and (wandb_run is not None):
            wandb_run.finish()

    if accelerator.is_main_process:
        print(f"Done. checkpoints: {save_root}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Avoid noisy NCCL warnings on clean shutdown in multi-process runs.
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
        except Exception:
            pass
