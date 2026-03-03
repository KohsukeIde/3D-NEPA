"""Token-stream PatchNEPA pretraining on v2 surface/query datasets.

Sequence format:
  [BOS, ctx_patch_1..P, SEP, qry_Q1, qry_A1, ..., qry_QN, qry_AN, EOS]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterator, Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.data.dataset_v2 import v2_collate_fn
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
    TYPE_SEP_CTX,
    TYPE_SEP_QA,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_patch_nepa_tokens")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs_patch_nepa_tokens")
    p.add_argument("--run_name", type=str, default="debug_tokens")
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument("--n_surf", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=1024)
    p.add_argument("--n_ray", type=int, default=0)
    p.add_argument(
        "--token_qa_layout",
        type=str,
        default="interleave",
        choices=["interleave", "split", "split_sep"],
        help=(
            "Q/A layout inside forward_tokens query segment: "
            "interleave=[Q1,A1,Q2,A2,...], split=[Q...A...], split_sep=[Q...,SEP_QA,A...]"
        ),
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--pm_pc_norm", type=int, default=1, choices=[0, 1], help="Apply Point-MAE pc_norm on v2 samples.")
    p.add_argument(
        "--pm_scale_translate",
        type=int,
        default=1,
        choices=[0, 1],
        help="Apply Point-MAE scale+translate augmentation on v2 train samples.",
    )
    p.add_argument("--pm_scale_low", type=float, default=(2.0 / 3.0))
    p.add_argument("--pm_scale_high", type=float, default=(3.0 / 2.0))
    p.add_argument("--pm_translate", type=float, default=0.2)
    p.add_argument(
        "--pm_transform_answers",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, transform ans_feat consistently under pm_pc_norm/pm_scale_translate.",
    )

    p.add_argument("--patch_embed", type=str, default="fps_knn", choices=["serial", "pointgpt", "fps_knn"])
    p.add_argument("--patch_local_encoder", type=str, default="pointmae_conv", choices=["mlp", "pointmae_conv"])
    p.add_argument("--patch_fps_random_start", type=int, default=1, choices=[0, 1])
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--serial_order", type=str, default="morton")
    p.add_argument("--serial_bits", type=int, default=10)
    p.add_argument("--serial_shuffle_within_patch", type=int, default=0, choices=[0, 1])
    p.add_argument("--patch_order_mode", type=str, default="none")

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
    p.add_argument("--backbone_mode", type=str, default="nepa2d", choices=["nepa2d", "vanilla", "pointmae"])

    p.add_argument("--answer_in_dim", type=int, default=0, help="0 => infer from first sample ans_feat dim")
    p.add_argument("--answer_mlp_layers", type=int, default=2)
    p.add_argument("--answer_pool", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--loss_target_mode", type=str, default="content_tokens", choices=["full_z", "content_tokens"])
    p.add_argument("--skip_k", type=int, default=1)

    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_steps", type=int, default=-1, help="If >=0, overrides warmup_ratio.")
    p.add_argument("--warmup_ratio", type=float, default=0.025, help="Warmup ratio over max_steps.")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--q_mask_prob", type=float, default=0.0)
    p.add_argument("--q_mask_mode", type=str, default="mask_token", choices=["mask_token", "zero"])
    p.add_argument("--dual_mask_near", type=float, default=0.0)
    p.add_argument("--dual_mask_far", type=float, default=0.0)
    p.add_argument("--dual_mask_window", type=int, default=0)
    p.add_argument("--dual_mask_type_aware", type=int, default=0, choices=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-pretrain")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=1)
    p.add_argument("--diag_every", type=int, default=1, help="Always-on copy diagnostics logging interval.")
    return p.parse_args()


def build_qa_sequence(
    model: PatchTransformerNepa,
    surf_xyz: torch.Tensor,
    qry_xyz: torch.Tensor,
    ans_feat: torch.Tensor,
    *,
    token_qa_layout: str = "interleave",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [tokens, type_id, centers_xyz] for forward_tokens()."""
    b = surf_xyz.shape[0]
    dev = surf_xyz.device

    ctx_tok, ctx_centers, _ = model.encode_patches(surf_xyz)
    q_tok, q_centers = model.encode_point_queries(qry_xyz)
    a_tok, a_centers = model.encode_point_answers(ans_feat, qry_xyz)

    p = int(ctx_tok.shape[1])
    n = int(q_tok.shape[1])
    layout = str(token_qa_layout)
    if layout == "interleave":
        qa_tok = torch.stack([q_tok, a_tok], dim=2).reshape(b, 2 * n, -1)
        qa_centers = torch.stack([q_centers, a_centers], dim=2).reshape(b, 2 * n, 3)
        qa_type = torch.empty((b, 2 * n), device=dev, dtype=torch.long)
        qa_type[:, 0::2] = int(TYPE_Q_POINT)
        qa_type[:, 1::2] = int(TYPE_A_POINT)
    elif layout == "split":
        qa_tok = torch.cat([q_tok, a_tok], dim=1)
        qa_centers = torch.cat([q_centers, a_centers], dim=1)
        qa_type = torch.cat(
            [
                torch.full((b, n), int(TYPE_Q_POINT), device=dev, dtype=torch.long),
                torch.full((b, n), int(TYPE_A_POINT), device=dev, dtype=torch.long),
            ],
            dim=1,
        )
    elif layout == "split_sep":
        qa_tok = torch.cat([q_tok, model.sep_token.expand(b, 1, -1), a_tok], dim=1)
        qa_centers = torch.cat(
            [
                q_centers,
                torch.zeros((b, 1, 3), device=dev, dtype=surf_xyz.dtype),
                a_centers,
            ],
            dim=1,
        )
        qa_type = torch.cat(
            [
                torch.full((b, n), int(TYPE_Q_POINT), device=dev, dtype=torch.long),
                torch.full((b, 1), int(TYPE_SEP_QA), device=dev, dtype=torch.long),
                torch.full((b, n), int(TYPE_A_POINT), device=dev, dtype=torch.long),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"unknown token_qa_layout={layout}")

    z0 = torch.zeros((b, 1, 3), device=dev, dtype=surf_xyz.dtype)
    tokens = torch.cat(
        [
            model.bos_token.expand(b, 1, -1),
            ctx_tok,
            model.sep_ctx_token.expand(b, 1, -1),
            qa_tok,
            model.eos_token.expand(b, 1, -1),
        ],
        dim=1,
    )
    centers = torch.cat([z0, ctx_centers, z0, qa_centers, z0], dim=1)

    type_id = torch.cat(
        [
            torch.full((b, 1), int(TYPE_BOS), device=dev, dtype=torch.long),
            torch.full((b, p), int(TYPE_POINT), device=dev, dtype=torch.long),
            torch.full((b, 1), int(TYPE_SEP_CTX), device=dev, dtype=torch.long),
            qa_type,
            torch.full((b, 1), int(TYPE_EOS), device=dev, dtype=torch.long),
        ],
        dim=1,
    )
    return tokens, type_id, centers


def _infinite_loader(dl: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch


def _save_ckpt(
    path: Path,
    model: PatchTransformerNepa,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(
        payload,
        str(path),
    )


def _init_wandb(args: argparse.Namespace, accelerator: Accelerator):
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
    if mode == "disabled":
        accelerator.print("[wandb] disabled by mode=disabled")
        return None
    try:
        run = wandb.init(
            project=str(args.wandb_project),
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            mode=mode,
            config=vars(args),
        )
        accelerator.print(
            f"[wandb] enabled project={args.wandb_project} run={run_name} "
            f"group={group if group else '-'} mode={mode}"
        )
        return run
    except Exception as e:
        accelerator.print(f"[wandb] disabled: init failed ({e})")
        return None


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
        & (tgt_ty != int(TYPE_EOS))
        & (tgt_ty != int(TYPE_SEP_CTX))
        & (tgt_ty != int(TYPE_SEP_QA))
        & (tgt_ty != int(TYPE_MISSING_RAY))
    )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> float:
    if mask is None:
        return float(values.mean().item())
    if not bool(mask.any()):
        return float("nan")
    return float(values[mask].mean().item())


def _compute_copy_diag(
    target_seq: torch.Tensor,
    z_hat: torch.Tensor,
    type_id: torch.Tensor | None,
    k: int,
) -> dict[str, float] | None:
    k = max(1, int(k))
    if target_seq.size(1) <= k or z_hat.size(1) <= k:
        return None

    pred = z_hat[:, :-k, :].detach()
    tgt = target_seq[:, k:, :].detach()
    prev = target_seq[:, :-k, :].detach()

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


def _resolve_warmup_steps(args: argparse.Namespace) -> int:
    if int(args.warmup_steps) >= 0:
        return int(args.warmup_steps)
    ratio = max(0.0, float(args.warmup_ratio))
    return int(round(float(args.max_steps) * ratio))


def _scheduler_scale(step: int, args: argparse.Namespace, warmup_steps: int) -> float:
    if str(args.lr_scheduler) != "cosine":
        return 1.0
    max_steps = max(1, int(args.max_steps))
    s = min(max(0, int(step)), max_steps)
    if warmup_steps > 0 and s < warmup_steps:
        return float(s + 1) / float(max(1, warmup_steps))

    denom = max(1, max_steps - max(0, warmup_steps))
    t = float(s - max(0, warmup_steps)) / float(denom)
    t = min(max(t, 0.0), 1.0)
    min_scale = float(args.min_lr) / float(args.lr)
    cosv = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_scale + (1.0 - min_scale) * cosv


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=int(args.grad_accum))
    set_seed(int(args.seed))

    save_root = Path(args.save_dir) / args.run_name
    if accelerator.is_main_process:
        save_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    dataset, sampler, info = build_mixed_pretrain(
        mix_config_path=str(args.mix_config_path),
        n_point=int(args.n_surf),
        n_ray=int(args.n_ray),
        num_workers=int(args.num_workers),
        mode="train",
        eval_seed=int(args.seed),
        return_raw=True,
        v2_n_qry=(None if int(args.n_qry) <= 0 else int(args.n_qry)),
        v2_pointmae_pc_norm=bool(int(args.pm_pc_norm)),
        v2_pointmae_scale_translate=bool(int(args.pm_scale_translate)),
        v2_pointmae_scale_low=float(args.pm_scale_low),
        v2_pointmae_scale_high=float(args.pm_scale_high),
        v2_pointmae_translate=float(args.pm_translate),
        v2_transform_answers=bool(int(args.pm_transform_answers)),
    )
    probe = dataset[0]
    if probe.get("ans_feat", None) is None:
        raise RuntimeError("v2 token pretrain requires ans_feat in dataset items.")
    answer_in_dim = int(args.answer_in_dim) if int(args.answer_in_dim) > 0 else int(probe["ans_feat"].shape[-1])
    if answer_in_dim <= 0:
        raise RuntimeError(f"invalid answer_in_dim={answer_in_dim}")

    dl = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=v2_collate_fn,
    )

    model = PatchTransformerNepa(
        patch_embed=str(args.patch_embed),
        patch_local_encoder=str(args.patch_local_encoder),
        patch_fps_random_start=bool(int(args.patch_fps_random_start)),
        n_point=int(args.n_surf),
        group_size=int(args.group_size),
        num_groups=int(args.num_groups),
        serial_order=str(args.serial_order),
        serial_bits=int(args.serial_bits),
        serial_shuffle_within_patch=int(args.serial_shuffle_within_patch),
        patch_order_mode=str(args.patch_order_mode),
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
        qa_tokens=1,
        qa_layout="split",
        qa_sep_token=True,
        qa_fuse="add",
        use_pt_dist=False,
        use_pt_grad=False,
        answer_in_dim=int(answer_in_dim),
        answer_mlp_layers=int(args.answer_mlp_layers),
        answer_pool=str(args.answer_pool),
        q_mask_mode=str(args.q_mask_mode),
        use_ray_patch=False,
    )
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    model_raw = accelerator.unwrap_model(model)
    warmup_steps = max(0, _resolve_warmup_steps(args))
    scheduler: optim.lr_scheduler.LambdaLR | None = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda st: _scheduler_scale(int(st), args, warmup_steps),
        )

    if accelerator.is_main_process:
        accelerator.print(f"[token_pretrain] answer_in_dim={answer_in_dim}")
        accelerator.print(f"[token_pretrain] mix_info={info}")
        accelerator.print(
            "[token_pretrain] pointmae_compat "
            f"pc_norm={bool(int(args.pm_pc_norm))} "
            f"scale_translate={bool(int(args.pm_scale_translate))} "
            f"scale=[{float(args.pm_scale_low):.4f},{float(args.pm_scale_high):.4f}] "
            f"translate={float(args.pm_translate):.4f} "
            f"transform_answers={bool(int(args.pm_transform_answers))}"
        )
        accelerator.print(f"[token_pretrain] token_qa_layout={str(args.token_qa_layout)}")
        accelerator.print(
            f"[token_pretrain] optimizer lr={float(args.lr):.3e} wd={float(args.weight_decay):.3f} "
            f"scheduler={str(args.lr_scheduler)} warmup_steps={int(warmup_steps)} "
            f"warmup_ratio={float(args.warmup_ratio):.4f} min_lr={float(args.min_lr):.2e}"
        )
    wandb_run = _init_wandb(args, accelerator)
    wandb_log_every = max(1, int(args.wandb_log_every))
    diag_every = max(1, int(args.diag_every))

    data_iter = _infinite_loader(dl)
    model.train()
    step = 0
    pbar = tqdm(total=int(args.max_steps), disable=not accelerator.is_main_process)
    while step < int(args.max_steps):
        batch = next(data_iter)
        surf_xyz = batch["surf_xyz"]
        qry_xyz = batch["qry_xyz"]
        ans_feat = batch["ans_feat"]
        if qry_xyz is None or ans_feat is None:
            raise RuntimeError("batch missing qry_xyz/ans_feat; ensure v2 config uses return_qry=true.")
        surf_xyz = surf_xyz.to(accelerator.device, non_blocking=True)
        qry_xyz = qry_xyz.to(accelerator.device, non_blocking=True)
        ans_feat = ans_feat.to(accelerator.device, non_blocking=True)

        with accelerator.accumulate(model):
            tokens, type_id, centers = build_qa_sequence(
                model,
                surf_xyz,
                qry_xyz,
                ans_feat,
                token_qa_layout=str(args.token_qa_layout),
            )
            out = model.forward_tokens(
                tokens=tokens,
                type_id=type_id,
                centers_xyz=centers,
                is_causal=True,
                q_mask_prob=float(args.q_mask_prob),
                dual_mask_near=float(args.dual_mask_near),
                dual_mask_far=float(args.dual_mask_far),
                dual_mask_window=int(args.dual_mask_window),
                dual_mask_type_aware=bool(int(args.dual_mask_type_aware)),
            )
            target = out.z if str(args.loss_target_mode) == "full_z" else out.tokens
            loss = model_raw.nepa_loss(
                out.z,
                out.z_hat,
                out.type_id,
                skip_k=int(args.skip_k),
                target=target,
            )
            diag = _compute_copy_diag(target, out.z_hat, out.type_id, int(args.skip_k))
            accelerator.backward(loss)
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            if scheduler is not None and accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_description(f"loss={loss.item():.4f}")
                if (step % diag_every == 0) and (diag is not None):
                    accelerator.print(
                        f"[step {step:06d}] "
                        f"loss={float(loss.item()):.6f} "
                        f"cos_tgt={_fmt_diag(float(diag['cos_tgt']))} "
                        f"cos_prev={_fmt_diag(float(diag['cos_prev']))} "
                        f"gap={_fmt_diag(float(diag['gap']))} "
                        f"copy_win={_fmt_diag(float(diag['copy_win']))}"
                    )
                if wandb_run is not None and (step % wandb_log_every == 0):
                    wb = {
                        "train/loss": float(loss.item()),
                        "train/step": int(step),
                    }
                    if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
                        wb["train/lr"] = float(optimizer.param_groups[0]["lr"])
                    if diag is not None:
                        wb["diag/cos_tgt"] = float(diag["cos_tgt"])
                        wb["diag/cos_prev"] = float(diag["cos_prev"])
                        wb["diag/gap"] = float(diag["gap"])
                        wb["diag/copy_win"] = float(diag["copy_win"])
                    wandb_run.log(wb, step=int(step))
            if int(args.save_every) > 0 and (step % int(args.save_every) == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(save_root / f"ckpt_step{step}.pt", model_raw, optimizer, scheduler, step, args)
    pbar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_ckpt(save_root / "ckpt_final.pt", model_raw, optimizer, scheduler, step, args)
        accelerator.print(f"[done] saved to {save_root}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
