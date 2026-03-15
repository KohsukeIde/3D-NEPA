"""Token-stream PatchNEPA pretraining on v2 surface/query datasets.

Sequence format:
  [BOS, ctx_patch_1..P, SEP, qry_Q1, qry_A1, ..., qry_QN, qry_AN, EOS]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from nepa3d.data.dataset_v2 import v2_collate_fn
from nepa3d.data.mixed_pretrain import build_mixed_pretrain
from nepa3d.core.models.causal_transformer import CausalTransformer
from nepa3d.tracks.patch_nepa.mainline.models.patch_nepa import PatchTransformerNepa
from nepa3d.token.tokenizer import (
    TYPE_A_POINT,
    TYPE_A_POINT_MESH,
    TYPE_A_POINT_PC,
    TYPE_A_POINT_UDF,
    TYPE_A_RAY,
    TYPE_BOS,
    TYPE_EOS,
    TYPE_MISSING_RAY,
    TYPE_POINT,
    TYPE_Q_POINT,
    TYPE_Q_POINT_MESH,
    TYPE_Q_POINT_PC,
    TYPE_Q_POINT_UDF,
    TYPE_RAY,
    TYPE_SEP_CTX,
    TYPE_SEP_QA,
)

Q_POINT_TYPES = (
    int(TYPE_Q_POINT),
    int(TYPE_Q_POINT_MESH),
    int(TYPE_Q_POINT_UDF),
    int(TYPE_Q_POINT_PC),
)
A_POINT_TYPES = (
    int(TYPE_A_POINT),
    int(TYPE_A_POINT_MESH),
    int(TYPE_A_POINT_UDF),
    int(TYPE_A_POINT_PC),
)

RECON_LOSS_MODE_TO_ID = {
    "composite": 0.0,
    "pointgpt_ctx_only": 1.0,
    "answer_only": 2.0,
    "context_plus_answer": 3.0,
    "query_plus_answer": 4.0,
}


@dataclass
class QASequenceBuild:
    tokens: torch.Tensor
    type_id: torch.Tensor
    centers: torch.Tensor
    ctx_centers: torch.Tensor
    ctx_group_idx: torch.Tensor
    n_ctx: int
    n_qry: int
    token_qa_layout: str


class ReconHeads(nn.Module):
    """Heads for raw reconstruction objectives."""

    def __init__(self, d_model: int, group_size: int, answer_in_dim: int) -> None:
        super().__init__()
        ctx_dim = int(group_size) * 3
        self.ctx_head = nn.Linear(int(d_model), int(ctx_dim))
        self.q_head = nn.Linear(int(d_model), 3)
        self.a_head = nn.Linear(int(d_model), int(answer_in_dim))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.ctx_head(h), self.q_head(h), self.a_head(h)


def _primitive_to_point_type_pair(name: str) -> tuple[int, int]:
    k = str(name).strip().lower()
    if k == "mesh":
        return int(TYPE_Q_POINT_MESH), int(TYPE_A_POINT_MESH)
    if k == "udf":
        return int(TYPE_Q_POINT_UDF), int(TYPE_A_POINT_UDF)
    if k == "pc":
        return int(TYPE_Q_POINT_PC), int(TYPE_A_POINT_PC)
    return int(TYPE_Q_POINT), int(TYPE_A_POINT)


def _resolve_primitive_point_types(
    primitive: Optional[torch.Tensor | list | tuple | str],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_default = int(TYPE_Q_POINT)
    a_default = int(TYPE_A_POINT)
    q = torch.full((int(batch_size),), q_default, device=device, dtype=torch.long)
    a = torch.full((int(batch_size),), a_default, device=device, dtype=torch.long)

    if primitive is None:
        return q, a

    if torch.is_tensor(primitive):
        vals = primitive.detach().cpu().tolist()
    elif isinstance(primitive, (list, tuple)):
        vals = list(primitive)
    else:
        vals = [primitive] * int(batch_size)

    n = min(int(batch_size), len(vals))
    for i in range(n):
        qv, av = _primitive_to_point_type_pair(str(vals[i]))
        q[i] = int(qv)
        a[i] = int(av)
    return q, a


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
        default="split",
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
    p.add_argument("--patch_order_mode", type=str, default="morton")

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
    p.add_argument(
        "--pretrain_objective",
        type=str,
        default="nepa_cosine",
        choices=["nepa_cosine", "infonce", "recon_mse", "recon_chamfer"],
        help="Main pretrain objective: latent cosine NEPA or raw reconstruction variants.",
    )
    p.add_argument(
        "--infonce_tau",
        type=float,
        default=0.07,
        help="Temperature for --pretrain_objective=infonce.",
    )
    p.add_argument("--recon_ctx_weight", type=float, default=1.0, help="Weight for context patch reconstruction loss.")
    p.add_argument("--recon_q_weight", type=float, default=1.0, help="Weight for query-point xyz reconstruction loss.")
    p.add_argument("--recon_a_weight", type=float, default=1.0, help="Weight for answer-feature reconstruction loss.")
    p.add_argument(
        "--recon_loss_mode",
        type=str,
        default="composite",
        choices=[
            "composite",
            "pointgpt_ctx_only",
            "answer_only",
            "context_plus_answer",
            "query_plus_answer",
        ],
        help=(
            "Reconstruction loss composition. "
            "'composite' = ctx+q+a (weighted), "
            "'pointgpt_ctx_only' = context reconstruction only (PointGPT-equivalent axis), "
            "'answer_only' = answer reconstruction only, "
            "'context_plus_answer' = context+answer reconstruction, "
            "'query_plus_answer' = query+answer reconstruction."
        ),
    )
    p.add_argument(
        "--recon_generator_depth",
        type=int,
        default=0,
        help=(
            "Extra causal-transformer depth for reconstruction branch only. "
            "0 disables generator (legacy), >=1 enables generator before recon heads."
        ),
    )
    p.add_argument(
        "--recon_chamfer_metric",
        type=str,
        default="l2",
        choices=["l1", "l2", "l12"],
        help="Chamfer variant for --pretrain_objective=recon_chamfer.",
    )
    p.add_argument(
        "--loss_target_mode",
        type=str,
        default="content_tokens",
        choices=["full_z", "content_tokens", "content_plus_center"],
        help="Target mode for NEPA loss: full_z | content_tokens | content_plus_center.",
    )
    p.add_argument(
        "--center_target_alpha",
        type=float,
        default=0.5,
        help="Scale for center_mlp term used by loss_target_mode=content_plus_center.",
    )
    p.add_argument(
        "--reg_var_weight",
        type=float,
        default=0.0,
        help="Weight for variance regularization on selected reg_source embeddings.",
    )
    p.add_argument(
        "--reg_cov_weight",
        type=float,
        default=0.0,
        help="Weight for covariance regularization on selected reg_source embeddings.",
    )
    p.add_argument(
        "--reg_var_gamma",
        type=float,
        default=1.0,
        help="Target std floor for variance regularization.",
    )
    p.add_argument(
        "--reg_var_eps",
        type=float,
        default=1e-4,
        help="Numerical epsilon used in variance regularization.",
    )
    p.add_argument(
        "--reg_scope",
        type=str,
        default="intra_shape",
        choices=["batch", "intra_shape"],
        help="Scope for var/cov regularization: batch | intra_shape.",
    )
    p.add_argument(
        "--reg_source",
        type=str,
        default="target",
        choices=["target", "hidden"],
        help="Source embeddings for var/cov regularization: target (legacy) | hidden (out.h).",
    )
    p.add_argument(
        "--loss_mask_mode",
        type=str,
        default="answer_and_point_context",
        choices=["answer_only_if_present", "answer_and_point_context", "non_special"],
        help=(
            "Token loss mask mode: "
            "answer_only_if_present (legacy), "
            "answer_and_point_context (A + TYPE_POINT/TYPE_RAY), "
            "non_special (all non-special tokens)."
        ),
    )
    p.add_argument("--skip_k", type=int, default=1)
    p.add_argument(
        "--nepa_center_mode",
        type=str,
        default="none",
        choices=["none", "shape", "segment"],
        help="Centering mode for nepa_cosine objective: none | shape | segment(Q/A-wise).",
    )
    p.add_argument(
        "--nepa_center_warmup_frac",
        type=float,
        default=0.0,
        help="Warmup fraction for centered-cosine strength (0->1 ramp over max_steps).",
    )

    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="If >0, override max_steps by epochs * steps_per_epoch after dataloader build.",
    )
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
    p.add_argument("--dual_mask_mode", type=str, default="element", choices=["element", "column"])
    p.add_argument("--dual_mask_keep_prefix", type=int, default=10)
    p.add_argument(
        "--dual_mask_column_ratio",
        type=float,
        default=0.7,
        help="PointGPT-like column drop ratio used when dual_mask_mode=column.",
    )
    p.add_argument(
        "--dual_mask_warmup_frac",
        type=float,
        default=0.0,
        help="Warmup fraction to linearly ramp dual_mask_near/far/column_ratio from 0 to target.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_wandb", type=int, default=0, choices=[0, 1])
    p.add_argument("--wandb_project", type=str, default="patchnepa-pretrain")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_run_name", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_log_every", type=int, default=1)
    p.add_argument("--diag_every", type=int, default=1, help="Diagnostics logging interval.")
    return p.parse_args()


def build_qa_sequence(
    model: PatchTransformerNepa,
    surf_xyz: torch.Tensor,
    qry_xyz: torch.Tensor,
    ans_feat: torch.Tensor,
    primitive: Optional[torch.Tensor | list | tuple | str] = None,
    *,
    token_qa_layout: str = "split",
) -> QASequenceBuild:
    """Build sequence + metadata for forward_tokens and optional recon losses."""
    b = surf_xyz.shape[0]
    dev = surf_xyz.device

    ctx_tok, ctx_centers, ctx_group_idx = model.encode_patches(surf_xyz)
    q_tok, q_centers = model.encode_point_queries(qry_xyz)
    a_tok, a_centers = model.encode_point_answers(ans_feat, qry_xyz)
    q_point_types, a_point_types = _resolve_primitive_point_types(
        primitive,
        batch_size=b,
        device=dev,
    )

    p = int(ctx_tok.shape[1])
    n = int(q_tok.shape[1])
    q_pt = q_point_types.unsqueeze(1).expand(-1, n)
    a_pt = a_point_types.unsqueeze(1).expand(-1, n)
    layout = str(token_qa_layout)
    if layout == "interleave":
        qa_tok = torch.stack([q_tok, a_tok], dim=2).reshape(b, 2 * n, -1)
        qa_centers = torch.stack([q_centers, a_centers], dim=2).reshape(b, 2 * n, 3)
        qa_type = torch.empty((b, 2 * n), device=dev, dtype=torch.long)
        qa_type[:, 0::2] = q_pt
        qa_type[:, 1::2] = a_pt
    elif layout == "split":
        qa_tok = torch.cat([q_tok, a_tok], dim=1)
        qa_centers = torch.cat([q_centers, a_centers], dim=1)
        qa_type = torch.cat([q_pt, a_pt], dim=1)
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
                q_pt,
                torch.full((b, 1), int(TYPE_SEP_QA), device=dev, dtype=torch.long),
                a_pt,
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
            q_point_types.unsqueeze(1).expand(-1, p),
            torch.full((b, 1), int(TYPE_SEP_CTX), device=dev, dtype=torch.long),
            qa_type,
            torch.full((b, 1), int(TYPE_EOS), device=dev, dtype=torch.long),
        ],
        dim=1,
    )
    return QASequenceBuild(
        tokens=tokens,
        type_id=type_id,
        centers=centers,
        ctx_centers=ctx_centers,
        ctx_group_idx=ctx_group_idx,
        n_ctx=p,
        n_qry=n,
        token_qa_layout=layout,
    )


def _forward_tokens_wrapped(
    model: torch.nn.Module,
    *,
    tokens: torch.Tensor,
    type_id: torch.Tensor,
    centers_xyz: torch.Tensor,
    is_causal: bool,
    q_mask_prob: float,
    dual_mask_near: float,
    dual_mask_far: float,
    dual_mask_window: int,
    dual_mask_type_aware: bool,
    dual_mask_mode: str,
    dual_mask_keep_prefix: int,
    dual_mask_column_ratio: float,
) -> PatchNepaOutput:
    """Call forward_tokens for raw/accelerate/DDP-wrapped modules."""
    if hasattr(model, "forward_tokens"):
        return model.forward_tokens(
            tokens=tokens,
            type_id=type_id,
            centers_xyz=centers_xyz,
            is_causal=is_causal,
            q_mask_prob=q_mask_prob,
            dual_mask_near=dual_mask_near,
            dual_mask_far=dual_mask_far,
            dual_mask_window=dual_mask_window,
            dual_mask_type_aware=dual_mask_type_aware,
            dual_mask_mode=dual_mask_mode,
            dual_mask_keep_prefix=dual_mask_keep_prefix,
            dual_mask_column_ratio=dual_mask_column_ratio,
        )
    wrapped = getattr(model, "module", None)
    if wrapped is not None and hasattr(wrapped, "forward_tokens"):
        return wrapped.forward_tokens(
            tokens=tokens,
            type_id=type_id,
            centers_xyz=centers_xyz,
            is_causal=is_causal,
            q_mask_prob=q_mask_prob,
            dual_mask_near=dual_mask_near,
            dual_mask_far=dual_mask_far,
            dual_mask_window=dual_mask_window,
            dual_mask_type_aware=dual_mask_type_aware,
            dual_mask_mode=dual_mask_mode,
            dual_mask_keep_prefix=dual_mask_keep_prefix,
            dual_mask_column_ratio=dual_mask_column_ratio,
        )
    raise AttributeError("model has no forward_tokens() (wrapped or unwrapped).")


def _sequence_absolute_positions(
    n_ctx: int,
    n_qry: int,
    layout: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Absolute sequence positions for context/Q/A tokens in [BOS,ctx,SEP_CTX,qa...,EOS]."""
    ctx_abs = torch.arange(1, 1 + int(n_ctx), device=device, dtype=torch.long)
    qa_start = 1 + int(n_ctx) + 1  # BOS + ctx + SEP_CTX
    q_idx = torch.arange(int(n_qry), device=device, dtype=torch.long)
    if str(layout) == "interleave":
        q_abs = qa_start + (2 * q_idx)
        a_abs = qa_start + (2 * q_idx) + 1
    elif str(layout) == "split":
        q_abs = qa_start + q_idx
        a_abs = qa_start + int(n_qry) + q_idx
    elif str(layout) == "split_sep":
        q_abs = qa_start + q_idx
        a_abs = qa_start + int(n_qry) + 1 + q_idx
    else:
        raise ValueError(f"unknown token_qa_layout={layout}")
    return ctx_abs, q_abs, a_abs


def _gather_rel_xyz_flat(
    surf_xyz: torch.Tensor,
    group_idx: torch.Tensor,
    centers_xyz: torch.Tensor,
) -> torch.Tensor:
    """Gather grouped points and return flattened relative xyz [B,P,K*3]."""
    b, p, k = int(group_idx.shape[0]), int(group_idx.shape[1]), int(group_idx.shape[2])
    idx = group_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    src = surf_xyz.unsqueeze(1).expand(-1, p, -1, 3)
    grouped = torch.gather(src, dim=2, index=idx)
    rel = grouped - centers_xyz.unsqueeze(2)
    return rel.reshape(b, p, k * 3)


def _masked_index_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    fn: str = "mse",
    group_size: int = 32,
    chamfer_loss: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Compute loss between selected sequence predictions and selected targets."""
    if int(pred_idx.numel()) == 0:
        return pred.sum() * 0.0
    pred_sel = pred[:, pred_idx, :]
    tgt_sel = tgt[:, tgt_idx, :]
    if fn == "mse":
        return F.mse_loss(pred_sel, tgt_sel)
    if fn == "chamfer":
        if chamfer_loss is None:
            raise RuntimeError("recon_chamfer requested but chamfer_loss is None")
        b, m, _ = pred_sel.shape
        pred_pts = pred_sel.reshape(b * m, int(group_size), 3)
        tgt_pts = tgt_sel.reshape(b * m, int(group_size), 3)
        return chamfer_loss(pred_pts, tgt_pts)
    raise ValueError(f"unknown loss fn={fn}")


def _pair_error_value(
    pred_sel: torch.Tensor,
    tgt_sel: torch.Tensor,
    *,
    fn: str,
    group_size: int = 32,
    chamfer_loss: Optional[nn.Module] = None,
) -> float:
    if int(pred_sel.numel()) == 0:
        return float("nan")
    if fn == "mse":
        return float(F.mse_loss(pred_sel, tgt_sel).item())
    if fn == "chamfer":
        if chamfer_loss is None:
            return float("nan")
        b, m, _ = pred_sel.shape
        pred_pts = pred_sel.reshape(b * m, int(group_size), 3)
        tgt_pts = tgt_sel.reshape(b * m, int(group_size), 3)
        return float(chamfer_loss(pred_pts, tgt_pts).item())
    return float("nan")


def _masked_index_error_value(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    pred_idx: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    fn: str,
    group_size: int = 32,
    chamfer_loss: Optional[nn.Module] = None,
) -> float:
    if int(pred_idx.numel()) == 0:
        return float("nan")
    pred_sel = pred[:, pred_idx, :]
    tgt_sel = tgt[:, tgt_idx, :]
    return _pair_error_value(
        pred_sel,
        tgt_sel,
        fn=fn,
        group_size=group_size,
        chamfer_loss=chamfer_loss,
    )


def _copy_baseline_error_value(
    tgt: torch.Tensor,
    tgt_idx: torch.Tensor,
    *,
    fn: str,
    group_size: int = 32,
    chamfer_loss: Optional[nn.Module] = None,
) -> float:
    if int(tgt_idx.numel()) == 0:
        return float("nan")
    valid = tgt_idx > 0
    if not bool(valid.any()):
        return float("nan")
    cur_idx = tgt_idx[valid]
    prev_idx = cur_idx - 1
    pred_sel = tgt[:, prev_idx, :]
    tgt_sel = tgt[:, cur_idx, :]
    return _pair_error_value(
        pred_sel,
        tgt_sel,
        fn=fn,
        group_size=group_size,
        chamfer_loss=chamfer_loss,
    )


def _compute_recon_diag(
    *,
    objective: str,
    pred_ctx: torch.Tensor,
    pred_q: torch.Tensor,
    pred_a: torch.Tensor,
    ctx_target: torch.Tensor,
    qry_target: torch.Tensor,
    ans_target: torch.Tensor,
    ctx_pred_idx: torch.Tensor,
    ctx_tgt_idx: torch.Tensor,
    q_pred_idx: torch.Tensor,
    q_tgt_idx: torch.Tensor,
    a_pred_idx: torch.Tensor,
    a_tgt_idx: torch.Tensor,
    group_size: int,
    chamfer_loss: Optional[nn.Module],
) -> dict[str, float]:
    ctx_fn = "chamfer" if str(objective) == "recon_chamfer" else "mse"

    recon_ctx_err = _masked_index_error_value(
        pred_ctx,
        ctx_target,
        ctx_pred_idx,
        ctx_tgt_idx,
        fn=ctx_fn,
        group_size=group_size,
        chamfer_loss=chamfer_loss,
    )
    recon_q_err = _masked_index_error_value(
        pred_q,
        qry_target,
        q_pred_idx,
        q_tgt_idx,
        fn="mse",
    )
    recon_a_err = _masked_index_error_value(
        pred_a,
        ans_target,
        a_pred_idx,
        a_tgt_idx,
        fn="mse",
    )

    copy_ctx_err = _copy_baseline_error_value(
        ctx_target,
        ctx_tgt_idx,
        fn=ctx_fn,
        group_size=group_size,
        chamfer_loss=chamfer_loss,
    )
    copy_q_err = _copy_baseline_error_value(qry_target, q_tgt_idx, fn="mse")
    copy_a_err = _copy_baseline_error_value(ans_target, a_tgt_idx, fn="mse")

    return {
        "recon_ctx_err": recon_ctx_err,
        "recon_q_err": recon_q_err,
        "recon_a_err": recon_a_err,
        "copy_ctx_err": copy_ctx_err,
        "copy_q_err": copy_q_err,
        "copy_a_err": copy_a_err,
        "lift_ctx": (copy_ctx_err - recon_ctx_err),
        "lift_q": (copy_q_err - recon_q_err),
        "lift_a": (copy_a_err - recon_a_err),
    }


def _load_chamfer_module(metric: str) -> nn.Module:
    """Load Point-MAE chamfer extension (L1/L2/L1+L2)."""
    repo_root = Path(__file__).resolve().parents[2]
    chamfer_dir = repo_root / "Point-MAE" / "extensions" / "chamfer_dist"
    init_py = chamfer_dir / "__init__.py"
    # Point-MAE's chamfer package uses `import chamfer` (absolute import).
    # Add only the extension directory to avoid shadowing HuggingFace `datasets`.
    chamfer_path = str(chamfer_dir)
    if chamfer_path not in sys.path:
        sys.path.insert(0, chamfer_path)
    try:
        spec = importlib.util.spec_from_file_location("pointmae_chamfer_dist", str(init_py))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to create import spec for {init_py}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ChamferDistanceL1 = getattr(module, "ChamferDistanceL1")
        ChamferDistanceL2 = getattr(module, "ChamferDistanceL2")
    except Exception as e:
        raise RuntimeError(
            "Point-MAE chamfer extension is unavailable. "
            "Build 3D-NEPA/Point-MAE/extensions/chamfer_dist first."
        ) from e
    m = str(metric).lower()
    if m == "l1":
        return ChamferDistanceL1()
    if m == "l2":
        return ChamferDistanceL2()
    if m == "l12":
        class _ChamferL12(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = ChamferDistanceL1()
                self.l2 = ChamferDistanceL2()

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return self.l1(x, y) + self.l2(x, y)

        return _ChamferL12()
    raise ValueError(f"unknown recon_chamfer_metric={metric}")


def _infinite_loader(dl: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dl:
            yield batch


def _save_ckpt(
    path: Path,
    model: PatchTransformerNepa,
    recon_heads: Optional[ReconHeads],
    recon_generator: Optional[CausalTransformer],
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
    if recon_heads is not None:
        payload["recon_heads"] = recon_heads.state_dict()
    if recon_generator is not None:
        payload["recon_generator"] = recon_generator.state_dict()
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


def _nepa_target_mask(
    type_id: torch.Tensor | None,
    k: int,
    *,
    loss_mask_mode: str = "answer_and_point_context",
) -> torch.Tensor | None:
    if type_id is None:
        return None
    tgt_ty = type_id[:, k:]
    mode = str(loss_mask_mode)
    is_answer_point = torch.zeros_like(tgt_ty, dtype=torch.bool)
    is_query_point = torch.zeros_like(tgt_ty, dtype=torch.bool)
    for ty in A_POINT_TYPES:
        is_answer_point = is_answer_point | (tgt_ty == int(ty))
    for ty in Q_POINT_TYPES:
        is_query_point = is_query_point | (tgt_ty == int(ty))
    has_answer = bool(is_answer_point.any() or (tgt_ty == int(TYPE_A_RAY)).any())
    is_answer = is_answer_point | (tgt_ty == int(TYPE_A_RAY))
    is_point_context = is_query_point | (tgt_ty == int(TYPE_POINT)) | (tgt_ty == int(TYPE_RAY))
    non_special = (
        (tgt_ty != int(TYPE_BOS))
        & (tgt_ty != int(TYPE_EOS))
        & (tgt_ty != int(TYPE_SEP_CTX))
        & (tgt_ty != int(TYPE_SEP_QA))
        & (tgt_ty != int(TYPE_MISSING_RAY))
    )
    if mode == "answer_only_if_present":
        if has_answer:
            return is_answer
        return non_special
    if mode == "answer_and_point_context":
        if has_answer:
            return is_answer | is_point_context
        return is_point_context
    if mode == "non_special":
        return non_special
    raise ValueError(
        f"unknown loss_mask_mode={mode}. "
        "expected one of: answer_only_if_present, answer_and_point_context, non_special"
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
    *,
    loss_mask_mode: str = "answer_and_point_context",
) -> dict[str, float] | None:
    k = max(1, int(k))
    if target_seq.size(1) <= k or z_hat.size(1) <= k:
        return None

    pred = z_hat[:, :-k, :].detach()
    tgt = target_seq[:, k:, :].detach()
    prev = target_seq[:, :-k, :].detach()

    mask = _nepa_target_mask(type_id, k, loss_mask_mode=str(loss_mask_mode))
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


def _center_triplet_for_nepa(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    prev: torch.Tensor,
    type_id: torch.Tensor | None,
    k: int,
    mask: torch.Tensor | None,
    *,
    center_mode: str,
    center_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(center_mode)
    alpha = float(center_alpha)
    if mode == "none" or alpha <= 0.0:
        return pred, tgt, prev

    bsz, tlen, dim = pred.shape
    if tlen <= 0:
        return pred, tgt, prev
    if mask is None:
        row_mask = torch.ones((bsz, tlen), device=pred.device, dtype=torch.bool)
    else:
        row_mask = mask.to(device=pred.device, dtype=torch.bool)

    w = row_mask.to(dtype=tgt.dtype).unsqueeze(-1)
    denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu_shape = (tgt * w).sum(dim=1, keepdim=True) / denom  # (B,1,D)
    mu_tok = mu_shape.expand(-1, tlen, -1)

    if mode == "segment" and type_id is not None:
        tgt_ty = type_id[:, k:].to(device=pred.device, dtype=torch.long)
        is_answer_point = torch.zeros_like(tgt_ty, dtype=torch.bool)
        is_query_point = torch.zeros_like(tgt_ty, dtype=torch.bool)
        for ty in A_POINT_TYPES:
            is_answer_point = is_answer_point | (tgt_ty == int(ty))
        for ty in Q_POINT_TYPES:
            is_query_point = is_query_point | (tgt_ty == int(ty))
        is_answer = is_answer_point | (tgt_ty == int(TYPE_A_RAY))
        is_query = is_query_point | (tgt_ty == int(TYPE_POINT)) | (tgt_ty == int(TYPE_RAY))
        is_query = is_query & row_mask
        is_answer = is_answer & row_mask

        q_w = is_query.to(dtype=tgt.dtype).unsqueeze(-1)
        q_den = q_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu_q = (tgt * q_w).sum(dim=1, keepdim=True) / q_den
        a_w = is_answer.to(dtype=tgt.dtype).unsqueeze(-1)
        a_den = a_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        mu_a = (tgt * a_w).sum(dim=1, keepdim=True) / a_den

        mu_tok = torch.where(
            is_query.unsqueeze(-1),
            mu_q.expand(-1, tlen, -1),
            mu_tok,
        )
        mu_tok = torch.where(
            is_answer.unsqueeze(-1),
            mu_a.expand(-1, tlen, -1),
            mu_tok,
        )

    mu_tok = alpha * mu_tok.detach()
    return pred - mu_tok, tgt - mu_tok, prev - mu_tok


def _compute_nepa_cosine_loss_and_diag(
    target_seq: torch.Tensor,
    z_hat: torch.Tensor,
    type_id: torch.Tensor | None,
    k: int,
    *,
    loss_mask_mode: str = "answer_and_point_context",
    center_mode: str = "none",
    center_alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    k = max(1, int(k))
    if target_seq.size(1) <= k or z_hat.size(1) <= k:
        zero = z_hat.sum() * 0.0
        nan = float("nan")
        return zero, {
            "k": float(k),
            "cos_tgt": nan,
            "cos_prev": nan,
            "gap": nan,
            "copy_win": nan,
            "center_mode": str(center_mode),
            "center_alpha": float(center_alpha),
        }

    pred = z_hat[:, :-k, :]
    tgt = target_seq[:, k:, :].detach()
    prev = target_seq[:, :-k, :].detach()
    mask = _nepa_target_mask(type_id, k, loss_mask_mode=str(loss_mask_mode))
    pred_u, tgt_u, prev_u = _center_triplet_for_nepa(
        pred,
        tgt,
        prev,
        type_id,
        k,
        mask,
        center_mode=str(center_mode),
        center_alpha=float(center_alpha),
    )

    cos_tgt = F.cosine_similarity(pred_u, tgt_u, dim=-1)
    if mask is None:
        loss = 1.0 - cos_tgt.mean()
    elif bool(mask.any()):
        loss = 1.0 - cos_tgt[mask].mean()
    else:
        loss = pred.sum() * 0.0

    cos_prev = F.cosine_similarity(pred_u, prev_u, dim=-1)
    cos_tgt_m = _masked_mean(cos_tgt, mask)
    cos_prev_m = _masked_mean(cos_prev, mask)
    if mask is None:
        cmp = cos_prev >= cos_tgt
    elif bool(mask.any()):
        cmp = (cos_prev >= cos_tgt)[mask]
    else:
        cmp = None
    copy_win = float(cmp.float().mean().item()) if cmp is not None else float("nan")

    return loss, {
        "k": float(k),
        "cos_tgt": float(cos_tgt_m),
        "cos_prev": float(cos_prev_m),
        "gap": float(cos_tgt_m - cos_prev_m),
        "copy_win": float(copy_win),
        "center_mode": str(center_mode),
        "center_alpha": float(center_alpha),
    }


def _compute_infonce_loss_and_diag(
    z_hat: torch.Tensor,
    target_seq: torch.Tensor,
    type_id: torch.Tensor | None,
    k: int,
    *,
    tau: float,
    loss_mask_mode: str = "answer_and_point_context",
) -> tuple[torch.Tensor, dict[str, float]]:
    """NEPA-style k-step InfoNCE over target tokens in each sample."""
    k = max(1, int(k))
    if target_seq.size(1) <= k or z_hat.size(1) <= k:
        zero = z_hat.sum() * 0.0
        nan = float("nan")
        return zero, {
            "tau": float(tau),
            "pos_cos": nan,
            "neg_cos": nan,
            "margin": nan,
            "pos_logit": nan,
            "neg_logit": nan,
            "r1": nan,
            "n_valid": 0.0,
        }

    pred = z_hat[:, :-k, :]
    tgt = target_seq[:, k:, :].detach()
    bsz, tlen, _ = pred.shape

    row_mask = _nepa_target_mask(type_id, k, loss_mask_mode=str(loss_mask_mode))
    if row_mask is None:
        row_mask = torch.ones((bsz, tlen), device=pred.device, dtype=torch.bool)
    else:
        row_mask = row_mask.to(device=pred.device, dtype=torch.bool)
    col_mask = row_mask

    pred_n = F.normalize(pred.float(), dim=-1)
    tgt_n = F.normalize(tgt.float(), dim=-1)
    sim = torch.bmm(pred_n, tgt_n.transpose(1, 2))  # (B,T,T), cosine logits before temperature
    logits = sim / float(tau)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~col_mask.unsqueeze(1), neg_inf)

    labels = torch.arange(tlen, device=pred.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)

    if bool(row_mask.any()):
        loss = F.cross_entropy(logits[row_mask], labels[row_mask])
    else:
        loss = pred.sum() * 0.0

    pos_cos = _masked_mean(sim.diagonal(dim1=1, dim2=2), row_mask)
    pos_logit = _masked_mean(logits.diagonal(dim1=1, dim2=2), row_mask)

    eye = torch.eye(tlen, device=pred.device, dtype=torch.bool).unsqueeze(0)
    neg_mask = (~eye) & col_mask.unsqueeze(1) & row_mask.unsqueeze(-1)
    if bool(neg_mask.any()):
        neg_cos = float(sim[neg_mask].mean().item())
        neg_logit = float(logits[neg_mask].mean().item())
    else:
        neg_cos = float("nan")
        neg_logit = float("nan")

    if (pos_cos == pos_cos) and (neg_cos == neg_cos):
        margin = float(pos_cos - neg_cos)
    else:
        margin = float("nan")

    top1 = logits.argmax(dim=-1)
    if bool(row_mask.any()):
        r1 = float((top1[row_mask] == labels[row_mask]).float().mean().item())
        n_valid = float(int(row_mask.sum().item()))
    else:
        r1 = float("nan")
        n_valid = 0.0

    return loss, {
        "tau": float(tau),
        "pos_cos": float(pos_cos),
        "neg_cos": float(neg_cos),
        "margin": float(margin),
        "pos_logit": float(pos_logit),
        "neg_logit": float(neg_logit),
        "r1": float(r1),
        "n_valid": float(n_valid),
    }


def _build_target_sequence(
    model_raw: PatchTransformerNepa,
    out: PatchNepaOutput,
    *,
    loss_target_mode: str,
    center_target_alpha: float,
) -> torch.Tensor:
    mode = str(loss_target_mode)
    if mode == "full_z":
        return out.z
    if mode == "content_tokens":
        return out.tokens
    if mode == "content_plus_center":
        alpha = float(center_target_alpha)
        if abs(alpha) < 1e-12:
            return out.tokens
        center_mlp = getattr(model_raw, "center_mlp", None)
        if center_mlp is None:
            raise RuntimeError("loss_target_mode=content_plus_center requires pos_mode=center_mlp.")
        center_term = center_mlp(out.centers_xyz)
        return out.tokens + (alpha * center_term)
    raise ValueError(
        f"unknown loss_target_mode={mode}. expected one of: full_z, content_tokens, content_plus_center"
    )


def _variance_covariance_loss(
    z_flat: torch.Tensor,
    *,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg-like variance/covariance regularizer over flattened embeddings."""
    if z_flat.numel() == 0 or int(z_flat.shape[0]) < 2:
        zero = z_flat.new_zeros(())
        nan = zero.new_full((), float("nan"))
        return zero, zero, nan, nan, nan

    z = z_flat.float()
    z_centered = z - z.mean(dim=0, keepdim=True)

    std = torch.sqrt(z_centered.pow(2).mean(dim=0) + float(eps))
    var_loss = torch.relu(float(gamma) - std).mean()

    denom = max(1, int(z_centered.shape[0]) - 1)
    cov = (z_centered.transpose(0, 1) @ z_centered) / float(denom)
    cov_off = cov - torch.diag_embed(torch.diagonal(cov, dim1=0, dim2=1))
    cov_loss = cov_off.pow(2).mean()

    std_mean = std.mean().detach()
    std_min = std.min().detach()
    cov_offdiag_rms = torch.sqrt(cov_off.pow(2).mean()).detach()
    return var_loss, cov_loss, std_mean, std_min, cov_offdiag_rms


def _variance_covariance_loss_intra_shape(
    z: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg-like var/cov loss computed per sample across token dimension."""
    if z.dim() != 3:
        raise ValueError(f"expected z with shape (B,T,D), got {tuple(z.shape)}")
    bsz, seq_len, dim = int(z.shape[0]), int(z.shape[1]), int(z.shape[2])
    if bsz == 0 or seq_len == 0 or dim == 0:
        zero = z.new_zeros(())
        nan = zero.new_full((), float("nan"))
        return zero, zero, nan, nan, nan

    if mask is None:
        mask = torch.ones((bsz, seq_len), device=z.device, dtype=torch.bool)
    else:
        if mask.shape[:2] != (bsz, seq_len):
            raise ValueError(
                f"intra-shape reg mask shape mismatch: mask={tuple(mask.shape)} vs z={(bsz, seq_len, dim)}"
            )
        mask = mask.to(device=z.device, dtype=torch.bool)

    var_terms: list[torch.Tensor] = []
    cov_terms: list[torch.Tensor] = []
    std_mean_terms: list[torch.Tensor] = []
    std_min_terms: list[torch.Tensor] = []
    cov_rms_terms: list[torch.Tensor] = []

    for bi in range(bsz):
        zi = z[bi][mask[bi]]
        if int(zi.shape[0]) < 2:
            continue
        zi = zi.float()
        zi = zi - zi.mean(dim=0, keepdim=True)

        std = torch.sqrt(zi.pow(2).mean(dim=0) + float(eps))
        var_terms.append(torch.relu(float(gamma) - std).mean())

        denom = max(1, int(zi.shape[0]) - 1)
        cov = (zi.transpose(0, 1) @ zi) / float(denom)
        cov_off = cov - torch.diag_embed(torch.diagonal(cov, dim1=0, dim2=1))
        cov_terms.append(cov_off.pow(2).mean())

        std_mean_terms.append(std.mean().detach())
        std_min_terms.append(std.min().detach())
        cov_rms_terms.append(torch.sqrt(cov_off.pow(2).mean()).detach())

    if len(var_terms) == 0:
        zero = z.new_zeros(())
        nan = zero.new_full((), float("nan"))
        return zero, zero, nan, nan, nan

    var_loss = torch.stack(var_terms).mean()
    cov_loss = torch.stack(cov_terms).mean()
    std_mean = torch.stack(std_mean_terms).mean()
    std_min = torch.stack(std_min_terms).mean()
    cov_offdiag_rms = torch.stack(cov_rms_terms).mean()
    return var_loss, cov_loss, std_mean, std_min, cov_offdiag_rms


def _regularization_loss(
    reg_source: torch.Tensor,
    reg_mask: torch.Tensor | None,
    *,
    scope: str,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(scope)
    if mode == "batch":
        if reg_mask is None:
            reg_flat = reg_source.reshape(-1, reg_source.shape[-1])
        else:
            reg_flat = reg_source[reg_mask]
        return _variance_covariance_loss(reg_flat, gamma=gamma, eps=eps)
    if mode == "intra_shape":
        return _variance_covariance_loss_intra_shape(reg_source, reg_mask, gamma=gamma, eps=eps)
    raise ValueError(f"unknown reg_scope={mode}. expected one of: batch, intra_shape")


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


def _recon_loss_mode_uses_partial_heads(mode: str) -> bool:
    return str(mode) in {
        "pointgpt_ctx_only",
        "answer_only",
        "context_plus_answer",
        "query_plus_answer",
    }


def main() -> None:
    args = parse_args()
    objective = str(args.pretrain_objective)
    ddp_find_unused = (
        objective in {"recon_mse", "recon_chamfer"}
        and _recon_loss_mode_uses_partial_heads(str(args.recon_loss_mode))
    )
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.grad_accum),
        kwargs_handlers=[ddp_kwargs],
    )
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
    chamfer_module: Optional[nn.Module] = None
    if objective == "recon_chamfer":
        chamfer_module = _load_chamfer_module(str(args.recon_chamfer_metric))
    recon_heads: Optional[ReconHeads] = None
    recon_generator: Optional[CausalTransformer] = None
    trainable_params = list(model.parameters())
    if objective in {"recon_mse", "recon_chamfer"}:
        recon_heads = ReconHeads(
            d_model=int(args.d_model),
            group_size=int(args.group_size),
            answer_in_dim=int(answer_in_dim),
        )
        trainable_params += list(recon_heads.parameters())
        if int(args.recon_generator_depth) > 0:
            recon_generator = CausalTransformer(
                d_model=int(args.d_model),
                nhead=int(args.n_heads),
                num_layers=int(args.recon_generator_depth),
                mlp_ratio=float(args.mlp_ratio),
                dropout=float(args.dropout),
                drop_path=float(args.drop_path_rate),
                backbone_impl=str(args.backbone_mode),
                qk_norm=bool(int(args.qk_norm)),
                qk_norm_affine=bool(int(args.qk_norm_affine)),
                qk_norm_bias=bool(int(args.qk_norm_bias)),
                layerscale_value=float(args.layerscale_value),
                rope_theta=float(args.rope_theta),
                hidden_dropout_prob=float(args.dropout),
                attention_probs_dropout_prob=float(args.dropout),
                use_gated_mlp=bool(int(args.use_gated_mlp)),
                hidden_act=str(args.hidden_act),
                final_layernorm=True,
            )
            trainable_params += list(recon_generator.parameters())
        optimizer = optim.AdamW(
            trainable_params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        if recon_generator is not None:
            model, recon_heads, recon_generator, optimizer, dl = accelerator.prepare(
                model, recon_heads, recon_generator, optimizer, dl
            )
        else:
            model, recon_heads, optimizer, dl = accelerator.prepare(model, recon_heads, optimizer, dl)
        recon_heads_raw: Optional[ReconHeads] = accelerator.unwrap_model(recon_heads)
        recon_generator_raw: Optional[CausalTransformer] = (
            accelerator.unwrap_model(recon_generator) if recon_generator is not None else None
        )
        trainable_params_for_clip = list(model.parameters()) + list(recon_heads.parameters())
        if recon_generator is not None:
            trainable_params_for_clip += list(recon_generator.parameters())
    else:
        optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
        recon_heads_raw = None
        recon_generator_raw = None
        trainable_params_for_clip = list(model.parameters())
    model_raw = accelerator.unwrap_model(model)
    try:
        steps_per_epoch = max(1, int(len(dl)))
    except Exception:
        # Fallback for dataloader wrappers without __len__.
        est_samples = int(info.get("num_samples", 0))
        est_batch = max(1, int(args.batch_size))
        steps_per_epoch = max(1, est_samples // est_batch)
    if int(args.epochs) > 0:
        args.max_steps = int(max(1, int(args.epochs) * int(steps_per_epoch)))
    warmup_steps = max(0, _resolve_warmup_steps(args))
    dual_mask_warmup_steps = max(0, int(round(float(args.dual_mask_warmup_frac) * float(args.max_steps))))
    center_warmup_steps = max(0, int(round(float(args.nepa_center_warmup_frac) * float(args.max_steps))))
    scheduler: optim.lr_scheduler.LambdaLR | None = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda st: _scheduler_scale(int(st), args, warmup_steps),
        )

    if accelerator.is_main_process:
        accelerator.print(f"[token_pretrain] answer_in_dim={answer_in_dim}")
        accelerator.print(f"[token_pretrain] pretrain_objective={objective}")
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
            f"[token_pretrain] loss_target_mode={str(args.loss_target_mode)} "
            f"center_target_alpha={float(args.center_target_alpha):.4f}"
        )
        accelerator.print(
            f"[token_pretrain] recon weights "
            f"ctx={float(args.recon_ctx_weight):.3f} "
            f"q={float(args.recon_q_weight):.3f} "
            f"a={float(args.recon_a_weight):.3f} "
            f"loss_mode={str(args.recon_loss_mode)} "
            f"chamfer_metric={str(args.recon_chamfer_metric)} "
            f"generator_depth={int(args.recon_generator_depth)}"
        )
        accelerator.print(f"[token_pretrain] ddp_find_unused_parameters={bool(ddp_find_unused)}")
        if objective == "infonce":
            accelerator.print(f"[token_pretrain] infonce tau={float(args.infonce_tau):.4f}")
        accelerator.print(
            f"[token_pretrain] reg var_w={float(args.reg_var_weight):.4f} "
            f"cov_w={float(args.reg_cov_weight):.4f} "
            f"gamma={float(args.reg_var_gamma):.4f} eps={float(args.reg_var_eps):.1e} "
            f"scope={str(args.reg_scope)} source={str(args.reg_source)}"
        )
        accelerator.print(f"[token_pretrain] loss_mask_mode={str(args.loss_mask_mode)}")
        accelerator.print(
            f"[token_pretrain] optimizer lr={float(args.lr):.3e} wd={float(args.weight_decay):.3f} "
            f"scheduler={str(args.lr_scheduler)} warmup_steps={int(warmup_steps)} "
            f"warmup_ratio={float(args.warmup_ratio):.4f} min_lr={float(args.min_lr):.2e}"
        )
        accelerator.print(
            f"[token_pretrain] dual_mask near={float(args.dual_mask_near):.4f} "
            f"far={float(args.dual_mask_far):.4f} window={int(args.dual_mask_window)} "
            f"type_aware={bool(int(args.dual_mask_type_aware))} "
            f"mode={str(args.dual_mask_mode)} keep_prefix={int(args.dual_mask_keep_prefix)} "
            f"column_ratio={float(args.dual_mask_column_ratio):.4f} "
            f"warmup_frac={float(args.dual_mask_warmup_frac):.4f} "
            f"warmup_steps={int(dual_mask_warmup_steps)}"
        )
        accelerator.print(
            f"[token_pretrain] nepa_center mode={str(args.nepa_center_mode)} "
            f"warmup_frac={float(args.nepa_center_warmup_frac):.4f} "
            f"warmup_steps={int(center_warmup_steps)}"
        )
        accelerator.print(f"[token_pretrain] steps_per_epoch={int(steps_per_epoch)}")
        accelerator.print(
            f"[token_pretrain] epochs={int(args.epochs)} effective_max_steps={int(args.max_steps)}"
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
        primitive = batch.get("primitive", None)
        if qry_xyz is None or ans_feat is None:
            raise RuntimeError("batch missing qry_xyz/ans_feat; ensure v2 config uses return_qry=true.")
        surf_xyz = surf_xyz.to(accelerator.device, non_blocking=True)
        qry_xyz = qry_xyz.to(accelerator.device, non_blocking=True)
        ans_feat = ans_feat.to(accelerator.device, non_blocking=True)
        if dual_mask_warmup_steps > 0:
            dm_ramp = min(1.0, float(step + 1) / float(max(1, dual_mask_warmup_steps)))
        else:
            dm_ramp = 1.0
        dm_near = float(args.dual_mask_near) * dm_ramp
        dm_far = float(args.dual_mask_far) * dm_ramp
        dm_col = float(args.dual_mask_column_ratio) * dm_ramp
        if center_warmup_steps > 0:
            center_ramp = min(1.0, float(step + 1) / float(max(1, center_warmup_steps)))
        else:
            center_ramp = 1.0

        with accelerator.accumulate(model):
            seq = build_qa_sequence(
                model_raw,
                surf_xyz,
                qry_xyz,
                ans_feat,
                primitive=primitive,
                token_qa_layout=str(args.token_qa_layout),
            )
            out = _forward_tokens_wrapped(
                model,
                tokens=seq.tokens,
                type_id=seq.type_id,
                centers_xyz=seq.centers,
                is_causal=True,
                q_mask_prob=float(args.q_mask_prob),
                dual_mask_near=dm_near,
                dual_mask_far=dm_far,
                dual_mask_window=int(args.dual_mask_window),
                dual_mask_type_aware=bool(int(args.dual_mask_type_aware)),
                dual_mask_mode=str(args.dual_mask_mode),
                dual_mask_keep_prefix=int(args.dual_mask_keep_prefix),
                dual_mask_column_ratio=dm_col,
            )
            diag = None
            diag_raw = None
            infonce_diag = None
            recon_diag = None
            target = _build_target_sequence(
                model_raw,
                out,
                loss_target_mode=str(args.loss_target_mode),
                center_target_alpha=float(args.center_target_alpha),
            )
            diag_raw = _compute_copy_diag(
                target,
                out.z_hat,
                out.type_id,
                int(args.skip_k),
                loss_mask_mode=str(args.loss_mask_mode),
            )
            diag = diag_raw
            loss_infonce = target.new_zeros(())
            if objective == "nepa_cosine":
                loss_nepa, diag = _compute_nepa_cosine_loss_and_diag(
                    target,
                    out.z_hat,
                    out.type_id,
                    loss_mask_mode=str(args.loss_mask_mode),
                    k=int(args.skip_k),
                    center_mode=str(args.nepa_center_mode),
                    center_alpha=float(center_ramp),
                )
                loss_recon_ctx = loss_nepa.new_zeros(())
                loss_recon_q = loss_nepa.new_zeros(())
                loss_recon_a = loss_nepa.new_zeros(())
                loss_main = loss_nepa
            elif objective == "infonce":
                with torch.no_grad():
                    loss_nepa = model_raw.nepa_loss(
                        out.z,
                        out.z_hat,
                        out.type_id,
                        skip_k=int(args.skip_k),
                        target=target,
                        loss_mask_mode=str(args.loss_mask_mode),
                    )
                loss_infonce, infonce_diag = _compute_infonce_loss_and_diag(
                    out.z_hat,
                    target,
                    out.type_id,
                    int(args.skip_k),
                    tau=float(args.infonce_tau),
                    loss_mask_mode=str(args.loss_mask_mode),
                )
                loss_recon_ctx = loss_nepa.new_zeros(())
                loss_recon_q = loss_nepa.new_zeros(())
                loss_recon_a = loss_nepa.new_zeros(())
                loss_main = loss_infonce
            elif objective in {"recon_mse", "recon_chamfer"}:
                with torch.no_grad():
                    loss_nepa = model_raw.nepa_loss(
                        out.z,
                        out.z_hat,
                        out.type_id,
                        skip_k=int(args.skip_k),
                        target=target,
                        loss_mask_mode=str(args.loss_mask_mode),
                    )
                k = int(args.skip_k)
                pred_h = out.h[:, :-k, :]
                ctx_abs, q_abs, a_abs = _sequence_absolute_positions(
                    seq.n_ctx,
                    seq.n_qry,
                    seq.token_qa_layout,
                    device=pred_h.device,
                )
                base_q_idx = torch.arange(seq.n_qry, device=pred_h.device, dtype=torch.long)
                base_a_idx = torch.arange(seq.n_qry, device=pred_h.device, dtype=torch.long)

                valid_ctx = ctx_abs >= k
                ctx_pred_idx = (ctx_abs[valid_ctx] - k).long()
                ctx_tgt_idx = (ctx_abs[valid_ctx] - 1).long()

                valid_q = q_abs >= k
                q_pred_idx = (q_abs[valid_q] - k).long()
                q_tgt_idx = base_q_idx[valid_q]

                valid_a = a_abs >= k
                a_pred_idx = (a_abs[valid_a] - k).long()
                a_tgt_idx = base_a_idx[valid_a]

                ctx_target = _gather_rel_xyz_flat(surf_xyz, seq.ctx_group_idx, seq.ctx_centers)
                if recon_heads is None:
                    raise RuntimeError(
                        "reconstruction objective requested but recon_heads is None "
                        "(internal setup error)"
                    )
                if recon_generator is not None:
                    pred_h = recon_generator(
                        pred_h,
                        is_causal=True,
                        type_id=out.type_id[:, :-k],
                        dual_mask_near=dm_near,
                        dual_mask_far=dm_far,
                        dual_mask_window=int(args.dual_mask_window),
                        dual_mask_type_aware=bool(int(args.dual_mask_type_aware)),
                        dual_mask_mode=str(args.dual_mask_mode),
                        dual_mask_keep_prefix=int(args.dual_mask_keep_prefix),
                        dual_mask_column_ratio=dm_col,
                    )
                pred_ctx, pred_q, pred_a = recon_heads(pred_h)

                ctx_loss_fn = "mse"
                if objective == "recon_chamfer":
                    ctx_loss_fn = "chamfer"
                loss_recon_ctx = _masked_index_loss(
                    pred_ctx,
                    ctx_target,
                    ctx_pred_idx,
                    ctx_tgt_idx,
                    fn=ctx_loss_fn,
                    group_size=int(args.group_size),
                    chamfer_loss=chamfer_module,
                )
                loss_recon_q = _masked_index_loss(
                    pred_q,
                    qry_xyz,
                    q_pred_idx,
                    q_tgt_idx,
                    fn="mse",
                )
                loss_recon_a = _masked_index_loss(
                    pred_a,
                    ans_feat,
                    a_pred_idx,
                    a_tgt_idx,
                    fn="mse",
                )
                recon_loss_mode = str(args.recon_loss_mode)
                if recon_loss_mode == "pointgpt_ctx_only":
                    # PointGPT-equivalent axis: optimize only patch-coordinate reconstruction.
                    loss_main = loss_recon_ctx
                elif recon_loss_mode == "answer_only":
                    loss_main = float(args.recon_a_weight) * loss_recon_a
                elif recon_loss_mode == "context_plus_answer":
                    loss_main = (
                        float(args.recon_ctx_weight) * loss_recon_ctx
                        + float(args.recon_a_weight) * loss_recon_a
                    )
                elif recon_loss_mode == "query_plus_answer":
                    loss_main = (
                        float(args.recon_q_weight) * loss_recon_q
                        + float(args.recon_a_weight) * loss_recon_a
                    )
                elif recon_loss_mode == "composite":
                    loss_main = (
                        float(args.recon_ctx_weight) * loss_recon_ctx
                        + float(args.recon_q_weight) * loss_recon_q
                        + float(args.recon_a_weight) * loss_recon_a
                    )
                else:
                    raise ValueError(
                        f"unknown recon_loss_mode={recon_loss_mode}. "
                        "expected one of: composite, pointgpt_ctx_only, "
                        "answer_only, context_plus_answer, query_plus_answer"
                    )
                with torch.no_grad():
                    recon_diag = _compute_recon_diag(
                        objective=objective,
                        pred_ctx=pred_ctx.detach(),
                        pred_q=pred_q.detach(),
                        pred_a=pred_a.detach(),
                        ctx_target=ctx_target.detach(),
                        qry_target=qry_xyz.detach(),
                        ans_target=ans_feat.detach(),
                        ctx_pred_idx=ctx_pred_idx,
                        ctx_tgt_idx=ctx_tgt_idx,
                        q_pred_idx=q_pred_idx,
                        q_tgt_idx=q_tgt_idx,
                        a_pred_idx=a_pred_idx,
                        a_tgt_idx=a_tgt_idx,
                        group_size=int(args.group_size),
                        chamfer_loss=chamfer_module,
                    )
            else:
                raise ValueError(
                    f"unknown pretrain_objective={objective}. "
                    "expected one of: nepa_cosine, infonce, recon_mse, recon_chamfer"
                )
            reg_mask = _nepa_target_mask(
                out.type_id,
                int(args.skip_k),
                loss_mask_mode=str(args.loss_mask_mode),
            )
            if str(args.reg_source) == "hidden":
                reg_source = out.h[:, int(args.skip_k) :, :]
            elif str(args.reg_source) == "target":
                reg_source = target[:, int(args.skip_k) :, :]
            else:
                raise ValueError(
                    f"unknown reg_source={args.reg_source}. expected one of: target, hidden"
                )
            var_loss, cov_loss, reg_std_mean, reg_std_min, reg_cov_offdiag_rms = _regularization_loss(
                reg_source,
                reg_mask,
                scope=str(args.reg_scope),
                gamma=float(args.reg_var_gamma),
                eps=float(args.reg_var_eps),
            )
            loss = (
                loss_main
                + (float(args.reg_var_weight) * var_loss)
                + (float(args.reg_cov_weight) * cov_loss)
            )
            accelerator.backward(loss)
            if float(args.max_grad_norm) > 0:
                accelerator.clip_grad_norm_(trainable_params_for_clip, float(args.max_grad_norm))
            optimizer.step()
            if scheduler is not None and accelerator.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            step += 1
            if accelerator.is_main_process:
                loss_nepa_val = float(loss_nepa.item())
                loss_infonce_val = float(loss_infonce.item())
                loss_main_val = float(loss_main.item())
                loss_recon_ctx_val = float(loss_recon_ctx.item())
                loss_recon_q_val = float(loss_recon_q.item())
                loss_recon_a_val = float(loss_recon_a.item())
                loss_val = float(loss.item())
                var_loss_val = float(var_loss.item())
                cov_loss_val = float(cov_loss.item())
                reg_std_mean_val = float(reg_std_mean.item()) if reg_std_mean.dim() == 0 else float("nan")
                reg_std_min_val = float(reg_std_min.item()) if reg_std_min.dim() == 0 else float("nan")
                reg_cov_offdiag_rms_val = (
                    float(reg_cov_offdiag_rms.item()) if reg_cov_offdiag_rms.dim() == 0 else float("nan")
                )
                # 2D ViT-NEPA reports pretrain objective as -cos.
                # PatchNEPA uses (1-cos), so convert by subtracting 1 for axis parity.
                loss_2d_equiv = loss_nepa_val - 1.0
                # PointGPT-equivalent scalar for reconstruction branch:
                # use context reconstruction term as the nearest objective proxy.
                if objective in {"recon_mse", "recon_chamfer"}:
                    loss_pointgpt_equiv = loss_recon_ctx_val
                    loss_pointgpt_equiv_x1k = loss_recon_ctx_val * 1000.0
                else:
                    loss_pointgpt_equiv = float("nan")
                    loss_pointgpt_equiv_x1k = float("nan")
                pbar.update(1)
                pbar.set_description(f"loss={loss_val:.4f}")
                if (step % diag_every == 0):
                    if objective == "nepa_cosine" and (diag is not None):
                        accelerator.print(
                            f"[step {step:06d}] "
                            f"loss_total={loss_val:.6f} "
                            f"loss_main={loss_main_val:.6f} "
                            f"loss_nepa={loss_nepa_val:.6f} "
                            f"loss_infonce={loss_infonce_val:.6f} "
                            f"loss_recon_ctx={loss_recon_ctx_val:.6f} "
                            f"loss_recon_q={loss_recon_q_val:.6f} "
                            f"loss_recon_a={loss_recon_a_val:.6f} "
                            f"loss_var={var_loss_val:.6f} "
                            f"loss_cov={cov_loss_val:.6f} "
                            f"loss2d={loss_2d_equiv:.6f} "
                            f"cos_tgt={_fmt_diag(float(diag['cos_tgt']))} "
                            f"cos_prev={_fmt_diag(float(diag['cos_prev']))} "
                            f"gap={_fmt_diag(float(diag['gap']))} "
                            f"copy_win={_fmt_diag(float(diag['copy_win']))}"
                        )
                    elif objective == "infonce" and (diag is not None) and (infonce_diag is not None):
                        accelerator.print(
                            f"[step {step:06d}] "
                            f"loss_total={loss_val:.6f} "
                            f"loss_main={loss_main_val:.6f} "
                            f"loss_nepa={loss_nepa_val:.6f} "
                            f"loss_infonce={loss_infonce_val:.6f} "
                            f"loss_recon_ctx={loss_recon_ctx_val:.6f} "
                            f"loss_recon_q={loss_recon_q_val:.6f} "
                            f"loss_recon_a={loss_recon_a_val:.6f} "
                            f"loss_var={var_loss_val:.6f} "
                            f"loss_cov={cov_loss_val:.6f} "
                            f"loss2d={loss_2d_equiv:.6f} "
                            f"pos_cos={_fmt_diag(float(infonce_diag['pos_cos']))} "
                            f"neg_cos={_fmt_diag(float(infonce_diag['neg_cos']))} "
                            f"margin={_fmt_diag(float(infonce_diag['margin']))} "
                            f"r1={_fmt_diag(float(infonce_diag['r1']))} "
                            f"cos_tgt={_fmt_diag(float(diag['cos_tgt']))} "
                            f"cos_prev={_fmt_diag(float(diag['cos_prev']))} "
                            f"gap={_fmt_diag(float(diag['gap']))} "
                            f"copy_win={_fmt_diag(float(diag['copy_win']))}"
                        )
                    elif objective in {"recon_mse", "recon_chamfer"} and (recon_diag is not None):
                        accelerator.print(
                            f"[step {step:06d}] "
                            f"loss_total={loss_val:.6f} "
                            f"loss_main={loss_main_val:.6f} "
                            f"loss_nepa={loss_nepa_val:.6f} "
                            f"loss_infonce={loss_infonce_val:.6f} "
                            f"loss_recon_ctx={loss_recon_ctx_val:.6f} "
                            f"loss_recon_q={loss_recon_q_val:.6f} "
                            f"loss_recon_a={loss_recon_a_val:.6f} "
                            f"loss_var={var_loss_val:.6f} "
                            f"loss_cov={cov_loss_val:.6f} "
                            f"loss_pgpt={loss_pointgpt_equiv:.6f} "
                            f"loss_pgpt_x1k={loss_pointgpt_equiv_x1k:.3f} "
                            f"copy_ctx={_fmt_diag(float(recon_diag['copy_ctx_err']))} "
                            f"copy_q={_fmt_diag(float(recon_diag['copy_q_err']))} "
                            f"copy_a={_fmt_diag(float(recon_diag['copy_a_err']))} "
                            f"lift_ctx={_fmt_diag(float(recon_diag['lift_ctx']))} "
                            f"lift_q={_fmt_diag(float(recon_diag['lift_q']))} "
                            f"lift_a={_fmt_diag(float(recon_diag['lift_a']))}"
                        )
                if wandb_run is not None and (step % wandb_log_every == 0):
                    epoch_float = float(step - 1) / float(max(1, steps_per_epoch))
                    wb = {
                        "train/loss": loss_val,
                        "train/step": int(step),
                        "train/epoch": epoch_float,
                    }
                    if objective in {"nepa_cosine", "infonce"}:
                        wb["train/loss_2d_equiv"] = loss_2d_equiv
                    if objective in {"recon_mse", "recon_chamfer"}:
                        wb["train/loss_pointgpt_equiv"] = loss_pointgpt_equiv
                        wb["train/loss_pointgpt_equiv_x1k"] = loss_pointgpt_equiv_x1k
                        wb["diag/recon_loss_mode_id"] = RECON_LOSS_MODE_TO_ID.get(
                            str(args.recon_loss_mode),
                            -1.0,
                        )
                    if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
                        wb["train/lr"] = float(optimizer.param_groups[0]["lr"])
                    if objective in {"nepa_cosine", "infonce"} and (diag is not None):
                        wb["diag/cos_tgt"] = float(diag["cos_tgt"])
                        wb["diag/cos_prev"] = float(diag["cos_prev"])
                        wb["diag/gap"] = float(diag["gap"])
                        wb["diag/copy_win"] = float(diag["copy_win"])
                        if "center_alpha" in diag:
                            wb["diag/center_alpha"] = float(diag["center_alpha"])
                        if "center_mode" in diag:
                            wb["diag/center_mode_id"] = (
                                0.0
                                if str(diag["center_mode"]) == "none"
                                else (1.0 if str(diag["center_mode"]) == "shape" else 2.0)
                            )
                    if objective in {"nepa_cosine", "infonce"} and (diag_raw is not None) and (diag_raw is not diag):
                        wb["diag/cos_tgt_raw"] = float(diag_raw["cos_tgt"])
                        wb["diag/cos_prev_raw"] = float(diag_raw["cos_prev"])
                        wb["diag/gap_raw"] = float(diag_raw["gap"])
                        wb["diag/copy_win_raw"] = float(diag_raw["copy_win"])
                    if objective == "infonce" and (infonce_diag is not None):
                        wb["diag/infonce_tau"] = float(infonce_diag["tau"])
                        wb["diag/infonce_pos_cos"] = float(infonce_diag["pos_cos"])
                        wb["diag/infonce_neg_cos"] = float(infonce_diag["neg_cos"])
                        wb["diag/infonce_margin"] = float(infonce_diag["margin"])
                        wb["diag/infonce_pos_logit"] = float(infonce_diag["pos_logit"])
                        wb["diag/infonce_neg_logit"] = float(infonce_diag["neg_logit"])
                        wb["diag/infonce_r1"] = float(infonce_diag["r1"])
                        wb["diag/infonce_n_valid"] = float(infonce_diag["n_valid"])
                    if objective != "nepa_cosine" and (recon_diag is not None):
                        wb["diag/recon_ctx_err"] = float(recon_diag["recon_ctx_err"])
                        wb["diag/recon_q_err"] = float(recon_diag["recon_q_err"])
                        wb["diag/recon_a_err"] = float(recon_diag["recon_a_err"])
                        wb["diag/copy_ctx_err"] = float(recon_diag["copy_ctx_err"])
                        wb["diag/copy_q_err"] = float(recon_diag["copy_q_err"])
                        wb["diag/copy_a_err"] = float(recon_diag["copy_a_err"])
                        wb["diag/recon_lift_ctx"] = float(recon_diag["lift_ctx"])
                        wb["diag/recon_lift_q"] = float(recon_diag["lift_q"])
                        wb["diag/recon_lift_a"] = float(recon_diag["lift_a"])
                    wb["diag/target_std_mean"] = reg_std_mean_val
                    wb["diag/target_std_min"] = reg_std_min_val
                    wb["diag/target_cov_offdiag_rms"] = reg_cov_offdiag_rms_val
                    wandb_run.log(wb, step=int(step))
            if int(args.save_every) > 0 and (step % int(args.save_every) == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(
                        save_root / f"ckpt_step{step}.pt",
                        model_raw,
                        recon_heads_raw,
                        recon_generator_raw,
                        optimizer,
                        scheduler,
                        step,
                        args,
                    )
    pbar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_ckpt(
            save_root / "ckpt_final.pt",
            model_raw,
            recon_heads_raw,
            recon_generator_raw,
            optimizer,
            scheduler,
            step,
            args,
        )
        accelerator.print(f"[done] saved to {save_root}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
