from __future__ import annotations

import argparse
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from ..data.dataset import ModelNet40QueryDataset, collate
from ..data.mixed_pretrain import build_mixed_pretrain
from ..data.modelnet40_index import list_npz
from ..models.query_nepa import QueryNepa
from ..token.tokenizer import (
    TYPE_BOS,
    TYPE_EOS,
    TYPE_MISSING_RAY,
    TYPE_POINT,
    TYPE_RAY,
    TYPE_A_POINT,
    TYPE_A_RAY,
)
from ..utils.seed import set_seed
from ..utils.ckpt_utils import load_state_dict_flexible, maybe_resize_pos_emb_in_state_dict


def build_token_mask(type_id, mask_ratio):
    """Sample per-sample random mask over non-BOS/EOS positions."""
    if mask_ratio <= 0.0:
        return torch.zeros_like(type_id, dtype=torch.bool)
    bsz, t = type_id.shape
    mask = torch.zeros_like(type_id, dtype=torch.bool)
    for b in range(bsz):
        valid = (type_id[b] != TYPE_BOS) & (type_id[b] != TYPE_EOS)
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        n = int(idx.numel())
        if n <= 0:
            continue
        k = int(n * float(mask_ratio))
        if k <= 0:
            continue
        perm = torch.randperm(n, device=type_id.device)[:k]
        mask[b, idx[perm]] = True
    return mask


def _ray_rank_hinge_loss(pred_t, gt_t, hit_mask, n_pairs: int, margin: float):
    """Pairwise ranking loss on ray depth order (smaller t should rank earlier)."""
    n_pairs = int(n_pairs)
    if n_pairs <= 0:
        return pred_t.new_tensor(0.0)

    bsz = int(pred_t.shape[0])
    total = pred_t.new_tensor(0.0)
    n_valid = 0
    for b in range(bsz):
        idx = torch.nonzero(hit_mask[b], as_tuple=False).flatten()
        m = int(idx.numel())
        if m < 2:
            continue
        ii = idx[torch.randint(0, m, (n_pairs,), device=pred_t.device)]
        jj = idx[torch.randint(0, m, (n_pairs,), device=pred_t.device)]
        ok = ii != jj
        if not bool(ok.any()):
            continue
        ii = ii[ok]
        jj = jj[ok]
        gt_diff = gt_t[b, ii] - gt_t[b, jj]
        sign = torch.sign(gt_diff)
        ok2 = sign != 0
        if not bool(ok2.any()):
            continue
        ii = ii[ok2]
        jj = jj[ok2]
        sign = sign[ok2]
        pred_diff = pred_t[b, ii] - pred_t[b, jj]
        loss_b = torch.relu(float(margin) - sign * pred_diff).mean()
        total = total + loss_b
        n_valid += 1
    if n_valid <= 0:
        return pred_t.new_tensor(0.0)
    return total / float(n_valid)


def _default_answer_mask(target_type: torch.Tensor) -> torch.Tensor:
    """Mask used by NEPA answer-only training and optional distillation."""
    valid = target_type != TYPE_MISSING_RAY
    has_answer_types = (target_type == TYPE_A_POINT).any() or (target_type == TYPE_A_RAY).any()
    if bool(has_answer_types):
        valid = valid & ((target_type == TYPE_A_POINT) | (target_type == TYPE_A_RAY))
    return valid


def _default_answer_input_mask(type_id: torch.Tensor) -> torch.Tensor:
    """Mask of input tokens that carry answer observations."""
    return (
        (type_id == TYPE_A_POINT)
        | (type_id == TYPE_A_RAY)
        | (type_id == TYPE_POINT)
        | (type_id == TYPE_RAY)
    )


def _drop_answer_observations(feat: torch.Tensor, type_id: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Randomly hide answer channels while keeping query channels intact.

    - Point answer: drop distance channel (10).
    - Ray answer: drop hit/t/normal/x_hit channels (0:3, 9, 11, 12:15).
    """
    p = float(drop_prob)
    if p <= 0.0:
        return feat
    out = feat.clone()
    ans_mask = _default_answer_input_mask(type_id)
    if not bool(ans_mask.any()):
        return out
    sample_mask = torch.rand(type_id.shape, device=type_id.device) < p
    drop_mask = ans_mask & sample_mask
    if not bool(drop_mask.any()):
        return out

    point_mask = drop_mask & ((type_id == TYPE_A_POINT) | (type_id == TYPE_POINT))
    ray_mask = drop_mask & ((type_id == TYPE_A_RAY) | (type_id == TYPE_RAY))

    if bool(point_mask.any()):
        out[..., 10] = torch.where(point_mask, torch.zeros_like(out[..., 10]), out[..., 10])
    if bool(ray_mask.any()):
        ray_mask_f = ray_mask.unsqueeze(-1)
        out[..., 0:3] = torch.where(ray_mask_f, torch.zeros_like(out[..., 0:3]), out[..., 0:3])
        out[..., 9] = torch.where(ray_mask, torch.zeros_like(out[..., 9]), out[..., 9])
        out[..., 11] = torch.where(ray_mask, torch.zeros_like(out[..., 11]), out[..., 11])
        out[..., 12:15] = torch.where(ray_mask_f, torch.zeros_like(out[..., 12:15]), out[..., 12:15])
    return out


def _hard_answer_topk_loss(
    token_loss: torch.Tensor,
    target_type: torch.Tensor,
    top_frac: float,
    min_tokens: int,
) -> torch.Tensor:
    """D: hard-query mining loss on the highest-error answer tokens."""
    valid = _default_answer_mask(target_type)
    if not bool(valid.any()):
        return token_loss.new_tensor(0.0)

    bsz = int(token_loss.shape[0])
    total = token_loss.new_tensor(0.0)
    n_valid = 0
    frac = max(0.0, min(1.0, float(top_frac)))
    min_tokens = max(1, int(min_tokens))
    for b in range(bsz):
        vals = token_loss[b][valid[b]]
        n = int(vals.numel())
        if n <= 0:
            continue
        k = max(int(np.ceil(frac * n)), min_tokens)
        k = min(k, n)
        if k <= 0:
            continue
        topk_vals = torch.topk(vals, k=k, largest=True, sorted=False).values
        total = total + topk_vals.mean()
        n_valid += 1
    if n_valid <= 0:
        return token_loss.new_tensor(0.0)
    return total / float(n_valid)


def _prune_optimizer_state_for_shape_mismatch(
    opt: optim.Optimizer,
    saved_opt_state: dict,
) -> tuple[dict, int, int]:
    """Drop per-parameter optimizer slots whose tensor shape mismatches current params.

    This is mainly used when `pos_emb` length changes across resume.
    We keep state for unchanged parameters and only reset mismatched ones.
    """
    pruned = copy.deepcopy(saved_opt_state)
    saved_groups = pruned.get("param_groups", [])
    cur_groups = opt.param_groups
    if len(saved_groups) != len(cur_groups):
        return pruned, 0, 0

    state = pruned.get("state", {})
    dropped = 0
    checked = 0
    for cur_g, sav_g in zip(cur_groups, saved_groups):
        cur_params = list(cur_g.get("params", []))
        sav_ids = list(sav_g.get("params", []))
        if len(cur_params) != len(sav_ids):
            continue
        for p_cur, sid in zip(cur_params, sav_ids):
            st = state.get(sid, None)
            if not isinstance(st, dict):
                continue
            checked += 1
            mismatch = False
            for v in st.values():
                if torch.is_tensor(v) and v.ndim > 0 and tuple(v.shape) != tuple(p_cur.shape):
                    mismatch = True
                    break
            if mismatch:
                state[sid] = {}
                dropped += 1
    pruned["state"] = state
    return pruned, dropped, checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, default="data/modelnet40_cache")
    ap.add_argument("--mix_config", type=str, default="", help="YAML config for mixed pretraining. If set, ignores --cache_root/--backend for dataset construction.")
    ap.add_argument("--mix_num_samples", type=int, default=0, help="override mix_num_samples in YAML (0=use YAML/default)")
    ap.add_argument("--mix_seed", type=int, default=0, help="override mix_seed in YAML (0=use YAML/default if provided)")
    ap.add_argument(
        "--backend",
        type=str,
        default="mesh",
        choices=["mesh", "pointcloud", "pointcloud_meshray", "pointcloud_noray", "voxel", "udfgrid"],
    )
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_point", type=int, default=512)
    ap.add_argument("--n_ray", type=int, default=512)
    ap.add_argument(
        "--max_len",
        type=int,
        default=-1,
        help=(
            "Transformer max sequence length / learned pos-emb length. "
            "If <0, auto-compute from (qa_tokens, add_eos, n_point/n_ray and schedules)."
        ),
    )
    ap.add_argument(
        "--n_point_schedule",
        type=str,
        default="",
        help=(
            "Optional epoch-based n_point schedule. Format: '0:256,10:512,20:1024'. "
            "If empty, uses --n_point for all epochs."
        ),
    )
    ap.add_argument(
        "--n_ray_schedule",
        type=str,
        default="",
        help=(
            "Optional epoch-based n_ray schedule. Format: '0:256,10:512'. "
            "If empty, uses --n_ray for all epochs."
        ),
    )
    ap.add_argument(
        "--resume_optimizer",
        type=int,
        default=1,
        help=(
            "When --resume is set, also load optimizer state (1) or not (0). "
            "If max_len changes (pos_emb resize), optimizer state is partially restored by default."
        ),
    )
    ap.add_argument(
        "--resume_optimizer_partial",
        type=int,
        default=1,
        help=(
            "If 1 and --resume_optimizer=1, attempt partial optimizer restore when parameter shapes change "
            "(drop mismatched per-parameter states, keep others)."
        ),
    )
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--save_dir", type=str, default="runs/querynepa3d_pretrain")
    ap.add_argument("--save_every", type=int, default=1, help="save periodic checkpoints every N epochs (>=1)")
    ap.add_argument("--save_last", type=int, default=1, help="if 1, also write save_dir/last.pt at checkpoint save points")
    ap.add_argument("--resume", type=str, default="", help="checkpoint path to resume from (e.g. save_dir/last.pt)")
    ap.add_argument("--auto_resume", type=int, default=1, help="if 1 and --resume is empty, auto-resume from save_dir/last.pt")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--drop_ray_prob", type=float, default=0.0)
    ap.add_argument("--force_missing_ray", action="store_true")
    ap.add_argument("--add_eos", type=int, default=1)
    ap.add_argument("--qa_tokens", type=int, default=0, help="Use Q/A separated tokenization (v2).")
    ap.add_argument(
        "--qa_layout",
        type=str,
        default="interleave",
        choices=["interleave", "split"],
        help="Token layout when qa_tokens=1.",
    )
    ap.add_argument(
        "--pt_xyz_key",
        type=str,
        default="pt_xyz_pool",
        help="Point pool key used for point-token sampling.",
    )
    ap.add_argument(
        "--pt_dist_key",
        type=str,
        default="pt_dist_pool",
        help="Distance pool key paired with --pt_xyz_key (set empty to disable explicit dist pool lookup).",
    )
    ap.add_argument(
        "--ablate_point_dist",
        type=int,
        default=0,
        help="If 1, zero-out point distance channel regardless of dist pool.",
    )
    ap.add_argument(
        "--pt_sample_mode_train",
        type=str,
        default="random",
        choices=["random", "fps", "rfps", "fixed_grid"],
        help="Point sampling mode used by pretrain dataset.",
    )
    ap.add_argument(
        "--pt_fps_key",
        type=str,
        default="auto",
        help="Key for precomputed FPS order (e.g., pc_fps_order/pt_fps_order/auto).",
    )
    ap.add_argument(
        "--pt_rfps_m",
        type=int,
        default=4096,
        help="Candidate size for RFPS mode.",
    )
    ap.add_argument(
        "--point_order_mode",
        type=str,
        default="morton",
        choices=["morton", "fps", "random"],
        help=(
            "Point token ordering after sampling: morton=legacy, "
            "fps=keep sampling order, random=shuffle."
        ),
    )
    ap.add_argument(
        "--include_pt_grad",
        type=int,
        default=0,
        help="Enable explicit point-gradient feature slots in tokenizer.",
    )
    ap.add_argument("--pt_grad_mode", type=str, default="raw", choices=["raw", "log"])
    ap.add_argument("--pt_grad_eps", type=float, default=1e-3)
    ap.add_argument("--pt_grad_clip", type=float, default=10.0)
    ap.add_argument("--pt_grad_orient", type=str, default="none", choices=["none", "ray"])
    ap.add_argument(
        "--include_ray_unc",
        type=int,
        default=0,
        help="Enable ray-answer uncertainty feature slot in tokenizer.",
    )
    ap.add_argument("--ray_unc_k", type=int, default=8)
    ap.add_argument("--ray_unc_mode", type=str, default="normal_var", choices=["normal_var"])
    ap.add_argument(
        "--arch",
        type=str,
        default="causal",
        choices=["causal", "encdec"],
        help="Backbone architecture.",
    )
    ap.add_argument("--topo_k", type=int, default=0, help="kNN size for encoder topology attention (encdec).")
    ap.add_argument("--topo_ray_coord", type=str, default="origin", choices=["origin", "proj", "bbox"])
    ap.add_argument("--topo_ray_bbox", type=float, default=0.5)
    ap.add_argument(
        "--encdec_src_causal",
        type=int,
        default=0,
        help="If 1, apply a causal (future-masking) attention mask inside the encoder when arch=encdec.",
    )
    ap.add_argument("--dual_mask_near", type=float, default=0.0, help="Dual masking prob for *near* past tokens (PointGPT-style).")
    ap.add_argument("--dual_mask_far", type=float, default=0.0, help="Dual masking prob for *far* past tokens.")
    ap.add_argument("--dual_mask_window", type=int, default=32, help="Near-window size in token steps for dual masking.")
    ap.add_argument("--dual_mask_warmup_frac", type=float, default=0.05, help="Warmup fraction for ramping dual masking to target probs.")
    ap.add_argument(
        "--dual_mask_type_aware",
        type=int,
        default=0,
        help="If 1, apply dual-mask only to Query-like token pairs (Q/Q).",
    )
    ap.add_argument(
        "--dual_mask_window_scale",
        type=str,
        default="linear",
        choices=["none", "linear", "sqrt"],
        help=(
            "How to scale dual_mask_window when n_point/n_ray are scheduled. "
            "none: keep fixed; linear: multiply by (cur_total/ref_total); "
            "sqrt: multiply by sqrt(cur_total/ref_total)."
        ),
    )
    ap.add_argument(
        "--dual_mask_window_ref_total",
        type=int,
        default=-1,
        help=(
            "Reference total queries (n_point+n_ray) used for dual_mask_window scaling. "
            "If <=0, uses the epoch0 total after schedule init (args.n_point+args.n_ray)."
        ),
    )
    # B-2: ray monotonicity / depth supervision aux loss.
    ap.add_argument("--aux_b2_weight", type=float, default=0.0, help="Global weight for B-2 ray auxiliary loss (0=off).")
    ap.add_argument("--aux_b2_hit_weight", type=float, default=1.0, help="Weight of ray hit BCE term inside B-2.")
    ap.add_argument("--aux_b2_t_weight", type=float, default=1.0, help="Weight of ray depth regression term inside B-2.")
    ap.add_argument("--aux_b2_rank_weight", type=float, default=1.0, help="Weight of ray depth rank hinge term inside B-2.")
    ap.add_argument("--aux_b2_rank_pairs", type=int, default=128, help="Number of sampled ray-hit pairs per sample for B-2 ranking.")
    ap.add_argument("--aux_b2_rank_margin", type=float, default=0.0, help="Margin for B-2 ranking hinge.")
    # B-3: near-surface point distance auxiliary target.
    ap.add_argument("--aux_b3_weight", type=float, default=0.0, help="Global weight for B-3 near-surface point auxiliary loss (0=off).")
    ap.add_argument("--aux_b3_near_tau", type=float, default=0.05, help="Near-surface threshold for B-3 (distance <= tau).")
    # C-0: teacher-student refresh (distillation) as a minimal pseudo-parallel prototype.
    ap.add_argument("--teacher_ckpt", type=str, default="", help="Optional teacher checkpoint for refresh distillation (C-0).")
    ap.add_argument("--teacher_distill_weight", type=float, default=0.0, help="Weight for teacher-student distillation loss (0=off).")
    ap.add_argument(
        "--teacher_answer_drop_prob",
        type=float,
        default=0.0,
        help="C-1: drop answer observations in student input for teacher distillation.",
    )
    ap.add_argument(
        "--cycle_weight",
        type=float,
        default=0.0,
        help="C-2: answer-token consistency weight across two input views.",
    )
    ap.add_argument(
        "--cycle_answer_drop_prob",
        type=float,
        default=0.3,
        help="C-2: answer observation dropout prob for cycle view.",
    )
    ap.add_argument(
        "--d_hard_weight",
        type=float,
        default=0.0,
        help="D: weight for hard-query answer-token mining loss (0=off).",
    )
    ap.add_argument(
        "--d_hard_top_frac",
        type=float,
        default=0.25,
        help="D: top fraction of answer tokens (by current error) used per sample.",
    )
    ap.add_argument(
        "--d_hard_min_tokens",
        type=int,
        default=32,
        help="D: minimum number of hard answer tokens selected per sample.",
    )
    ap.add_argument(
        "--aux_e_weight",
        type=float,
        default=0.0,
        help="E: weight for decoder-side point-distance auxiliary head loss (0=off).",
    )
    ap.add_argument("--objective", type=str, default="nepa", choices=["nepa", "mae"])
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument(
        "--mixed_precision",
        type=str,
        default="auto",
        choices=["auto", "no", "fp16", "bf16"],
        help="Mixed precision mode for Accelerate (auto/no/fp16/bf16).",
    )
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if isinstance(args.pt_dist_key, str) and len(args.pt_dist_key.strip()) == 0:
        args.pt_dist_key = None

    arch_warns: list[str] = []
    if str(args.arch) == "encdec":
        if int(args.qa_tokens) != 1:
            arch_warns.append("[WARN] --arch encdec implies --qa_tokens=1")
            args.qa_tokens = 1
        if str(args.qa_layout) != "split":
            arch_warns.append("[WARN] --arch encdec implies --qa_layout=split")
            args.qa_layout = "split"

    req_mp = str(args.mixed_precision)
    if req_mp == "auto":
        if not torch.cuda.is_available():
            req_mp = "no"
        elif torch.cuda.is_bf16_supported():
            req_mp = "bf16"
        else:
            req_mp = "fp16"
    accelerator = Accelerator(mixed_precision=req_mp)
    args.mixed_precision = str(accelerator.mixed_precision)
    mprint = accelerator.print
    mprint(
        f"[accelerate] num_processes={accelerator.num_processes} "
        f"distributed_type={accelerator.distributed_type} mixed_precision={accelerator.mixed_precision}"
    )
    for _w in arch_warns:
        mprint(_w)

    set_seed(args.seed)

    def _worker_init_fn(worker_id: int):
        # Ensure numpy RNG differs across dataloader workers.
        import random
        base = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(base)
        random.seed(base)

    device = accelerator.device

    # -------------------------
    # Point/ray scaling schedule (curriculum)
    # -------------------------
    qa_tokens = bool(args.qa_tokens)
    add_eos = bool(args.add_eos)

    def _parse_epoch_value_schedule(s: str) -> list[tuple[int, int]]:
        s = (s or "").strip()
        if not s:
            return []
        items: list[tuple[int, int]] = []
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(
                    f"bad schedule item '{part}'. Expected 'epoch:value' (e.g., '10:512')."
                )
            ep_s, val_s = part.split(":", 1)
            items.append((int(ep_s), int(val_s)))
        items.sort(key=lambda x: x[0])
        return items

    def _schedule_value(items: list[tuple[int, int]], epoch: int, default: int) -> int:
        v = int(default)
        for ep, val in items:
            if int(epoch) >= int(ep):
                v = int(val)
            else:
                break
        return v

    def _scaled_dual_mask_window(
        base_window: int,
        cur_total: int,
        ref_total: int,
        mode: str,
    ) -> int:
        """Scale dual_mask_window for larger query counts.

        We keep the semantics simple: base_window is in *token distance* (as used by the
        attention-bias implementation). We scale it by a ratio derived from the total
        number of queries (n_point+n_ray).
        """

        if int(base_window) <= 0:
            return int(base_window)
        if str(mode) == "none":
            return int(base_window)

        denom = max(1, int(ref_total))
        ratio = float(cur_total) / float(denom)
        if str(mode) == "sqrt":
            ratio = math.sqrt(max(1e-12, ratio))

        w = int(round(float(base_window) * ratio))
        return max(1, w)

    n_point_sched = _parse_epoch_value_schedule(args.n_point_schedule)
    n_ray_sched = _parse_epoch_value_schedule(args.n_ray_schedule)

    def _max_sched_value(items: list[tuple[int, int]], default: int) -> int:
        if not items:
            return int(default)
        return max(int(default), max(int(v) for _, v in items))

    def _required_seq_len(n_point: int, n_ray: int) -> int:
        if qa_tokens:
            # BOS + interleaved (Q,A) pairs + optional EOS
            return 1 + 2 * int(n_point) + 2 * int(n_ray) + (1 if add_eos else 0)
        # legacy: BOS + points + rays + optional EOS
        return 1 + int(n_point) + int(n_ray) + (1 if add_eos else 0)

    # Auto max_len must cover the maximum sizes used in schedule.
    n_point_max = _max_sched_value(n_point_sched, args.n_point)
    n_ray_max = _max_sched_value(n_ray_sched, args.n_ray)
    required_max_len = _required_seq_len(n_point_max, n_ray_max)
    if int(args.max_len) < 0:
        args.max_len = required_max_len
    if int(args.max_len) < required_max_len:
        raise ValueError(
            f"--max_len too small for requested schedule: max_len={args.max_len} < required={required_max_len} "
            f"(qa_tokens={qa_tokens}, add_eos={add_eos}, n_point_max={n_point_max}, n_ray_max={n_ray_max})."
        )

    # Initial dataset sizes (epoch 0). If schedule is provided, it overrides.
    n_point_init = _schedule_value(n_point_sched, 0, args.n_point)
    n_ray_init = _schedule_value(n_ray_sched, 0, args.n_ray)
    if n_point_init != args.n_point or n_ray_init != args.n_ray:
        mprint(
            f"[schedule:init] overriding dataset sizes at epoch0: n_point {args.n_point}->{n_point_init}, "
            f"n_ray {args.n_ray}->{n_ray_init}"
        )
        args.n_point = int(n_point_init)
        args.n_ray = int(n_ray_init)

    # Dual-mask window scaling reference.
    # By default, we use the epoch0 total queries after schedule init.
    dual_mask_ref_total = (
        int(args.dual_mask_window_ref_total)
        if int(args.dual_mask_window_ref_total) > 0
        else int(args.n_point) + int(args.n_ray)
    )

    mprint(
        "[point_tokens] "
        f"pt_xyz_key={args.pt_xyz_key} "
        f"pt_dist_key={args.pt_dist_key} "
        f"ablate_point_dist={int(bool(args.ablate_point_dist))} "
        f"pt_sample_mode_train={args.pt_sample_mode_train} "
        f"pt_fps_key={args.pt_fps_key} "
        f"pt_rfps_m={int(args.pt_rfps_m)} "
        f"point_order_mode={args.point_order_mode}"
    )

    if args.mix_config:
        ds, sampler, mix_info = build_mixed_pretrain(
            args.mix_config,
            qa_tokens=bool(args.qa_tokens),
            qa_layout=str(args.qa_layout),
            n_point=args.n_point,
            n_ray=args.n_ray,
            mode="train",
            drop_ray_prob=args.drop_ray_prob,
            force_missing_ray=args.force_missing_ray,
            add_eos=bool(args.add_eos),
            include_pt_grad=bool(args.include_pt_grad),
            pt_grad_mode=str(args.pt_grad_mode),
            pt_grad_eps=float(args.pt_grad_eps),
            pt_grad_clip=float(args.pt_grad_clip),
            pt_grad_orient=str(args.pt_grad_orient),
            include_ray_unc=bool(args.include_ray_unc),
            ray_unc_k=int(args.ray_unc_k),
            ray_unc_mode=str(args.ray_unc_mode),
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
            pt_xyz_key=str(args.pt_xyz_key),
            pt_dist_key=args.pt_dist_key,
            ablate_point_dist=bool(args.ablate_point_dist),
            pt_sample_mode=str(args.pt_sample_mode_train),
            pt_fps_key=str(args.pt_fps_key),
            pt_rfps_m=int(args.pt_rfps_m),
            point_order_mode=str(args.point_order_mode),
        )
        # Optional overrides from CLI (useful for PBS -v variables)
        if args.mix_num_samples and args.mix_num_samples > 0:
            sampler.num_samples = int(args.mix_num_samples)
        if args.mix_seed and args.mix_seed != mix_info.get("seed", 0):
            sampler.seed = int(args.mix_seed)

        mprint("[mix] components:")
        for n, w, sz in zip(mix_info["names"], mix_info["weights"], mix_info["sizes"]):
            mprint(f"  - {n}: weight={w:.3f} size={sz}")
        mprint(
            f"[mix] num_samples_per_epoch={len(sampler)} replacement={mix_info['replacement']} "
            f"seed={sampler.seed}"
        )

        dl = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
            worker_init_fn=_worker_init_fn,
        )
    else:
        train_paths = list_npz(args.cache_root, "train")
        ds = ModelNet40QueryDataset(
            train_paths,
            backend=args.backend,
            n_point=args.n_point,
            n_ray=args.n_ray,
            drop_ray_prob=args.drop_ray_prob,
            force_missing_ray=args.force_missing_ray,
            add_eos=bool(args.add_eos),
            qa_tokens=bool(args.qa_tokens),
            qa_layout=str(args.qa_layout),
            include_pt_grad=bool(args.include_pt_grad),
            pt_grad_mode=str(args.pt_grad_mode),
            pt_grad_eps=float(args.pt_grad_eps),
            pt_grad_clip=float(args.pt_grad_clip),
            pt_grad_orient=str(args.pt_grad_orient),
            include_ray_unc=bool(args.include_ray_unc),
            ray_unc_k=int(args.ray_unc_k),
            ray_unc_mode=str(args.ray_unc_mode),
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
            pt_xyz_key=str(args.pt_xyz_key),
            pt_dist_key=args.pt_dist_key,
            ablate_point_dist=bool(args.ablate_point_dist),
            pt_sample_mode=str(args.pt_sample_mode_train),
            pt_fps_key=str(args.pt_fps_key),
            pt_rfps_m=int(args.pt_rfps_m),
            point_order_mode=str(args.point_order_mode),
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
            worker_init_fn=_worker_init_fn,
        )

    qa_tokens = bool(args.qa_tokens)
    n_types = 9 if qa_tokens else 5
    # max_len is the *capacity*; actual sequence length varies with n_point/n_ray.
    t = int(args.max_len)

    model = QueryNepa(
        feat_dim=15,
        d_model=args.d_model,
        n_types=n_types,
        nhead=args.heads,
        num_layers=args.layers,
        max_len=t,
        arch=str(args.arch),
        topo_k=int(args.topo_k),
        topo_ray_coord=str(args.topo_ray_coord),
        topo_ray_bbox=float(args.topo_ray_bbox),
        encdec_src_causal=int(args.encdec_src_causal),
    ).to(device)

    teacher_model = None
    teacher_on = (len(str(args.teacher_ckpt).strip()) > 0) and (float(args.teacher_distill_weight) > 0.0)
    if teacher_on:
        teacher_ckpt = torch.load(str(args.teacher_ckpt).strip(), map_location="cpu")
        teacher_state = teacher_ckpt["model"]
        teacher_pre_args = teacher_ckpt.get("args", {})
        teacher_d_model = int(teacher_state["type_emb.weight"].shape[1])
        teacher_n_types = int(teacher_state["type_emb.weight"].shape[0])
        teacher_heads = int(teacher_pre_args.get("heads", args.heads))
        teacher_layers = int(teacher_pre_args.get("layers", args.layers))
        teacher_arch = str(teacher_pre_args.get("arch", "causal"))
        teacher_topo_k = int(teacher_pre_args.get("topo_k", 0))
        teacher_topo_ray_coord = str(teacher_pre_args.get("topo_ray_coord", "origin"))
        teacher_topo_ray_bbox = float(teacher_pre_args.get("topo_ray_bbox", 0.5))
        teacher_encdec_src_causal = int(teacher_pre_args.get("encdec_src_causal", 0))
        teacher_len = int(teacher_state["pos_emb"].shape[1])
        if teacher_n_types != int(n_types):
            raise RuntimeError(
                f"teacher n_types mismatch: teacher={teacher_n_types}, student={n_types}. "
                "Use a compatible checkpoint or disable teacher distillation."
            )
        if teacher_len != int(t):
            mprint(f"[teacher] resizing pos_emb: ckpt_len={teacher_len} -> max_len={t}")
            teacher_state = maybe_resize_pos_emb_in_state_dict(dict(teacher_state), int(t))
        teacher_model = QueryNepa(
            feat_dim=15,
            d_model=teacher_d_model,
            n_types=teacher_n_types,
            nhead=teacher_heads,
            num_layers=teacher_layers,
            max_len=t,
            arch=teacher_arch,
            topo_k=teacher_topo_k,
            topo_ray_coord=teacher_topo_ray_coord,
            topo_ray_bbox=teacher_topo_ray_bbox,
            encdec_src_causal=teacher_encdec_src_causal,
        ).to(device)
        load_state_dict_flexible(teacher_model, teacher_state, strict=True)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        mprint(
            f"[teacher] enabled: ckpt={os.path.abspath(str(args.teacher_ckpt).strip())} "
            f"weight={float(args.teacher_distill_weight)}"
        )

    aux_heads_dict = {}
    if float(args.aux_b2_weight) > 0.0:
        aux_heads_dict["ray_hit"] = nn.Linear(args.d_model, 1)
        aux_heads_dict["ray_t"] = nn.Linear(args.d_model, 1)
    if float(args.aux_b3_weight) > 0.0:
        aux_heads_dict["point_dist"] = nn.Linear(args.d_model, 1)
    if float(args.aux_e_weight) > 0.0:
        aux_heads_dict["e_point_dist"] = nn.Linear(args.d_model, 1)

    aux_heads = nn.ModuleDict(aux_heads_dict).to(device) if len(aux_heads_dict) > 0 else None
    aux_enabled = aux_heads is not None
    params = list(model.parameters())
    if aux_enabled:
        params += list(aux_heads.parameters())

    opt = optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    if aux_enabled:
        model, aux_heads, opt, dl = accelerator.prepare(model, aux_heads, opt, dl)
        raw_aux_heads = accelerator.unwrap_model(aux_heads)
    else:
        model, opt, dl = accelerator.prepare(model, opt, dl)
        raw_aux_heads = None
    raw_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    save_every = max(1, int(args.save_every))
    save_last = bool(int(args.save_last))
    auto_resume = bool(int(args.auto_resume))

    start_epoch = 0
    step = 0
    resume_path = args.resume.strip()
    if (not resume_path) and auto_resume:
        candidate = os.path.join(args.save_dir, "last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate

    if resume_path:
        if not os.path.isfile(resume_path):
            if auto_resume:
                mprint(f"[resume] checkpoint not found ({resume_path}); starting fresh")
                resume_path = ""
            else:
                raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")

        ckpt_model = ckpt["model"]
        ckpt_pos_len: int | None = None
        if (
            "pos_emb" in ckpt_model
            and torch.is_tensor(ckpt_model["pos_emb"])
            and ckpt_model["pos_emb"].ndim == 3
        ):
            ckpt_pos_len = int(ckpt_model["pos_emb"].shape[1])
            if ckpt_pos_len != int(t):
                mprint(f"[resume] resizing pos_emb: ckpt_len={ckpt_pos_len} -> max_len={t}")
                ckpt_model = maybe_resize_pos_emb_in_state_dict(dict(ckpt_model), int(t))

        load_state_dict_flexible(raw_model, ckpt_model, strict=True)

        can_resume_opt = bool(int(args.resume_optimizer))
        pos_emb_resized = bool(ckpt_pos_len is not None and ckpt_pos_len != int(t))
        partial_resume_opt = bool(int(args.resume_optimizer_partial))
        if pos_emb_resized and (not partial_resume_opt):
            can_resume_opt = False

        ckpt_aux = ckpt.get("aux_heads", None)
        if aux_enabled:
            if ckpt_aux is None:
                # aux heads are newly enabled for this run.
                can_resume_opt = False
                mprint("[resume] aux_heads missing in checkpoint; using fresh aux head state")
            else:
                rep = raw_aux_heads.load_state_dict(ckpt_aux, strict=False)
                if len(rep.missing_keys) > 0 or len(rep.unexpected_keys) > 0:
                    can_resume_opt = False
                    mprint(
                        "[resume] aux_heads mismatch; using fresh optimizer state "
                        f"(missing={rep.missing_keys}, unexpected={rep.unexpected_keys})"
                    )
        elif ckpt_aux is not None:
            # checkpoint contains aux params but current run disabled them.
            can_resume_opt = False

        if can_resume_opt and ("opt" in ckpt):
            try:
                opt_state = ckpt["opt"]
                if pos_emb_resized and partial_resume_opt:
                    opt_state, dropped, checked = _prune_optimizer_state_for_shape_mismatch(opt, opt_state)
                    mprint(
                        "[resume] pos_emb resized; partial optimizer restore enabled "
                        f"(dropped_states={dropped}/{checked})"
                    )
                opt.load_state_dict(opt_state)
            except Exception as e:
                mprint(f"[resume] failed to load optimizer state ({e}); using fresh optimizer state")
                can_resume_opt = False
        else:
            if "opt" not in ckpt:
                mprint("[resume] optimizer state missing in checkpoint; using fresh optimizer state")
            elif not can_resume_opt:
                mprint("[resume] skipping optimizer state load")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        step = int(ckpt.get("step", start_epoch * len(dl)))
        mprint(f"[resume] loaded={resume_path} start_epoch={start_epoch} step={step}")
    else:
        mprint("[resume] disabled (no checkpoint found/requested)")

    if start_epoch >= args.epochs:
        mprint(f"[resume] checkpoint already reached target epochs: start_epoch={start_epoch} >= epochs={args.epochs}")
        accelerator.end_training()
        return

    # Dual-masking schedule (PointGPT-style) for AR shortcut mitigation.
    # We ramp probabilities from 0 -> target over an initial warmup fraction.
    total_steps = int(args.epochs) * max(1, len(dl))
    warmup_steps = max(1, int(float(args.dual_mask_warmup_frac) * total_steps))
    # Keep schedule consistent with resume.
    global_step = int(step)

    # Track current dataset sizes so we only print when they change.
    cur_n_point = int(args.n_point)
    cur_n_ray = int(args.n_ray)
    last_dm_window = None

    model.train()
    for ep in range(start_epoch, args.epochs):
        # Update dataset sizes according to schedule.
        if n_point_sched or n_ray_sched:
            new_n_point = _schedule_value(n_point_sched, ep, cur_n_point)
            new_n_ray = _schedule_value(n_ray_sched, ep, cur_n_ray)
            if new_n_point != cur_n_point or new_n_ray != cur_n_ray:
                needed = _required_seq_len(new_n_point, new_n_ray)
                if needed > int(t):
                    raise ValueError(
                        f"schedule requests seq_len={needed} at epoch {ep}, but --max_len={t}. "
                        f"(n_point={new_n_point}, n_ray={new_n_ray}, qa_tokens={qa_tokens}, add_eos={add_eos})"
                    )
                ds.set_sizes(n_point=new_n_point, n_ray=new_n_ray)
                cur_n_point, cur_n_ray = int(new_n_point), int(new_n_ray)
                mprint(f"[schedule] epoch {ep}: n_point={cur_n_point}, n_ray={cur_n_ray}")

        # Dual-mask window scaling for larger point/ray counts.
        cur_total = int(cur_n_point) + int(cur_n_ray)
        dm_window = _scaled_dual_mask_window(
            int(args.dual_mask_window),
            cur_total,
            int(dual_mask_ref_total),
            str(args.dual_mask_window_scale),
        )
        if last_dm_window is None or int(dm_window) != int(last_dm_window):
            if str(args.dual_mask_window_scale) != "none":
                mprint(
                    f"[dual_mask] epoch {ep}: window {int(args.dual_mask_window)}->{int(dm_window)} "
                    f"(scale={args.dual_mask_window_scale}, ref_total={dual_mask_ref_total}, cur_total={cur_total})"
                )
            last_dm_window = int(dm_window)

        if args.mix_config:
            # Deterministic per-epoch sampler.
            if hasattr(dl, "set_epoch"):
                try:
                    dl.set_epoch(ep)
                except Exception:
                    pass
            try:
                dl.sampler.set_epoch(ep)
            except Exception:
                pass
        for batch in dl:
            feat = batch["feat"].to(device, non_blocking=True).float()
            type_id = batch["type_id"].to(device, non_blocking=True).long()

            opt.zero_grad(set_to_none=True)
            with accelerator.autocast():
                if args.objective == "nepa":
                    # Dual masking only affects the causal attention during training.
                    # Note: we keep it off for MAE objective to avoid confounding baselines.
                    ramp = min(1.0, float(global_step) / float(warmup_steps))
                    dm_near = float(args.dual_mask_near) * ramp
                    dm_far = float(args.dual_mask_far) * ramp
                    dm_seed = int(args.seed) * 1000003 + int(global_step)

                    z, z_hat, h_main = model(
                        feat,
                        type_id,
                        dual_mask_near=dm_near,
                        dual_mask_far=dm_far,
                        dual_mask_window=int(dm_window),
                        dual_mask_seed=dm_seed,
                        dual_mask_type_aware=int(args.dual_mask_type_aware),
                    )
                    loss_main = raw_model.nepa_loss(z, z_hat, type_id=type_id)

                    b2_loss = loss_main.new_tensor(0.0)
                    b3_loss = loss_main.new_tensor(0.0)
                    aux_loss = loss_main.new_tensor(0.0)
                    distill_loss = loss_main.new_tensor(0.0)
                    cycle_loss = loss_main.new_tensor(0.0)
                    d_loss = loss_main.new_tensor(0.0)
                    e_loss = loss_main.new_tensor(0.0)
                    if aux_enabled:
                        pred_tok = z_hat[:, :-1, :]
                        hid_tok = h_main[:, :-1, :]
                        tgt_feat = feat[:, 1:, :]
                        tgt_type = type_id[:, 1:]

                        if float(args.aux_b2_weight) > 0.0:
                            ray_mask = (tgt_type == TYPE_A_RAY) | (tgt_type == TYPE_RAY)
                            if bool(ray_mask.any()):
                                gt_hit = tgt_feat[..., 11].clamp(0.0, 1.0)
                                gt_t = tgt_feat[..., 9].clamp_min(0.0)

                                pred_hit_logit = aux_heads["ray_hit"](pred_tok).squeeze(-1)
                                pred_t = F.softplus(aux_heads["ray_t"](pred_tok).squeeze(-1))

                                hit_bce = F.binary_cross_entropy_with_logits(
                                    pred_hit_logit[ray_mask], gt_hit[ray_mask]
                                )
                                hit_mask = ray_mask & (gt_hit > 0.5)
                                if bool(hit_mask.any()):
                                    t_reg = F.smooth_l1_loss(pred_t[hit_mask], gt_t[hit_mask])
                                    rank = _ray_rank_hinge_loss(
                                        pred_t,
                                        gt_t,
                                        hit_mask,
                                        n_pairs=int(args.aux_b2_rank_pairs),
                                        margin=float(args.aux_b2_rank_margin),
                                    )
                                else:
                                    t_reg = loss_main.new_tensor(0.0)
                                    rank = loss_main.new_tensor(0.0)

                                b2_loss = (
                                    float(args.aux_b2_hit_weight) * hit_bce
                                    + float(args.aux_b2_t_weight) * t_reg
                                    + float(args.aux_b2_rank_weight) * rank
                                )

                        if float(args.aux_b3_weight) > 0.0:
                            point_mask = (tgt_type == TYPE_A_POINT) | (tgt_type == TYPE_POINT)
                            if bool(point_mask.any()):
                                gt_dist = tgt_feat[..., 10].clamp_min(0.0)
                                near_mask = point_mask & (gt_dist <= float(args.aux_b3_near_tau))
                                if bool(near_mask.any()):
                                    pred_dist = F.softplus(aux_heads["point_dist"](pred_tok).squeeze(-1))
                                    b3_loss = F.smooth_l1_loss(pred_dist[near_mask], gt_dist[near_mask])

                        if float(args.aux_e_weight) > 0.0:
                            point_mask_e = (tgt_type == TYPE_A_POINT) | (tgt_type == TYPE_POINT)
                            if bool(point_mask_e.any()):
                                gt_dist_e = tgt_feat[..., 10].clamp_min(0.0)
                                pred_dist_e = F.softplus(aux_heads["e_point_dist"](hid_tok).squeeze(-1))
                                e_loss = F.smooth_l1_loss(pred_dist_e[point_mask_e], gt_dist_e[point_mask_e])

                        aux_loss = (
                            float(args.aux_b2_weight) * b2_loss
                            + float(args.aux_b3_weight) * b3_loss
                            + float(args.aux_e_weight) * e_loss
                        )

                    if teacher_model is not None:
                        s_pred = z_hat[:, :-1, :]
                        if float(args.teacher_answer_drop_prob) > 0.0:
                            feat_student = _drop_answer_observations(
                                feat,
                                type_id,
                                drop_prob=float(args.teacher_answer_drop_prob),
                            )
                            _, z_hat_distill, _ = model(
                                feat_student,
                                type_id,
                                dual_mask_near=dm_near,
                                dual_mask_far=dm_far,
                                dual_mask_window=int(dm_window),
                                dual_mask_seed=dm_seed,
                                dual_mask_type_aware=int(args.dual_mask_type_aware),
                            )
                            s_pred = z_hat_distill[:, :-1, :]
                        with torch.no_grad():
                            _, t_z_hat, _ = teacher_model(
                                feat,
                                type_id,
                                dual_mask_near=dm_near,
                                dual_mask_far=dm_far,
                                dual_mask_window=int(dm_window),
                                dual_mask_seed=dm_seed,
                                dual_mask_type_aware=int(args.dual_mask_type_aware),
                            )
                        t_pred = t_z_hat[:, :-1, :].detach()
                        target_type = type_id[:, 1:]
                        dmask = _default_answer_mask(target_type)
                        if bool(dmask.any()):
                            distill_loss = (1.0 - F.cosine_similarity(s_pred[dmask], t_pred[dmask], dim=-1, eps=1e-8)).mean()

                    if float(args.cycle_weight) > 0.0:
                        feat_cycle = _drop_answer_observations(
                            feat,
                            type_id,
                            drop_prob=float(args.cycle_answer_drop_prob),
                        )
                        _, z_hat_cycle, _ = model(
                            feat_cycle,
                            type_id,
                            dual_mask_near=dm_near,
                            dual_mask_far=dm_far,
                            dual_mask_window=int(dm_window),
                            dual_mask_seed=dm_seed + 17,
                            dual_mask_type_aware=int(args.dual_mask_type_aware),
                        )
                        s_pred = z_hat[:, :-1, :]
                        c_pred = z_hat_cycle[:, :-1, :]
                        target_type = type_id[:, 1:]
                        cmask = _default_answer_mask(target_type)
                        if bool(cmask.any()):
                            cycle_loss = (
                                1.0
                                - F.cosine_similarity(
                                    c_pred[cmask],
                                    s_pred.detach()[cmask],
                                    dim=-1,
                                    eps=1e-8,
                                )
                            ).mean()

                    if float(args.d_hard_weight) > 0.0:
                        tok_loss = 1.0 - F.cosine_similarity(
                            z_hat[:, :-1, :],
                            z[:, 1:, :].detach(),
                            dim=-1,
                            eps=1e-8,
                        )
                        d_loss = _hard_answer_topk_loss(
                            tok_loss,
                            type_id[:, 1:],
                            top_frac=float(args.d_hard_top_frac),
                            min_tokens=int(args.d_hard_min_tokens),
                        )

                    loss = (
                        loss_main
                        + aux_loss
                        + float(args.teacher_distill_weight) * distill_loss
                        + float(args.cycle_weight) * cycle_loss
                        + float(args.d_hard_weight) * d_loss
                    )
                else:
                    token_mask = build_token_mask(type_id, args.mask_ratio)
                    feat_in = feat.clone()
                    feat_in[token_mask] = 0.0
                    with torch.no_grad():
                        z_target = raw_model.embed_tokens(feat, type_id)
                    _, z_hat, _ = model(feat_in, type_id)
                    loss = raw_model.mae_loss(z_hat, z_target, token_mask)

            accelerator.backward(loss)
            opt.step()

            if step % 100 == 0:
                if args.objective == "nepa":
                    mprint(
                        "ep={} step={} loss={:.4f} main={:.4f} b2={:.4f} b3={:.4f} e={:.4f} d={:.4f} aux={:.4f}".format(
                            ep,
                            step,
                            float(loss.item()),
                            float(loss_main.item()),
                            float(b2_loss.item()),
                            float(b3_loss.item()),
                            float(e_loss.item()),
                            float(d_loss.item()),
                            float(aux_loss.item()),
                        )
                    )
                    if teacher_model is not None:
                        mprint(
                            "ep={} step={} distill={:.4f} w={:.4f}".format(
                                ep,
                                step,
                                float(distill_loss.item()),
                                float(args.teacher_distill_weight),
                            )
                        )
                    if float(args.cycle_weight) > 0.0:
                        mprint(
                            "ep={} step={} cycle={:.4f} w={:.4f}".format(
                                ep,
                                step,
                                float(cycle_loss.item()),
                                float(args.cycle_weight),
                            )
                        )
                else:
                    mprint(f"ep={ep} step={step} loss={loss.item():.4f}")
            step += 1
            global_step += 1

        is_last = ep == (args.epochs - 1)
        should_save = (ep % save_every == 0) or is_last
        if should_save and accelerator.is_main_process:
            ckpt = {
                "model": raw_model.state_dict(),
                "aux_heads": (raw_aux_heads.state_dict() if aux_enabled else None),
                "opt": opt.state_dict(),
                "scaler": None,
                "args": vars(args),
                "epoch": ep,
                "step": step,
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt, ckpt_path)
            if save_last:
                torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
