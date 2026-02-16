import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
            "When --resume is set, also load optimizer/scaler state (1) or not (0). "
            "If max_len changes (pos_emb resize), optimizer state is automatically skipped."
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
    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    def _worker_init_fn(worker_id: int):
        # Ensure numpy RNG differs across dataloader workers.
        import random
        base = (torch.initial_seed() + worker_id) % (2**32)
        np.random.seed(base)
        random.seed(base)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

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
        print(
            f"[schedule:init] overriding dataset sizes at epoch0: n_point {args.n_point}->{n_point_init}, "
            f"n_ray {args.n_ray}->{n_ray_init}"
        )
        args.n_point = int(n_point_init)
        args.n_ray = int(n_ray_init)

    if args.mix_config:
        ds, sampler, mix_info = build_mixed_pretrain(
            args.mix_config,
            qa_tokens=bool(args.qa_tokens),
            n_point=args.n_point,
            n_ray=args.n_ray,
            mode="train",
            drop_ray_prob=args.drop_ray_prob,
            force_missing_ray=args.force_missing_ray,
            add_eos=bool(args.add_eos),
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
        )
        # Optional overrides from CLI (useful for PBS -v variables)
        if args.mix_num_samples and args.mix_num_samples > 0:
            sampler.num_samples = int(args.mix_num_samples)
        if args.mix_seed and args.mix_seed != mix_info.get("seed", 0):
            sampler.seed = int(args.mix_seed)

        print("[mix] components:")
        for n, w, sz in zip(mix_info["names"], mix_info["weights"], mix_info["sizes"]):
            print(f"  - {n}: weight={w:.3f} size={sz}")
        print(f"[mix] num_samples_per_epoch={len(sampler)} replacement={mix_info['replacement']} seed={sampler.seed}")

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
            voxel_grid=args.voxel_grid,
            voxel_dilate=args.voxel_dilate,
            voxel_max_steps=args.voxel_max_steps,
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
        teacher_len = int(teacher_state["pos_emb"].shape[1])
        if teacher_n_types != int(n_types):
            raise RuntimeError(
                f"teacher n_types mismatch: teacher={teacher_n_types}, student={n_types}. "
                "Use a compatible checkpoint or disable teacher distillation."
            )
        if teacher_len != int(t):
            print(f"[teacher] resizing pos_emb: ckpt_len={teacher_len} -> max_len={t}")
            teacher_state = maybe_resize_pos_emb_in_state_dict(dict(teacher_state), int(t))
        teacher_model = QueryNepa(
            feat_dim=15,
            d_model=teacher_d_model,
            n_types=teacher_n_types,
            nhead=teacher_heads,
            num_layers=teacher_layers,
            max_len=t,
        ).to(device)
        load_state_dict_flexible(teacher_model, teacher_state, strict=True)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"[teacher] enabled: ckpt={os.path.abspath(str(args.teacher_ckpt).strip())} weight={float(args.teacher_distill_weight)}")

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
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
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
                print(f"[resume] checkpoint not found ({resume_path}); starting fresh")
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
                print(f"[resume] resizing pos_emb: ckpt_len={ckpt_pos_len} -> max_len={t}")
                ckpt_model = maybe_resize_pos_emb_in_state_dict(dict(ckpt_model), int(t))

        load_state_dict_flexible(model, ckpt_model, strict=True)

        can_resume_opt = bool(int(args.resume_optimizer))
        if ckpt_pos_len is not None and ckpt_pos_len != int(t):
            # Optimizer state tensors for pos_emb would mismatch.
            can_resume_opt = False

        ckpt_aux = ckpt.get("aux_heads", None)
        if aux_enabled:
            if ckpt_aux is None:
                # aux heads are newly enabled for this run.
                can_resume_opt = False
                print("[resume] aux_heads missing in checkpoint; using fresh aux head state")
            else:
                rep = aux_heads.load_state_dict(ckpt_aux, strict=False)
                if len(rep.missing_keys) > 0 or len(rep.unexpected_keys) > 0:
                    can_resume_opt = False
                    print(
                        "[resume] aux_heads mismatch; using fresh optimizer state "
                        f"(missing={rep.missing_keys}, unexpected={rep.unexpected_keys})"
                    )
        elif ckpt_aux is not None:
            # checkpoint contains aux params but current run disabled them.
            can_resume_opt = False

        if can_resume_opt and ("opt" in ckpt):
            try:
                opt.load_state_dict(ckpt["opt"])
            except Exception as e:
                print(f"[resume] failed to load optimizer state ({e}); using fresh optimizer state")
                can_resume_opt = False
        else:
            if "opt" not in ckpt:
                print("[resume] optimizer state missing in checkpoint; using fresh optimizer state")
            elif not can_resume_opt:
                print("[resume] skipping optimizer state load")

        if use_amp and can_resume_opt and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                print(f"[resume] failed to load scaler state ({e}); using fresh scaler state")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        step = int(ckpt.get("step", start_epoch * len(dl)))
        print(f"[resume] loaded={resume_path} start_epoch={start_epoch} step={step}")
    else:
        print("[resume] disabled (no checkpoint found/requested)")

    if start_epoch >= args.epochs:
        print(f"[resume] checkpoint already reached target epochs: start_epoch={start_epoch} >= epochs={args.epochs}")
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
                print(f"[schedule] epoch {ep}: n_point={cur_n_point}, n_ray={cur_n_ray}")

        if args.mix_config:
            # Deterministic per-epoch sampler.
            try:
                dl.sampler.set_epoch(ep)
            except Exception:
                pass
        for batch in dl:
            feat = batch["feat"].to(device, non_blocking=True).float()
            type_id = batch["type_id"].to(device, non_blocking=True).long()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
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
                        dual_mask_window=int(args.dual_mask_window),
                        dual_mask_seed=dm_seed,
                        dual_mask_type_aware=int(args.dual_mask_type_aware),
                    )
                    loss_main = model.nepa_loss(z, z_hat, type_id=type_id)

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
                                dual_mask_window=int(args.dual_mask_window),
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
                                dual_mask_window=int(args.dual_mask_window),
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
                            dual_mask_window=int(args.dual_mask_window),
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
                        z_target = model.embed_tokens(feat, type_id)
                    _, z_hat, _ = model(feat_in, type_id)
                    loss = model.mae_loss(z_hat, z_target, token_mask)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if step % 100 == 0:
                if args.objective == "nepa":
                    print(
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
                        print(
                            "ep={} step={} distill={:.4f} w={:.4f}".format(
                                ep,
                                step,
                                float(distill_loss.item()),
                                float(args.teacher_distill_weight),
                            )
                        )
                    if float(args.cycle_weight) > 0.0:
                        print(
                            "ep={} step={} cycle={:.4f} w={:.4f}".format(
                                ep,
                                step,
                                float(cycle_loss.item()),
                                float(args.cycle_weight),
                            )
                        )
                else:
                    print(f"ep={ep} step={step} loss={loss.item():.4f}")
            step += 1
            global_step += 1

        is_last = ep == (args.epochs - 1)
        should_save = (ep % save_every == 0) or is_last
        if should_save:
            ckpt = {
                "model": model.state_dict(),
                "aux_heads": (aux_heads.state_dict() if aux_enabled else None),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "args": vars(args),
                "epoch": ep,
                "step": step,
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt, ckpt_path)
            if save_last:
                torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))


if __name__ == "__main__":
    main()
