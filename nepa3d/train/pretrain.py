import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..data.dataset import ModelNet40QueryDataset, collate
from ..data.mixed_pretrain import build_mixed_pretrain
from ..data.modelnet40_index import list_npz
from ..models.query_nepa import QueryNepa
from ..token.tokenizer import TYPE_BOS, TYPE_EOS
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

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
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

        if can_resume_opt and ("opt" in ckpt):
            opt.load_state_dict(ckpt["opt"])
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

                    z, z_hat, _ = model(
                        feat,
                        type_id,
                        dual_mask_near=dm_near,
                        dual_mask_far=dm_far,
                        dual_mask_window=int(args.dual_mask_window),
                        dual_mask_seed=dm_seed,
                        dual_mask_type_aware=int(args.dual_mask_type_aware),
                    )
                    loss = model.nepa_loss(z, z_hat, type_id=type_id)
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
                print(f"ep={ep} step={step} loss={loss.item():.4f}")
            step += 1
            global_step += 1

        is_last = ep == (args.epochs - 1)
        should_save = (ep % save_every == 0) or is_last
        if should_save:
            ckpt = {
                "model": model.state_dict(),
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
