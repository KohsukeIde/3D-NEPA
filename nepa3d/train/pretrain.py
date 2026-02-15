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
    if qa_tokens:
        t = 1 + 2 * args.n_point + 2 * args.n_ray + (1 if bool(args.add_eos) else 0)
        n_types = 9
    else:
        t = 1 + args.n_point + args.n_ray + (1 if bool(args.add_eos) else 0)
        n_types = 5

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
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        else:
            print("[resume] optimizer state missing in checkpoint; using fresh optimizer state")
        if use_amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
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

    model.train()
    for ep in range(start_epoch, args.epochs):
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
