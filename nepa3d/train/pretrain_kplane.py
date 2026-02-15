from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from ..data.kplane_dataset import build_kplane_loader
from ..models.kplane import KPlaneConfig, KPlaneRegressor
from ..utils.seed import set_seed


def _worker_seed_info() -> None:
    # Keep BLAS threads bounded in multi-worker training jobs.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix_config", type=str, required=True)
    ap.add_argument("--mix_num_samples", type=int, default=0, help="Override mix_num_samples in YAML (0=use YAML)")
    ap.add_argument("--mix_seed", type=int, default=0, help="Override mix_seed in YAML (0=use YAML)")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--n_context", type=int, default=256)
    ap.add_argument("--n_query", type=int, default=256)
    ap.add_argument("--query_source", type=str, default="pool", choices=["pool", "grid"])
    ap.add_argument("--target_mode", type=str, default="backend", choices=["backend", "udf"])
    ap.add_argument("--disjoint_context_query", type=int, default=1)

    # Tri/K-plane baseline options.
    ap.add_argument("--plane_type", type=str, default="kplane", choices=["triplane", "kplane"])
    ap.add_argument("--fusion", type=str, default="auto", choices=["auto", "sum", "product"])
    ap.add_argument("--plane_resolutions", type=str, default="64", help="Comma-separated, e.g. 64 or 32,64,128")
    ap.add_argument("--plane_channels", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=128)

    ap.add_argument("--voxel_grid", type=int, default=64)
    ap.add_argument("--voxel_dilate", type=int, default=1)
    ap.add_argument("--voxel_max_steps", type=int, default=0)

    ap.add_argument("--save_dir", type=str, default="runs/eccv_kplane_baseline")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--save_last", type=int, default=1)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--auto_resume", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _worker_seed_info()
    set_seed(int(args.seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    fusion = str(args.fusion)
    if fusion == "auto":
        fusion = "sum" if str(args.plane_type) == "triplane" else "product"

    cfg = KPlaneConfig.from_args(
        plane_resolutions=args.plane_resolutions,
        plane_channels=int(args.plane_channels),
        hidden_dim=int(args.hidden_dim),
        fusion=fusion,
    )
    model = KPlaneRegressor(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = GradScaler(enabled=use_amp)

    dl, mix_info = build_kplane_loader(
        mix_config_path=args.mix_config,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        n_context=int(args.n_context),
        n_query=int(args.n_query),
        mode="train",
        disjoint_context_query=bool(int(args.disjoint_context_query)),
        query_source=str(args.query_source),
        target_mode=str(args.target_mode),
        voxel_grid=int(args.voxel_grid),
        voxel_dilate=int(args.voxel_dilate),
        voxel_max_steps=int(args.voxel_max_steps),
    )
    if int(args.mix_num_samples) > 0:
        dl.sampler.num_samples = int(args.mix_num_samples)
    if int(args.mix_seed) != 0 and int(args.mix_seed) != int(mix_info.get("seed", 0)):
        dl.sampler.seed = int(args.mix_seed)

    print("[mix] components:")
    for n, w, sz in zip(mix_info["names"], mix_info["weights"], mix_info["sizes"]):
        print(f"  - {n}: weight={w:.3f} size={sz}")
    print(f"[mix] num_samples_per_epoch={len(dl.sampler)} replacement={mix_info['replacement']} seed={dl.sampler.seed}")
    print(
        f"[model] plane_type={args.plane_type} fusion={fusion} "
        f"res={cfg.plane_resolutions} ch={cfg.plane_channels} hidden={cfg.hidden_dim}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_every = max(1, int(args.save_every))
    save_last = bool(int(args.save_last))
    auto_resume = bool(int(args.auto_resume))

    start_epoch = 0
    step = 0
    resume_path = str(args.resume).strip()
    if (not resume_path) and auto_resume:
        candidate = os.path.join(args.save_dir, "last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if use_amp and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        step = int(ckpt.get("step", start_epoch * max(1, len(dl))))
        print(f"[resume] loaded={resume_path} start_epoch={start_epoch} step={step}")
    else:
        print("[resume] disabled (no checkpoint found/requested)")

    if start_epoch >= int(args.epochs):
        print(f"[resume] target already reached: start_epoch={start_epoch} >= epochs={args.epochs}")
        return

    model.train()
    for ep in range(start_epoch, int(args.epochs)):
        try:
            dl.sampler.set_epoch(ep)
        except Exception:
            pass

        losses = []
        for batch in dl:
            ctx_xyz = batch["ctx_xyz"].to(device, non_blocking=True).float()
            ctx_dist = batch["ctx_dist"].to(device, non_blocking=True).float()
            qry_xyz = batch["qry_xyz"].to(device, non_blocking=True).float()
            qry_dist = batch["qry_dist"].to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                pred, _, _, _ = model(ctx_xyz, ctx_dist, qry_xyz)
                loss = torch.mean((pred - qry_dist) ** 2)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().cpu().item()))
            if step % 100 == 0:
                print(f"ep={ep} step={step} loss={losses[-1]:.6f}")
            step += 1

        ep_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch] ep={ep} loss_mean={ep_loss:.6f} n_steps={len(losses)}")

        is_last = ep == (int(args.epochs) - 1)
        should_save = (ep % save_every == 0) or is_last
        if should_save:
            ckpt = {
                "arch": "kplane_baseline",
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "args": vars(args),
                "kplane_cfg": {
                    "plane_resolutions": list(cfg.plane_resolutions),
                    "plane_channels": int(cfg.plane_channels),
                    "hidden_dim": int(cfg.hidden_dim),
                    "fusion": str(cfg.fusion),
                },
                "epoch": ep,
                "step": step,
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt, ckpt_path)
            if save_last:
                torch.save(ckpt, os.path.join(args.save_dir, "last.pt"))


if __name__ == "__main__":
    main()
