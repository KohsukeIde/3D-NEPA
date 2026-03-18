from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from nepa3d.tracks.kplane.data.udfdist_worldv3_dataset import build_worldv3_udfdist_loader
from nepa3d.tracks.kplane.models.kplane import KPlaneConfig, KPlaneRegressor
from nepa3d.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_udfdist_worldv3")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default="kplane_udfdist_worldv3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--plane_type", type=str, default="kplane", choices=["kplane", "triplane"])
    p.add_argument("--fusion", type=str, default="auto", choices=["auto", "sum", "product", "rg_product"])
    p.add_argument("--product_rank_groups", type=int, default=0)
    p.add_argument("--product_group_reduce", type=str, default="sum", choices=["sum", "mean"])
    p.add_argument("--plane_resolutions", type=str, default="64")
    p.add_argument("--plane_channels", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))
    save_dir = Path(str(args.save_dir)) / str(args.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")

    fusion = str(args.fusion)
    if fusion == "auto":
        fusion = "sum" if str(args.plane_type) == "triplane" else "product"
    cfg = KPlaneConfig.from_args(
        plane_resolutions=args.plane_resolutions,
        plane_channels=int(args.plane_channels),
        hidden_dim=int(args.hidden_dim),
        fusion=fusion,
        product_rank_groups=int(args.product_rank_groups),
        product_group_reduce=str(args.product_group_reduce),
    )
    model = KPlaneRegressor(cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    dl, info = build_worldv3_udfdist_loader(
        str(args.mix_config_path),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        mode="train",
        seed=int(args.seed),
    )
    print("[mix]", json.dumps(info, indent=2))
    global_step = 0
    for ep in range(int(args.epochs)):
        model.train()
        losses = []
        for batch in dl:
            ctx_xyz = batch["ctx_xyz"].to(device, non_blocking=True).float()
            ctx_dist = batch["ctx_dist"].to(device, non_blocking=True).float()
            qry_xyz = batch["qry_xyz"].to(device, non_blocking=True).float()
            qry_dist = batch["qry_dist"].to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            pred, *_ = model(ctx_xyz, ctx_dist, qry_xyz)
            loss = F.mse_loss(pred, qry_dist)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            if global_step % 100 == 0:
                print(f"ep={ep} step={global_step} loss={losses[-1]:.6f}")
            global_step += 1
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch] ep={ep} loss_mean={ep_loss:.6f}")
        if (ep % max(1, int(args.save_every)) == 0) or (ep == int(args.epochs) - 1):
            ckpt = {
                "arch": "kplane_udfdist_worldv3",
                "model": model.state_dict(),
                "args": vars(args),
                "kplane_cfg": {
                    "plane_resolutions": list(cfg.plane_resolutions),
                    "plane_channels": int(cfg.plane_channels),
                    "hidden_dim": int(cfg.hidden_dim),
                    "fusion": str(cfg.fusion),
                    "product_rank_groups": int(cfg.product_rank_groups),
                    "product_group_reduce": str(cfg.product_group_reduce),
                },
                "epoch": ep,
                "global_step": int(global_step),
            }
            torch.save(ckpt, save_dir / f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt, save_dir / "ckpt_final.pt")


if __name__ == "__main__":
    main()
