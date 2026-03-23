from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from nepa3d.tracks.kplane.data.udfdist_worldv3_dataset import (
    build_worldv3_udfdist_mixed,
    collate_udfdist_worldv3,
)
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering_udfdist_continuous import (
    PrimitiveAnsweringUDFDistanceContinuousModel,
)
from nepa3d.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("pretrain_udfdist_continuous")
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="runs/cqa")
    p.add_argument("--run_name", type=str, default="cqa_udfdist_continuous")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--save_every_steps", type=int, default=500)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop_path", type=float, default=0.0)
    p.add_argument("--backbone_impl", type=str, default="nepa2d", choices=["nepa2d", "pointmae", "legacy"])
    p.add_argument("--num_groups", type=int, default=64)
    p.add_argument("--group_size", type=int, default=32)
    p.add_argument("--patch_center_mode", type=str, default="fps", choices=["fps", "first"])
    p.add_argument("--patch_fps_random_start", type=int, default=1, choices=[0, 1])
    p.add_argument("--local_encoder", type=str, default="pointmae_conv", choices=["mlp", "pointmae_conv"])
    p.add_argument("--query_type_vocab", type=int, default=6)
    p.add_argument("--generator_depth", type=int, default=2)
    p.add_argument("--distance_floor", type=float, default=0.0)
    return p.parse_args()


def _resolve_max_steps(args: argparse.Namespace, steps_per_epoch: int) -> int:
    if int(args.max_steps) > 0:
        return int(args.max_steps)
    return int(max(1, int(args.epochs) * max(1, int(steps_per_epoch))))


def _resolve_warmup_steps(args: argparse.Namespace, max_steps: int) -> int:
    if int(args.warmup_steps) >= 0:
        return int(args.warmup_steps)
    return int(round(float(max_steps) * max(0.0, float(args.warmup_ratio))))


def _scheduler_scale(step: int, args: argparse.Namespace, max_steps: int, warmup_steps: int) -> float:
    if str(args.lr_scheduler) != "cosine":
        return 1.0
    s = min(max(0, int(step)), int(max_steps))
    if warmup_steps > 0 and s < warmup_steps:
        return float(s + 1) / float(max(1, warmup_steps))
    denom = max(1, int(max_steps) - max(0, int(warmup_steps)))
    t = float(s - max(0, int(warmup_steps))) / float(denom)
    t = min(max(t, 0.0), 1.0)
    min_scale = float(args.min_lr) / float(args.lr)
    cosv = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_scale + (1.0 - min_scale) * cosv


def _save_ckpt(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    save_dir: Path,
    epoch: int,
    global_step: int,
    name: str,
) -> None:
    ckpt = {
        "arch": "cqa_udfdist_continuous",
        "model": model.state_dict(),
        "args": vars(args),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "target_mode": "continuous_udfdistance",
    }
    torch.save(ckpt, save_dir / name)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = torch.device(str(args.device))

    dataset, sampler, info = build_worldv3_udfdist_mixed(
        str(args.mix_config_path),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        mode="train",
        seed=int(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_udfdist_worldv3,
        drop_last=True,
    )

    model = PrimitiveAnsweringUDFDistanceContinuousModel(
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        drop_path=float(args.drop_path),
        backbone_impl=str(args.backbone_impl),
        num_groups=int(args.num_groups),
        group_size=int(args.group_size),
        patch_center_mode=str(args.patch_center_mode),
        patch_fps_random_start=bool(args.patch_fps_random_start),
        local_encoder=str(args.local_encoder),
        query_type_vocab=int(args.query_type_vocab),
        generator_depth=int(args.generator_depth),
        distance_floor=float(args.distance_floor),
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    steps_per_epoch = int(len(loader))
    max_steps = _resolve_max_steps(args, steps_per_epoch)
    warmup_steps = max(0, _resolve_warmup_steps(args, max_steps))
    scheduler = None
    if str(args.lr_scheduler) == "cosine":
        scheduler = optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda st: _scheduler_scale(int(st), args, max_steps, warmup_steps),
        )

    save_dir = Path(str(args.save_dir)) / str(args.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    (save_dir / "mix_info.json").write_text(json.dumps(info, indent=2) + "\n")

    global_step = 0
    stop = False
    for ep in range(int(args.epochs)):
        model.train()
        losses = []
        maes = []
        rmses = []
        pred_stds = []
        for batch in loader:
            ctx_xyz = batch["ctx_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_xyz = batch["qry_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
            qry_dist = batch["qry_dist"].to(device=device, dtype=torch.float32, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(ctx_xyz, qry_xyz).pred_distance
            loss = F.mse_loss(pred, qry_dist)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            if scheduler is not None:
                scheduler.step()

            err = pred.detach() - qry_dist.detach()
            loss_val = float(loss.detach().cpu().item())
            mae_val = float(err.abs().mean().cpu().item())
            rmse_val = float(torch.sqrt((err ** 2).mean()).cpu().item())
            pred_std_val = float(pred.detach().std(unbiased=False).cpu().item())
            losses.append(loss_val)
            maes.append(mae_val)
            rmses.append(rmse_val)
            pred_stds.append(pred_std_val)

            if global_step % 100 == 0:
                lr_now = float(opt.param_groups[0]["lr"])
                print(
                    f"ep={ep} step={global_step} "
                    f"loss={loss_val:.6f} mae={mae_val:.6f} rmse={rmse_val:.6f} "
                    f"pred_std={pred_std_val:.6f} lr={lr_now:.8f}"
                )

            global_step += 1
            if int(args.save_every_steps) > 0 and global_step % int(args.save_every_steps) == 0:
                _save_ckpt(
                    model=model,
                    args=args,
                    save_dir=save_dir,
                    epoch=ep,
                    global_step=global_step,
                    name=f"ckpt_step{global_step}.pt",
                )
            if global_step >= max_steps:
                stop = True
                break
        ep_loss = float(np.mean(losses)) if losses else float("nan")
        ep_mae = float(np.mean(maes)) if maes else float("nan")
        ep_rmse = float(np.mean(rmses)) if rmses else float("nan")
        ep_pred_std = float(np.mean(pred_stds)) if pred_stds else float("nan")
        print(
            f"[epoch] ep={ep} loss_mean={ep_loss:.6f} mae_mean={ep_mae:.6f} "
            f"rmse_mean={ep_rmse:.6f} pred_std_mean={ep_pred_std:.6f}"
        )
        if (ep % max(1, int(args.save_every)) == 0) or stop or (ep == int(args.epochs) - 1):
            _save_ckpt(
                model=model,
                args=args,
                save_dir=save_dir,
                epoch=ep,
                global_step=global_step,
                name=f"ckpt_ep{ep:03d}.pt",
            )
            _save_ckpt(
                model=model,
                args=args,
                save_dir=save_dir,
                epoch=ep,
                global_step=global_step,
                name="ckpt_final.pt",
            )
        if stop:
            break


if __name__ == "__main__":
    main()
