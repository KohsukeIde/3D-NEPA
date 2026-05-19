#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.models.viewaction_nepa import ViewActionNEPA
from viewaction_nepa.models.simple_point_encoder import SimplePointEncoder


def collate(batch):
    out = {}
    for k in batch[0].keys():
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def write_stats(writer, step: int, stats: dict):
    row = {"step": step, **stats}
    writer.writerow(row)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init(seed: int):
    def _init(worker_id: int) -> None:
        worker_seed = seed + worker_id + 1
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(raw)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[error] CUDA requested but torch.cuda.is_available() is false")
    return dev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", default="action", choices=["action", "no_action", "shuffled_action", "action_only"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--latent-dim", type=int, default=384)
    ap.add_argument("--ema-momentum", type=float, default=0.996)
    ap.add_argument("--inverse-weight", type=float, default=0.1)
    ap.add_argument("--variance-weight", type=float, default=0.0)
    ap.add_argument("--contrast-weight", type=float, default=0.0)
    ap.add_argument("--contrast-temperature", type=float, default=0.1)
    ap.add_argument("--hard-contrast-weight", type=float, default=0.0)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--drop-last", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    graph = torch.load if False else None
    # Infer action dimensions from dataset graph.
    ds = ViewActionDataset(args.cache_root, split="train", mode="pair", max_shapes=args.max_shapes, seed=args.seed)
    if len(ds.files) == 0 or len(ds) == 0:
        raise SystemExit(f"[error] empty train dataset for cache={args.cache_root}")
    num_actions = int(ds.action_id.max()) + 1
    action_vec_dim = int(ds.action_vec.shape[1])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=device.type == "cuda", drop_last=args.drop_last, collate_fn=collate,
                    worker_init_fn=make_worker_init(args.seed) if args.num_workers > 0 else None)
    if len(dl) == 0:
        raise SystemExit(
            f"[error] empty dataloader: transitions={len(ds)} batch_size={args.batch_size} drop_last={args.drop_last}"
        )

    encoder = SimplePointEncoder(args.latent_dim)
    model = ViewActionNEPA(encoder=encoder, num_actions=num_actions, action_vec_dim=action_vec_dim,
                           latent_dim=args.latent_dim, ema_momentum=args.ema_momentum,
                           inverse_weight=args.inverse_weight, variance_weight=args.variance_weight,
                           contrast_weight=args.contrast_weight,
                           contrast_temperature=args.contrast_temperature,
                           hard_contrast_weight=args.hard_contrast_weight).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_f = (out / "train_log.csv").open("w", newline="")
    fieldnames = ["step", "epoch", "loss", "loss_fwd", "loss_inv", "loss_var", "loss_contrast", "loss_hard", "hard_acc", "forward_cos_mean", "forward_cos_std",
                  "current_cos_mean", "pred_z_var_mean",
                  "inverse_acc", "online_z_norm_mean", "online_z_norm_std", "online_z_var_mean",
                  "target_z_norm_mean", "target_z_norm_std", "target_z_var_mean"]
    writer = csv.DictWriter(log_f, fieldnames=fieldnames)
    writer.writeheader()
    step = 0
    stats = {"loss": float("nan"), "forward_cos_mean": float("nan"), "inverse_acc": float("nan")}
    print(f"[data] train_shapes={len(ds.files)} transitions={len(ds)} batches_per_epoch={len(dl)} device={device}")
    for epoch in range(args.epochs):
        model.train()
        for batch in dl:
            opt.zero_grad(set_to_none=True)
            loss, stats = model.forward_pair(batch, variant=args.variant, device=device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            stats["epoch"] = epoch
            writer.writerow({k: stats.get(k, step if k == "step" else None) for k in fieldnames})
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        log_f.flush()
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out / f"ckpt_e{epoch+1}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out / "ckpt_last.pth")
            print(f"[epoch {epoch+1}] saved; last loss={stats.get('loss'):.4f} cos={stats.get('forward_cos_mean'):.4f} inv={stats.get('inverse_acc'):.3f}")
        if args.max_steps and step >= args.max_steps:
            break
    log_f.close()


if __name__ == "__main__":
    main()
