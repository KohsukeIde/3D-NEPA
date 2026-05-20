#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset
from noise_nepa.models.noise_nepa import NoiseNEPA


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--embed-dim", type=int, default=384)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--min-gap", type=int, default=50)
    ap.add_argument("--ema-momentum", type=float, default=0.996)
    ap.add_argument("--semigroup-weight", type=float, default=0.0)
    ap.add_argument("--var-weight", type=float, default=0.0)
    ap.add_argument("--variant", default="time", choices=["time", "no_time", "shuffled_time"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def apply_variant(batch, variant: str):
    if variant == "time":
        return batch
    if variant == "no_time":
        # Collapse time conditioning to a constant pair while keeping inputs intact.
        batch = dict(batch)
        batch["t"] = torch.zeros_like(batch["t"])
        batch["s"] = torch.zeros_like(batch["s"])
        if "r" in batch:
            batch["r"] = torch.zeros_like(batch["r"])
        return batch
    if variant == "shuffled_time":
        batch = dict(batch)
        perm = torch.randperm(batch["t"].shape[0], device=batch["t"].device)
        batch["t"] = batch["t"][perm]
        batch["s"] = batch["s"][perm]
        if "r" in batch:
            batch["r"] = batch["r"][perm]
        return batch
    return batch


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    raw_device = "cuda" if args.device == "auto" else args.device
    device = torch.device(raw_device if torch.cuda.is_available() or raw_device == "cpu" else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    ds = NoisePairDataset(
        root=args.data_root,
        split_file=args.split_file or None,
        max_shapes=args.max_shapes,
        npoints=args.npoints,
        num_steps=args.num_steps,
        min_gap=args.min_gap,
        seed=args.seed,
        deterministic=True,
    )
    if len(ds) < 2:
        raise SystemExit(f"[error] need at least 2 shapes, got {len(ds)}")
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    drop_last = len(train_ds) >= args.batch_size
    loader_gen = torch.Generator().manual_seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = args.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
        generator=loader_gen,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    model = NoiseNEPA(
        embed_dim=args.embed_dim,
        num_steps=args.num_steps,
        ema_momentum=args.ema_momentum,
        semigroup_weight=args.semigroup_weight,
        var_weight=args.var_weight,
    ).to(device)
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    log_path = out_dir / "train_log.jsonl"
    best_val = 1e9
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            batch = apply_variant(batch, args.variant)
            opt.zero_grad(set_to_none=True)
            res = model(batch)
            res.loss.backward()
            opt.step()
            model.update_target()
            if step % args.log_every == 0:
                row = {
                    "epoch": epoch,
                    "step": step,
                    "split": "train",
                    "loss": float(res.loss.detach().cpu()),
                    "loss_forward": float(res.loss_forward.cpu()),
                    "loss_semigroup": float(res.loss_semigroup.cpu()),
                    "cos_forward": float(res.cos_forward.cpu()),
                    "z_var": float(res.z_var.cpu()),
                }
                with log_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(row, flush=True)
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        # validation
        model.eval()
        vals = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                batch = apply_variant(batch, args.variant)
                res = model(batch)
                vals.append(float(res.loss.cpu()))
        val = sum(vals) / max(1, len(vals))
        row = {"epoch": epoch, "step": step, "split": "val", "loss": val}
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "ckpt_best.pth")
        torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, out_dir / "ckpt_last.pth")
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[done] out={out_dir} best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
