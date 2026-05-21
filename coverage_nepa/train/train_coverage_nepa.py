#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from coverage_nepa.data.coverage_dataset import CoveragePairDataset
from coverage_nepa.models.coverage_nepa import CoverageNEPA


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def move(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--max-levels", type=int, default=12)
    ap.add_argument("--ema-momentum", type=float, default=0.996)
    ap.add_argument("--variant", default="conditioned", choices=["conditioned", "no_level", "level_only", "z_shuffled"])
    ap.add_argument("--pair-mode", default="forward", choices=["forward", "adjacent"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = CoveragePairDataset(args.cache_root, split="train", seed=args.seed, deterministic=False, pair_mode=args.pair_mode)
    drop_last = len(ds) >= args.batch_size
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=drop_last)
    model = CoverageNEPA(dim=args.dim, hidden=args.hidden, max_levels=args.max_levels, ema_momentum=args.ema_momentum).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        sums = {"loss": 0.0, "cos": 0.0, "current_cos": 0.0, "z_var": 0.0, "z_norm": 0.0}
        n = 0
        for batch in loader:
            batch = move(batch, device)
            outd = model(batch, variant=args.variant)
            loss = outd["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            model.update_target()
            bs = batch["x_k"].shape[0]
            n += bs
            for k in sums:
                sums[k] += float(outd[k].detach().cpu()) * bs
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        row = {"epoch": epoch, **{k: v / max(n, 1) for k, v in sums.items()}}
        history.append(row)
        print(json.dumps(row))
        (out / "history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "args": vars(args)}, out / "ckpt_last.pth")
        if args.max_steps and step >= args.max_steps:
            break
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
