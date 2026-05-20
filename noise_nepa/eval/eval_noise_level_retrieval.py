#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset, load_points
from noise_nepa.data.noise_ops import add_gaussian_noise
from common import load_model, write_json_md


def parse_levels(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--max-shapes", type=int, default=512)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--variant", default="time", choices=["time", "no_time", "shuffled_time", "identity"])
    ap.add_argument("--t-level", type=int, default=900)
    ap.add_argument("--candidate-levels", default="50,150,300,450,600,750")
    ap.add_argument("--include-current", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_args = load_model(args.ckpt, device=device)
    levels = parse_levels(args.candidate_levels)
    if not levels:
        raise SystemExit("[error] no candidate levels")
    if args.t_level <= max(levels):
        raise SystemExit(f"[error] t-level must be greater than candidate levels: {args.t_level} <= {max(levels)}")

    ds = NoisePairDataset(
        args.data_root,
        split_file=args.split_file or None,
        max_shapes=args.max_shapes,
        npoints=args.npoints,
        num_steps=int(ckpt_args.get("num_steps", 1000)),
        min_gap=int(ckpt_args.get("min_gap", 50)),
        fixed_eval=True,
        seed=args.seed,
    )
    schedule = ds.schedule
    hits1, hits3, mrrs, margins = [], [], [], []
    current_hits = []

    with torch.no_grad():
        for start in range(0, len(ds.files), args.batch_size):
            files = ds.files[start : start + args.batch_size]
            xs_t, cand_xs, labels = [], [], []
            for local_j, path in enumerate(files):
                idx = start + local_j
                rng = np.random.default_rng(args.seed * 1_000_003 + idx)
                x0 = ds._sample_points(load_points(path), rng)
                label = idx % len(levels)
                labels.append(label)
                t = torch.tensor(args.t_level, dtype=torch.long)
                gen_t = torch.Generator().manual_seed(args.seed * 10_000_019 + idx * 97 + 1)
                xs_t.append(add_gaussian_noise(x0, t, schedule, generator=gen_t))
                cands = []
                for k, s_level in enumerate(levels):
                    gen_s = torch.Generator().manual_seed(args.seed * 10_000_019 + idx * 97 + 100 + k)
                    cands.append(add_gaussian_noise(x0, torch.tensor(s_level, dtype=torch.long), schedule, generator=gen_s))
                if args.include_current:
                    cands.append(xs_t[-1])
                cand_xs.append(torch.stack(cands, dim=0))

            x_t = torch.stack(xs_t, dim=0).to(device)
            cands = torch.stack(cand_xs, dim=0).to(device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)
            B, C = cands.shape[:2]

            z_t = model.encode_online(x_t)
            z_cand = model.encode_target(cands.reshape(B * C, *cands.shape[2:])).reshape(B, C, -1)
            if args.variant == "identity":
                z_pred = z_t
            else:
                t = torch.full((B,), args.t_level, dtype=torch.long, device=device)
                s_vals = torch.tensor([levels[i] for i in labels], dtype=torch.long, device=device)
                if args.variant == "no_time":
                    t = torch.zeros_like(t)
                    s_vals = torch.zeros_like(s_vals)
                elif args.variant == "shuffled_time":
                    perm = torch.randperm(B, device=device)
                    s_vals = s_vals[perm]
                z_pred = model.predict(z_t, t, s_vals)

            sim = torch.einsum(
                "bd,bcd->bc",
                torch.nn.functional.normalize(z_pred, dim=-1),
                torch.nn.functional.normalize(z_cand, dim=-1),
            )
            order = torch.argsort(sim, dim=1, descending=True)
            ranks = (order == labels_t[:, None]).nonzero()[:, 1] + 1
            hits1.append((ranks == 1).float().cpu())
            hits3.append((ranks <= min(3, C)).float().cpu())
            mrrs.append((1.0 / ranks.float()).cpu())
            pos = sim[torch.arange(B, device=device), labels_t]
            masked = sim.clone()
            masked[torch.arange(B, device=device), labels_t] = -1e9
            margins.append((pos - masked.max(dim=1).values).cpu())
            if args.include_current:
                current_hits.append((order[:, 0] == (C - 1)).float().cpu())

    metrics = {
        "variant": args.variant,
        "t_level": args.t_level,
        "candidate_levels": str(levels),
        "num_candidates": len(levels) + int(args.include_current),
        "chance_top1": 1.0 / (len(levels) + int(args.include_current)),
        "top1": float(torch.cat(hits1).mean()),
        "top3": float(torch.cat(hits3).mean()),
        "mrr": float(torch.cat(mrrs).mean()),
        "margin": float(torch.cat(margins).mean()),
    }
    if current_hits:
        metrics["current_select_rate"] = float(torch.cat(current_hits).mean())
    out = Path(args.out_dir)
    write_json_md(
        metrics,
        out / f"noise_level_retrieval_{args.variant}.json",
        out / f"noise_level_retrieval_{args.variant}.md",
        f"Same-shape noise-level retrieval ({args.variant})",
    )
    print(metrics)


if __name__ == "__main__":
    main()
