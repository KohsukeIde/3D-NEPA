#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset
from noise_nepa.data.noise_ops import NoiseSchedule, add_gaussian_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-shapes", type=int, default=3)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--levels", default="0,100,250,500,750,950")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    ds = NoisePairDataset(args.data_root, max_shapes=args.max_shapes, npoints=args.npoints, fixed_eval=True)
    sched = NoiseSchedule()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for i in range(min(args.max_shapes, len(ds))):
        item = ds[i]
        x0 = item["x0"]
        fig, axes = plt.subplots(1, len(levels), figsize=(3 * len(levels), 3))
        for ax, tval in zip(axes, levels):
            x = add_gaussian_noise(x0, torch.tensor(tval), sched).numpy()
            ax.scatter(x[:, 0], x[:, 1], s=1)
            ax.set_title(f"t={tval}")
            ax.axis("equal"); ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"noise_levels_{i:03d}.png", dpi=150)
        plt.close(fig)
    print(f"[ok] wrote visualizations to {out}")

if __name__ == "__main__":
    main()
