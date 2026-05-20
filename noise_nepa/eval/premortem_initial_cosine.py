#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset
from noise_nepa.models.simple_point_encoder import SimplePointEncoder
from noise_nepa.data.noise_ops import NoiseSchedule, add_gaussian_noise
from common import write_json_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--max-shapes", type=int, default=500)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--noise-pairs", default="750:350,650:250,500:100")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = NoisePairDataset(args.data_root, split_file=args.split_file or None, max_shapes=args.max_shapes, npoints=args.npoints, fixed_eval=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    enc = SimplePointEncoder(384).to(device).eval()
    schedule = NoiseSchedule()
    metrics = {}
    with torch.no_grad():
        for pair in args.noise_pairs.split(","):
            t_s = pair.strip().split(":")
            tval, sval = int(t_s[0]), int(t_s[1])
            cosines = []
            for batch in loader:
                x0 = batch["x0"].to(device)
                B = x0.shape[0]
                t = torch.full((B,), tval, dtype=torch.long, device=device)
                s = torch.full((B,), sval, dtype=torch.long, device=device)
                xt = add_gaussian_noise(x0, t, schedule)
                xs = add_gaussian_noise(x0, s, schedule)
                zt = enc(xt); zs = enc(xs)
                cosines.append(torch.nn.functional.cosine_similarity(zt, zs, dim=-1).cpu())
            c = torch.cat(cosines)
            metrics[f"cos_mean_t{tval}_s{sval}"] = float(c.mean())
            metrics[f"cos_std_t{tval}_s{sval}"] = float(c.std())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json_md(metrics, out / "initial_cosine.json", out / "initial_cosine.md", "Pre-mortem initial cosine")
    print(metrics)

if __name__ == "__main__":
    main()
