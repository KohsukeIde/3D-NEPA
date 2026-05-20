#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from noise_nepa.data.noise_dataset import NoisePairDataset
from noise_nepa.models.simple_point_encoder import SimplePointEncoder
from noise_nepa.data.noise_ops import NoiseSchedule, add_gaussian_noise
from common import write_json_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--max-shapes", type=int, default=200)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--levels", default="0,100,250,500,750,950")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    ds = NoisePairDataset(args.data_root, split_file=args.split_file or None, max_shapes=args.max_shapes, npoints=args.npoints, fixed_eval=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    enc = SimplePointEncoder(384).to(device).eval()
    schedule = NoiseSchedule()
    step_cos = []
    endpoint_cos = []
    with torch.no_grad():
        for batch in loader:
            x0 = batch["x0"].to(device)
            B = x0.shape[0]
            zs = []
            for tval in levels:
                t = torch.full((B,), tval, dtype=torch.long, device=device)
                zs.append(enc(add_gaussian_noise(x0, t, schedule)))
            for a, b in zip(zs[:-1], zs[1:]):
                step_cos.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).cpu())
            endpoint_cos.append(torch.nn.functional.cosine_similarity(zs[0], zs[-1], dim=-1).cpu())
    step = torch.cat(step_cos); end = torch.cat(endpoint_cos)
    metrics = {"levels": str(levels), "step_cos_mean": float(step.mean()), "step_cos_std": float(step.std()), "endpoint_cos_mean": float(end.mean()), "endpoint_cos_std": float(end.std())}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json_md(metrics, out / "latent_path.json", out / "latent_path.md", "Pre-mortem latent path smoothness")
    print(metrics)

if __name__ == "__main__":
    main()
