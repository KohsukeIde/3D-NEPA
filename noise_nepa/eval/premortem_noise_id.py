#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

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
    ap.add_argument("--bins", default="50,200,400,600,800")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bins = [int(x) for x in args.bins.split(",") if x.strip()]
    ds = NoisePairDataset(args.data_root, split_file=args.split_file or None, max_shapes=args.max_shapes, npoints=args.npoints, fixed_eval=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    enc = SimplePointEncoder(384).to(device).eval()
    schedule = NoiseSchedule()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            x0 = batch["x0"].to(device)
            B = x0.shape[0]
            for j, tval in enumerate(bins):
                t = torch.full((B,), tval, dtype=torch.long, device=device)
                xt = add_gaussian_noise(x0, t, schedule)
                feats.append(enc(xt).cpu())
                labels.append(torch.full((B,), j, dtype=torch.long))
    X = torch.cat(feats); y = torch.cat(labels)
    perm = torch.randperm(X.shape[0])
    ntr = int(0.8 * X.shape[0])
    tr, te = perm[:ntr], perm[ntr:]
    clf = torch.nn.Linear(X.shape[1], len(bins)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xtr, ytr = X[tr].to(device), y[tr].to(device)
    Xte, yte = X[te].to(device), y[te].to(device)
    for _ in range(args.epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(clf(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        train_acc = (clf(Xtr).argmax(-1) == ytr).float().mean()
        test_acc = (clf(Xte).argmax(-1) == yte).float().mean()
    metrics = {"num_bins": len(bins), "chance": 1.0 / len(bins), "train_acc": float(train_acc), "test_acc": float(test_acc)}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json_md(metrics, out / "noise_id.json", out / "noise_id.md", "Pre-mortem noise-level identifiability")
    print(metrics)

if __name__ == "__main__":
    main()
