#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from coverage_nepa.data.coverage_dataset import list_cache_files, select_cache_files
from coverage_nepa.eval.common import load_ckpt


def eval_one(model, path: Path, wrong_path: Path, device: torch.device, variant: str) -> tuple[int, int, float, float]:
    data = np.load(path, allow_pickle=True)
    cov = torch.from_numpy(np.asarray(data["coverage"], dtype=np.float32)).to(device)
    wrong_data = np.load(wrong_path, allow_pickle=True)
    wrong_cov = torch.from_numpy(np.asarray(wrong_data["coverage"], dtype=np.float32)).to(device)
    kmax = cov.shape[0]
    levels = torch.arange(kmax, device=device, dtype=torch.long)
    correct = 0; current = 0; total = 0; margin_sum = 0.0
    with torch.no_grad():
        cand = model.encode_target(cov)
        online = model.encode_online(cov)
        wrong_online = model.encode_online(wrong_cov)
        for k in range(kmax - 1):
            for m in range(k + 1, kmax):
                z_k = online[k:k+1]
                if variant == "identity":
                    pred = z_k
                elif variant == "level_only":
                    pred = model.predict(z_k, levels[k:k+1], levels[m:m+1], variant="level_only")
                elif variant == "no_level":
                    pred = model.predict(z_k, levels[k:k+1], levels[m:m+1], variant="no_level")
                elif variant == "z_shuffled":
                    wk = min(k, wrong_cov.shape[0] - 1)
                    wrong = wrong_online[wk:wk+1]
                    pred = model.predict(wrong, levels[k:k+1], levels[m:m+1], variant="conditioned")
                else:
                    pred = model.predict(z_k, levels[k:k+1], levels[m:m+1], variant="conditioned")
                sim = (pred @ cand.t()).view(-1)
                rank = int(torch.argmax(sim).item())
                correct += int(rank == m)
                current += int(rank == k)
                total += 1
                pos = float(sim[m].item())
                neg = float(torch.max(torch.cat([sim[:m], sim[m+1:]])).item()) if kmax > 1 else pos
                margin_sum += pos - neg
    return correct, total, margin_sum / max(total, 1), current / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--max-shapes", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_ckpt(args.ckpt, device)
    files = select_cache_files(args.cache_root, split=args.split, seed=args.seed)
    if args.max_shapes > 0:
        files = files[: args.max_shapes]
    variants = ["conditioned", "no_level", "level_only", "z_shuffled", "identity"]
    out = {"split": args.split, "num_shapes": len(files), "variants": {}}
    for v in variants:
        c = t = 0; margins = []; current_rates = []
        for i, p in enumerate(files):
            wrong_path = files[(i + 1) % len(files)] if len(files) > 1 else p
            cc, tt, ma, cr = eval_one(model, p, wrong_path, device, v)
            c += cc; t += tt; margins.append(ma)
            current_rates.append(cr)
        out["variants"][v] = {
            "top1": c / max(t, 1),
            "n": t,
            "margin": float(np.mean(margins)) if margins else 0.0,
            "current_select_rate": float(np.mean(current_rates)) if current_rates else 0.0,
        }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    lines = ["# Coverage retrieval", "", "| variant | top1 | margin | current_select | n |", "|---|---:|---:|---:|---:|"]
    for v, r in out["variants"].items():
        lines.append(f"| {v} | {r['top1']:.4f} | {r['margin']:.4f} | {r['current_select_rate']:.4f} | {r['n']} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
