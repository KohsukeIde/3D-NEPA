#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import load_model, make_loader, topk_retrieval, write_json_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--max-shapes", type=int, default=512)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--variant", default="time", choices=["time", "no_time", "shuffled_time", "identity"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_args = load_model(args.ckpt, device=device)
    loader = make_loader(
        args.data_root,
        args.split_file,
        args.max_shapes,
        args.npoints,
        args.batch_size,
        args.num_workers,
        num_steps=int(ckpt_args.get("num_steps", 1000)),
        min_gap=int(ckpt_args.get("min_gap", 50)),
        seed=args.seed,
    )
    direct, roll, target = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            z_t = model.encode_online(b["x_t"])
            z_r = model.encode_target(b["x_r"])
            t, s, r = b["t"], b["s"], b["r"]
            if args.variant == "identity":
                z_r_direct = z_t
                z_r_roll = z_t
            else:
                if args.variant == "no_time":
                    t = torch.zeros_like(t); s = torch.zeros_like(s); r = torch.zeros_like(r)
                elif args.variant == "shuffled_time":
                    perm = torch.randperm(t.shape[0], device=t.device)
                    t = t[perm]; s = s[perm]; r = r[perm]
                z_s_hat = model.predict(z_t, t, s)
                z_r_roll = model.predict(z_s_hat, s, r)
                z_r_direct = model.predict(z_t, t, r)
            direct.append(z_r_direct.cpu()); roll.append(z_r_roll.cpu()); target.append(z_r.cpu())
    direct = torch.cat(direct, 0).to(device)
    roll = torch.cat(roll, 0).to(device)
    target = torch.cat(target, 0).to(device)
    metrics = {}
    metrics.update({f"direct_{k}": v for k, v in topk_retrieval(direct, target, topk=(1, 5)).items()})
    metrics.update({f"rollout_{k}": v for k, v in topk_retrieval(roll, target, topk=(1, 5)).items()})
    metrics["direct_target_cos"] = float(torch.nn.functional.cosine_similarity(direct, target, dim=-1).mean().cpu())
    metrics["rollout_target_cos"] = float(torch.nn.functional.cosine_similarity(roll, target, dim=-1).mean().cpu())
    metrics["direct_rollout_cos"] = float(torch.nn.functional.cosine_similarity(direct, roll, dim=-1).mean().cpu())
    metrics["variant"] = args.variant
    out = Path(args.out_dir)
    write_json_md(metrics, out / f"semigroup_{args.variant}.json", out / f"semigroup_{args.variant}.md", f"Semigroup consistency ({args.variant})")
    print(metrics)

if __name__ == "__main__":
    main()
