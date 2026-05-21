#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import effective_rank, load_model, make_loader, topk_retrieval, write_json_md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-file", default="")
    ap.add_argument("--max-shapes", type=int, default=512)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--variant", default="time", choices=["time", "no_time", "shuffled_time", "time_only", "z_shuffled", "identity"])
    ap.add_argument("--epsilon-mode", default="", choices=["", "independent", "shared"])
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
        epsilon_mode=args.epsilon_mode or ckpt_args.get("epsilon_mode", "independent"),
    )
    preds, tgts, currents = [], [], []
    with torch.no_grad():
        for batch in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            z_t = model.encode_online(b["x_t"])
            z_s = model.encode_target(b["x_s"])
            if args.variant == "identity":
                z_pred = z_t
            else:
                t, s = b["t"], b["s"]
                if args.variant == "no_time":
                    t = torch.zeros_like(t); s = torch.zeros_like(s)
                elif args.variant == "shuffled_time":
                    perm = torch.randperm(t.shape[0], device=t.device)
                    t = t[perm]; s = s[perm]
                if args.variant == "time_only":
                    z_in = torch.zeros_like(z_t)
                elif args.variant == "z_shuffled":
                    perm = torch.randperm(z_t.shape[0], device=z_t.device)
                    z_in = z_t[perm]
                else:
                    z_in = z_t
                z_pred = model.predict(z_in, t, s)
            preds.append(z_pred.cpu()); tgts.append(z_s.cpu()); currents.append(z_t.cpu())
    pred = torch.cat(preds, 0).to(device)
    tgt = torch.cat(tgts, 0).to(device)
    cur = torch.cat(currents, 0).to(device)
    metrics = topk_retrieval(pred, tgt, topk=(1, 3, 5))
    metrics["cos_to_target"] = float(torch.nn.functional.cosine_similarity(pred, tgt, dim=-1).mean().cpu())
    metrics["cos_to_current"] = float(torch.nn.functional.cosine_similarity(pred, cur, dim=-1).mean().cpu())
    metrics["pred_eff_rank"] = effective_rank(pred.cpu())
    metrics["target_eff_rank"] = effective_rank(tgt.cpu())
    metrics["current_eff_rank"] = effective_rank(cur.cpu())
    metrics["pred_var"] = float(pred.cpu().var(dim=0, unbiased=False).mean())
    metrics["variant"] = args.variant
    metrics["epsilon_mode"] = args.epsilon_mode or ckpt_args.get("epsilon_mode", "independent")
    out = Path(args.out_dir)
    write_json_md(metrics, out / f"retrieval_{args.variant}.json", out / f"retrieval_{args.variant}.md", f"Denoising-time retrieval ({args.variant})")
    print(metrics)

if __name__ == "__main__":
    main()
