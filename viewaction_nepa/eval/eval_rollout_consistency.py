#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.eval.common import load_model, encode_all_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-shapes", type=int, default=200)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.cache_root, device=device)
    ds = ViewActionDataset(args.cache_root, split=args.split, mode="all_views", max_shapes=args.max_shapes)
    pair = ViewActionDataset(args.cache_root, split=args.split, mode="pair", max_shapes=args.max_shapes)
    edges = torch.as_tensor(pair.edges)
    aid = torch.as_tensor(pair.action_id, dtype=torch.long)
    avec = torch.as_tensor(pair.action_vec, dtype=torch.float32)
    outgoing = {i: torch.nonzero(edges[:, 0] == i, as_tuple=False).view(-1) for i in range(int(edges.max()) + 1)}
    variants = ["action", "no_action", "shuffled_action", "action_only"]
    metrics = {v: {"top1": 0, "mrr": 0.0, "n": 0} for v in variants}
    for item in ds:
        z = F.normalize(encode_all_views(model, item["views"], device=device), dim=-1)
        for e0, (s, t) in enumerate(edges.tolist()):
            outs = outgoing[int(t)]
            if outs.numel() == 0:
                continue
            e1 = int(outs[0].item())
            u = int(edges[e1, 1].item())
            z0 = z[s:s+1].to(device)
            wrong0 = int(outs[0].item())
            for cand in outs.tolist():
                if int(cand) != int(e0):
                    wrong0 = int(cand)
                    break
            outs2 = outgoing[int(t)]
            wrong1 = int(outs2[0].item()) if outs2.numel() else e1
            for cand in outs2.tolist():
                if int(cand) != int(e1):
                    wrong1 = int(cand)
                    break
            for variant in variants:
                if variant == "no_action":
                    z1h = model.predict_next(z0, aid[e0:e0+1].to(device), avec[e0:e0+1].to(device), variant="no_action")
                    z2h = model.predict_next(z1h, aid[e1:e1+1].to(device), avec[e1:e1+1].to(device), variant="no_action")
                elif variant == "shuffled_action":
                    z1h = model.predict_next(z0, aid[wrong0:wrong0+1].to(device), avec[wrong0:wrong0+1].to(device), variant="action")
                    z2h = model.predict_next(z1h, aid[wrong1:wrong1+1].to(device), avec[wrong1:wrong1+1].to(device), variant="action")
                elif variant == "action_only":
                    z1h = model.predict_next(z0, aid[e0:e0+1].to(device), avec[e0:e0+1].to(device), variant="action_only")
                    z2h = model.predict_next(z1h, aid[e1:e1+1].to(device), avec[e1:e1+1].to(device), variant="action_only")
                else:
                    z1h = model.predict_next(z0, aid[e0:e0+1].to(device), avec[e0:e0+1].to(device), variant="action")
                    z2h = model.predict_next(z1h, aid[e1:e1+1].to(device), avec[e1:e1+1].to(device), variant="action")
                z2h = F.normalize(z2h.detach().cpu(), dim=-1)
                sims = (z2h @ z.T).view(-1)
                sims[s] = -1e9
                rank = int((sims > sims[u]).sum().item()) + 1
                metrics[variant]["top1"] += int(rank == 1)
                metrics[variant]["mrr"] += 1.0 / rank
                metrics[variant]["n"] += 1
    summary = {}
    for variant, m in metrics.items():
        n = max(m["n"], 1)
        summary[variant] = {"n": m["n"], "rollout2_top1": m["top1"] / n, "rollout2_mrr": m["mrr"] / n}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    lines = ["# 2-step rollout consistency", "", "| variant | rollout2_top1 | rollout2_mrr | n |", "|---|---:|---:|---:|"]
    for variant, row in summary.items():
        lines.append(f"| {variant} | {row['rollout2_top1']:.4f} | {row['rollout2_mrr']:.4f} | {row['n']} |")
    Path(args.out_md).write_text("\n".join(lines)+"\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
