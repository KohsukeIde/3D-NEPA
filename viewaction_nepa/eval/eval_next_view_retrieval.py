#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.eval.common import load_model, encode_all_views


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="action", choices=["action", "no_action", "shuffled_action", "action_only", "all"])
    ap.add_argument("--max-shapes", type=int, default=200)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.cache_root, device=device)
    ds = ViewActionDataset(args.cache_root, split=args.split, mode="all_views", max_shapes=args.max_shapes)
    pair_ds = ViewActionDataset(args.cache_root, split=args.split, mode="pair", max_shapes=args.max_shapes)
    edges = torch.as_tensor(pair_ds.edges)
    action_id = torch.as_tensor(pair_ds.action_id, dtype=torch.long)
    action_vec = torch.as_tensor(pair_ds.action_vec, dtype=torch.float32)
    variants = ["action", "no_action", "shuffled_action", "action_only"] if args.variant == "all" else [args.variant]
    metrics = {
        v: {"top1": 0, "top3": 0, "top5": 0, "mrr": 0.0, "n": 0}
        for v in variants
    }
    for item in ds:
        views = item["views"]
        z_views = F.normalize(encode_all_views(model, views, device=device), dim=-1)
        for ei, (s, t) in enumerate(edges.tolist()):
            z0 = z_views[s:s+1].to(device)
            aid = action_id[ei:ei+1].to(device)
            avec = action_vec[ei:ei+1].to(device)
            wrong = torch.nonzero((edges[:, 0] == s) & (edges[:, 1] != t), as_tuple=False).view(-1)
            wrong_ei = int(wrong[0].item()) if wrong.numel() else int(ei)
            for variant in variants:
                if variant == "no_action":
                    pred = model.predict_next(z0, aid, avec, variant="no_action")
                elif variant == "shuffled_action":
                    pred = model.predict_next(
                        z0,
                        action_id[wrong_ei:wrong_ei+1].to(device),
                        action_vec[wrong_ei:wrong_ei+1].to(device),
                        variant="action",
                    )
                elif variant == "action_only":
                    pred = model.predict_next(z0, aid, avec, variant="action_only")
                else:
                    pred = model.predict_next(z0, aid, avec, variant="action")
                pred = F.normalize(pred.detach().cpu(), dim=-1)
                sims = (pred @ z_views.T).view(-1)
                sims[s] = -1e9  # exclude current view
                rank = int((sims > sims[t]).sum().item()) + 1
                m = metrics[variant]
                m["top1"] += int(rank <= 1)
                m["top3"] += int(rank <= 3)
                m["top5"] += int(rank <= 5)
                m["mrr"] += 1.0 / rank
                m["n"] += 1
    summary = {}
    for variant, m in metrics.items():
        n = max(m["n"], 1)
        summary[variant] = {k: (v / n if k != "n" else v) for k, v in m.items()}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    md = ["# Next-view retrieval", "", "| variant | top1 | top3 | top5 | MRR | n |", "|---|---:|---:|---:|---:|---:|"]
    for variant, row in summary.items():
        md.append(
            f"| {variant} | {row['top1']:.4f} | {row['top3']:.4f} | {row['top5']:.4f} | {row['mrr']:.4f} | {row['n']} |"
        )
    Path(args.out_md).write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
