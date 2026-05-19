#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from viewaction_nepa.data.viewaction_dataset import ViewActionDataset
from viewaction_nepa.data.camera_graph import shortest_path_distance
from viewaction_nepa.eval.common import load_model, encode_all_views


def main() -> None:
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
    pair_ds = ViewActionDataset(args.cache_root, split=args.split, mode="pair", max_shapes=args.max_shapes)
    edges = pair_ds.edges
    aid = torch.as_tensor(pair_ds.action_id, dtype=torch.long)
    avec = torch.as_tensor(pair_ds.action_vec, dtype=torch.float32)
    num_views = int(edges.max()) + 1
    dist = shortest_path_distance(num_views, edges)
    outgoing = {i: np.nonzero(edges[:, 0] == i)[0] for i in range(num_views)}
    variants = ["action", "no_action", "shuffled_action", "action_only", "random", "oracle"]
    metrics = {v: {"total": 0, "correct": 0, "reduce": 0, "delta_sum": 0.0} for v in variants}
    rng = np.random.default_rng(0)
    for item in ds:
        views = item["views"]
        z = F.normalize(encode_all_views(model, views, device=device), dim=-1)
        for src in range(num_views):
            for goal in range(num_views):
                if src == goal or len(outgoing[src]) == 0:
                    continue
                cand = outgoing[src]
                z0 = z[src:src+1].to(device)
                # Oracle best one-step action is any outgoing edge that minimizes graph distance to target.
                best_dist = min(int(dist[int(edges[e,1]), goal]) for e in cand)
                curr_dist = int(dist[src, goal])
                for variant in variants:
                    if variant == "random":
                        chosen_edge_v = int(cand[int(rng.integers(len(cand)))])
                        chosen_next_v = int(edges[chosen_edge_v, 1])
                    elif variant == "oracle":
                        chosen_next_v = int(edges[cand[0], 1])
                        for e in cand:
                            nxt = int(edges[e, 1])
                            if int(dist[nxt, goal]) == best_dist:
                                chosen_next_v = nxt
                                break
                    else:
                        preds = []
                        for offset, ei in enumerate(cand):
                            if variant == "no_action":
                                pred = model.predict_next(z0, aid[ei:ei+1].to(device), avec[ei:ei+1].to(device), variant="no_action")
                            elif variant == "shuffled_action":
                                wrong_ei = cand[(offset + 1) % len(cand)]
                                pred = model.predict_next(z0, aid[wrong_ei:wrong_ei+1].to(device), avec[wrong_ei:wrong_ei+1].to(device), variant="action")
                            elif variant == "action_only":
                                pred = model.predict_next(z0, aid[ei:ei+1].to(device), avec[ei:ei+1].to(device), variant="action_only")
                            else:
                                pred = model.predict_next(z0, aid[ei:ei+1].to(device), avec[ei:ei+1].to(device), variant="action")
                            preds.append(F.normalize(pred.detach().cpu(), dim=-1))
                        preds = torch.cat(preds, dim=0)
                        sims = preds @ z[goal:goal+1].T
                        chosen_idx = int(torch.argmax(sims.view(-1)).item())
                        chosen_edge_v = cand[chosen_idx]
                        chosen_next_v = int(edges[chosen_edge_v, 1])
                    chosen_dist_v = int(dist[chosen_next_v, goal])
                    m = metrics[variant]
                    m["correct"] += int(chosen_dist_v == best_dist)
                    m["reduce"] += int(chosen_dist_v < curr_dist)
                    m["delta_sum"] += float(chosen_dist_v - curr_dist)
                    m["total"] += 1
    summary = {}
    for variant, m in metrics.items():
        total = max(m["total"], 1)
        summary[variant] = {
            "n": m["total"],
            "best_step_acc": m["correct"] / total,
            "distance_reduction_rate": m["reduce"] / total,
            "mean_delta_dist": m["delta_sum"] / total,
        }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2))
    lines = ["# Goal-view planning", "", "| variant | best_step_acc | distance_reduction_rate | mean_delta_dist | n |", "|---|---:|---:|---:|---:|"]
    for variant, row in summary.items():
        lines.append(
            f"| {variant} | {row['best_step_acc']:.4f} | {row['distance_reduction_rate']:.4f} | {row['mean_delta_dist']:.4f} | {row['n']} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
