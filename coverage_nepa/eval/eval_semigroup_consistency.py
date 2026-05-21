#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch

from coverage_nepa.data.coverage_dataset import select_cache_files
from coverage_nepa.eval.common import load_ckpt


def predict(model, z, levels, a, b, variant):
    if variant == "identity":
        return z
    return model.predict(
        z,
        levels[a:a+1],
        levels[b:b+1],
        variant="level_only" if variant == "level_only" else ("no_level" if variant == "no_level" else "conditioned"),
    )


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
    sums = {
        v: {"direct_ok": 0, "rollout_ok": 0, "total": 0, "dr_cos": 0.0, "target_direct": 0.0, "target_rollout": 0.0}
        for v in variants
    }
    for pi, p in enumerate(files):
        data = np.load(p, allow_pickle=True)
        cov = torch.from_numpy(np.asarray(data["coverage"], dtype=np.float32)).to(device)
        wrong_data = np.load(files[(pi + 1) % len(files)] if len(files) > 1 else p, allow_pickle=True)
        wrong_cov = torch.from_numpy(np.asarray(wrong_data["coverage"], dtype=np.float32)).to(device)
        K = cov.shape[0]
        if K < 3: continue
        levels = torch.arange(K, device=device, dtype=torch.long)
        with torch.no_grad():
            cand = model.encode_target(cov)
            online = model.encode_online(cov)
            wrong_online = model.encode_online(wrong_cov)
            for k in range(K - 2):
                for l in range(k + 1, K - 1):
                    for m in range(l + 1, K):
                        zk_base = online[k:k+1]
                        wrong_k = min(k, wrong_cov.shape[0] - 1)
                        wrong_z = wrong_online[wrong_k:wrong_k+1]
                        for variant in variants:
                            zk = wrong_z if variant == "z_shuffled" else zk_base
                            direct = predict(model, zk, levels, k, m, variant)
                            mid = predict(model, zk, levels, k, l, variant)
                            rollout = predict(model, mid, levels, l, m, variant)
                            sd = (direct @ cand.t()).view(-1)
                            sr = (rollout @ cand.t()).view(-1)
                            acc = sums[variant]
                            acc["direct_ok"] += int(torch.argmax(sd).item() == m)
                            acc["rollout_ok"] += int(torch.argmax(sr).item() == m)
                            acc["total"] += 1
                            acc["dr_cos"] += float((direct * rollout).sum().item())
                            acc["target_direct"] += float((direct * cand[m:m+1]).sum().item())
                            acc["target_rollout"] += float((rollout * cand[m:m+1]).sum().item())
    out = {
        "split": args.split,
        "num_shapes": len(files),
        "variants": {},
    }
    for variant, acc in sums.items():
        total = acc["total"]
        out["variants"][variant] = {
            "n": total,
            "direct_top1": acc["direct_ok"] / max(total, 1),
            "rollout_top1": acc["rollout_ok"] / max(total, 1),
            "direct_rollout_cos": acc["dr_cos"] / max(total, 1),
            "direct_target_cos": acc["target_direct"] / max(total, 1),
            "rollout_target_cos": acc["target_rollout"] / max(total, 1),
        }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    lines = ["# Coverage semigroup consistency", "", "| variant | direct_top1 | rollout_top1 | direct_rollout_cos | direct_target_cos | rollout_target_cos | n |", "|---|---:|---:|---:|---:|---:|---:|"]
    for v, r in out["variants"].items():
        lines.append(f"| {v} | {r['direct_top1']:.4f} | {r['rollout_top1']:.4f} | {r['direct_rollout_cos']:.4f} | {r['direct_target_cos']:.4f} | {r['rollout_target_cos']:.4f} | {r['n']} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(out)

if __name__ == "__main__":
    main()
