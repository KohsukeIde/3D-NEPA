#!/usr/bin/env python3
"""Lightweight textual verification that PointGPT.py has PosetNEPA order modes."""
from __future__ import annotations
import argparse
from pathlib import Path

REQUIRED = [
    "axis_sorting",
    "radial_sorting",
    "farthest_greedy_sorting",
    "bfs_shell_sorting",
    "geodesic_shell_sorting",
    "diffusion_shell_sorting",
    "fixed_random_sorting",
    "torch.linalg.eigh(K)",
    'order_mode in {"fixed_random", "stable_random"}',
    'order_mode == "diffusion_shell"',
]
FORBIDDEN = [
    "_last_center_device",
    "torch.linalg.eigh(K[b])",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--target", default="PointGPT/models/PointGPT.py")
    args = ap.parse_args()
    path = Path(args.repo_root) / args.target
    text = path.read_text()
    missing = [x for x in REQUIRED if x not in text]
    forbidden = [x for x in FORBIDDEN if x in text]
    if missing:
        print("[missing]")
        for x in missing:
            print(" -", x)
        raise SystemExit(1)
    if forbidden:
        print("[forbidden]")
        for x in forbidden:
            print(" -", x)
        raise SystemExit(1)
    print("[ok] PosetNEPA order-mode patch appears installed")

if __name__ == "__main__":
    main()
