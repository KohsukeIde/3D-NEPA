#!/usr/bin/env python3
"""Generate PointGPT-S pretrain/finetune configs for mask/order pre-flight.

This script creates a matrix over:
  - order_mode: simplified_morton, fixed_random, diffusion_shell, etc.
  - mask_ratio: 0.0, 0.3, 0.7
while keeping group_mode fixed by default.

The intent is to test whether order/filtration matters independently of grouping,
and whether its effect is masked by PointGPT-style masking.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    import yaml
except Exception as e:
    raise SystemExit("[error] PyYAML is required. Install with: pip install pyyaml") from e


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def deep_set_mask_ratio(cfg: dict, mask: float) -> dict:
    cfg = copy.deepcopy(cfg)
    # The repo uses model.transformer_config.mask_ratio in PointGPT-S configs.
    model = cfg.setdefault("model", {})
    tcfg = model.setdefault("transformer_config", {})
    tcfg["mask_ratio"] = float(mask)
    # Some configs may also carry it top-level; keep consistent if present.
    if "mask_ratio" in cfg:
        cfg["mask_ratio"] = float(mask)
    return cfg


def set_common(cfg: dict, order: str, group: str, max_epoch: int | None, total_bs: int | None, mask: float | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    model = cfg.setdefault("model", {})
    model["order_mode"] = str(order)
    model["group_mode"] = str(group)
    if mask is not None:
        cfg = deep_set_mask_ratio(cfg, mask)
    if max_epoch is not None:
        cfg["max_epoch"] = int(max_epoch)
        if isinstance(cfg.get("scheduler"), dict) and isinstance(cfg["scheduler"].get("kwargs"), dict):
            cfg["scheduler"]["kwargs"]["epochs"] = int(max_epoch)
    if total_bs is not None:
        cfg["total_bs"] = int(total_bs)
    return cfg


def rel_to_pointgpt(path: Path, pointgpt_dir: Path) -> str:
    return str(path.resolve().relative_to(pointgpt_dir.resolve()))


def safe_float(x: str) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--orders", default="simplified_morton,fixed_random,diffusion_shell")
    ap.add_argument("--masks", default="0.0,0.3,0.7")
    ap.add_argument("--group-mode", default="fps_knn")
    ap.add_argument("--base-pretrain", default="cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0.yaml")
    ap.add_argument("--base-ft-objbg", default="cfgs/PointGPT-S/finetune_scan_objbg.yaml")
    ap.add_argument("--base-ft-objonly", default="cfgs/PointGPT-S/finetune_scan_objonly.yaml")
    ap.add_argument("--base-ft-hardest", default="cfgs/PointGPT-S/finetune_scan_hardest.yaml")
    ap.add_argument("--max-epoch", type=int, default=30)
    ap.add_argument("--ft-max-epoch", type=int, default=50)
    ap.add_argument("--total-bs", type=int, default=None)
    ap.add_argument("--ft-total-bs", type=int, default=None)
    ap.add_argument("--out-dir", default="PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = (repo / args.pointgpt_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    manifest_path = repo / args.manifest
    orders = [x.strip() for x in args.orders.split(",") if x.strip()]
    masks = [float(x.strip()) for x in args.masks.split(",") if x.strip()]

    base_pre = load_yaml(pointgpt / args.base_pretrain)
    base_fts = {
        "objbg": load_yaml(pointgpt / args.base_ft_objbg),
        "objonly": load_yaml(pointgpt / args.base_ft_objonly),
        "hardest": load_yaml(pointgpt / args.base_ft_hardest),
    }

    manifest = {
        "purpose": "mask-aware order/filtration pre-flight for PosetNEPA",
        "group_mode_fixed": args.group_mode,
        "orders": orders,
        "masks": masks,
        "max_epoch": args.max_epoch,
        "ft_max_epoch": args.ft_max_epoch,
        "runs": [],
    }

    for mask in masks:
        mtag = f"m{safe_float(mask)}"
        for order in orders:
            safe_order = order.replace("/", "_")
            run_id = f"{safe_order}_{mtag}"
            pre_cfg = set_common(base_pre, order, args.group_mode, args.max_epoch, args.total_bs, mask=mask)
            pre_path = out_dir / f"pretrain_nepa_{run_id}_e{args.max_epoch}.yaml"
            write_yaml(pre_cfg, pre_path)

            ft_entries = {}
            for split, base_ft in base_fts.items():
                ft_cfg = set_common(base_ft, order, args.group_mode, args.ft_max_epoch, args.ft_total_bs, mask=None)
                ft_path = out_dir / f"finetune_scan_{split}_{run_id}_e{args.ft_max_epoch}.yaml"
                write_yaml(ft_cfg, ft_path)
                ft_entries[split] = rel_to_pointgpt(ft_path, pointgpt)

            manifest["runs"].append({
                "run_id": run_id,
                "order": order,
                "mask_ratio": mask,
                "group_mode": args.group_mode,
                "pretrain_config": rel_to_pointgpt(pre_path, pointgpt),
                "pretrain_exp": f"posetmo_pre_{run_id}_e{args.max_epoch}",
                "finetune_configs": ft_entries,
                "finetune_exp_prefix": f"posetmo_ft_{run_id}",
                "is_naive_maskoff_baseline": order == "simplified_morton" and abs(mask - 0.0) < 1e-9,
                "is_maskon_baseline": order == "simplified_morton" and abs(mask - 0.7) < 1e-9,
            })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote configs to {out_dir}")
    print(f"[done] wrote manifest to {manifest_path}")
    print(f"[matrix] {len(orders)} orders x {len(masks)} masks = {len(manifest['runs'])} runs")
    print("[orders]", ", ".join(orders))
    print("[masks]", ", ".join(map(str, masks)))


if __name__ == "__main__":
    main()
