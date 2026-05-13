#!/usr/bin/env python3
"""Extract ScanObjectNN fine-tune accuracies for the mask/order pre-flight.

The fine-tune runner stores the best validation accuracy inside checkpoint
metadata. This script reads ckpt-best/ckpt-last when available and falls back to
parsing validation log lines. It emits one row per run_id with columns consumed
by 05_summarize_pf3.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SPLIT_ALIASES = {
    "obj_bg": "objbg",
    "objbg": "objbg",
    "obj_only": "objonly",
    "objonly": "objonly",
    "hardest": "hardest",
    "pb_t50_rs": "hardest",
    "pb": "hardest",
}
OUT_COL = {
    "objbg": "obj_bg",
    "objonly": "obj_only",
    "hardest": "pb_t50_rs",
}
FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
VAL_RE = re.compile(rf"\[Validation\].*acc\s*=\s*({FLOAT})")


def normalize_splits(raw: str) -> list[str]:
    splits: list[str] = []
    for item in [x.strip().lower() for x in raw.split(",") if x.strip()]:
        if item not in SPLIT_ALIASES:
            raise SystemExit(f"[error] unsupported split: {item}")
        split = SPLIT_ALIASES[item]
        if split not in splits:
            splits.append(split)
    return splits


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict) and "acc" in value:
        return to_float(value["acc"])
    try:
        return float(value)
    except Exception:
        return None


def acc_from_checkpoint(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        import torch  # type: ignore

        state = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(state, dict):
        return None
    for key in ("best_metrics", "metrics"):
        acc = to_float(state.get(key))
        if acc is not None:
            return acc
    return None


def best_acc_from_logs(exp_dir: Path) -> float | None:
    vals: list[float] = []
    for log in sorted(exp_dir.rglob("*.log")):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue
        for match in VAL_RE.finditer(text):
            vals.append(float(match.group(1)))
    return max(vals) if vals else None


def exp_root(pointgpt: Path, cfg_rel: str) -> Path:
    cfg = Path(cfg_rel)
    return pointgpt / "experiments" / cfg.stem / cfg.parent.name


def find_exp_dir(pointgpt: Path, ft_cfg_rel: str, exp_prefix: str, run_tag: str) -> Path | None:
    root = exp_root(pointgpt, ft_cfg_rel)
    if not root.exists():
        return None
    if run_tag:
        candidates = [p for p in root.glob(exp_prefix + f"*{run_tag}*") if p.is_dir()]
    else:
        candidates = [p for p in root.glob(exp_prefix + "*") if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--splits", default="hardest")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--out-csv", default="posetnepa_mask_order_preflight/generated/finetune_results.csv")
    ap.add_argument("--out-md", default="posetnepa_mask_order_preflight/generated/finetune_results.md")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    manifest = json.loads((repo / args.manifest).read_text())
    splits = normalize_splits(args.splits)

    rows = []
    for entry in manifest["runs"]:
        row: dict[str, str] = {
            "run_id": entry["run_id"],
            "order": entry["order"],
            "mask_ratio": str(entry["mask_ratio"]),
            "group_mode": entry["group_mode"],
            "obj_bg": "",
            "obj_only": "",
            "pb_t50_rs": "",
            "shapenetpart_inst_miou": "",
        }
        for split in splits:
            ft_cfg = entry["finetune_configs"][split]
            exp_prefix = f"{entry['finetune_exp_prefix']}_{split}_from_"
            exp_dir = find_exp_dir(pointgpt, ft_cfg, exp_prefix, args.run_tag)
            out_col = OUT_COL[split]
            if exp_dir is None:
                row[f"{split}_exp_dir"] = ""
                row[f"{split}_status"] = "missing"
                continue
            row[f"{split}_exp_dir"] = str(exp_dir)
            acc = acc_from_checkpoint(exp_dir / "ckpt-best.pth")
            if acc is None:
                acc = acc_from_checkpoint(exp_dir / "ckpt-last.pth")
            if acc is None:
                acc = best_acc_from_logs(exp_dir)
            if acc is None:
                row[f"{split}_status"] = "no_metric"
            else:
                row[out_col] = f"{acc:.6f}"
                row[f"{split}_status"] = "ok"
        rows.append(row)

    fields = [
        "run_id",
        "order",
        "mask_ratio",
        "group_mode",
        "obj_bg",
        "obj_only",
        "pb_t50_rs",
        "shapenetpart_inst_miou",
    ]
    extra = sorted({k for r in rows for k in r if k not in fields})
    out_csv = repo / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra)
        writer.writeheader()
        writer.writerows(rows)

    out_md = repo / args.out_md
    with out_md.open("w") as f:
        f.write("# Fine-tune results\n\n")
        f.write("| run | order | mask | obj_bg | obj_only | PB_T50_RS |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['run_id']} | {r['order']} | {r['mask_ratio']} | "
                f"{r['obj_bg']} | {r['obj_only']} | {r['pb_t50_rs']} |\n"
            )
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
