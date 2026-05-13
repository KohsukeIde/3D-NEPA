#!/usr/bin/env python3
"""Summarize previous-token/copy-baseline diagnostics from pretrain logs.

The PointGPT NEPA diagnostic logs already report cosine(pred, target),
cosine(pred, previous_target), their gap, and copy_win. This script turns those
batch diagnostics into a compact previous-token baseline table:

  model_loss ~= 1 - mean(cos_tgt)
  prev_token_loss ~= 1 - mean(cos_prev)

Positive model_better_by_loss means the learned prediction beats the previous
token baseline. Negative values mean copying/previous-token prediction is still
competitive or better under this diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable


FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?|nan"
EPOCH_RE = re.compile(r"\[Epoch\s+(\d+)/(\d+)\]\[Batch\s+(\d+)/(\d+)\]")
DIAG_PATTERNS = {
    "loss_main": re.compile(rf"loss_main=({FLOAT})"),
    "cos_tgt": re.compile(rf"cos_tgt=({FLOAT})"),
    "cos_prev": re.compile(rf"cos_prev=({FLOAT})"),
    "gap": re.compile(rf"gap=({FLOAT})"),
    "copy_win": re.compile(rf"copy_win=({FLOAT})"),
}


def parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    if raw.lower() == "nan":
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(clean) if clean else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def resolve_manifest(repo: Path, raw: str) -> Path:
    requested = Path(raw)
    path = requested if requested.is_absolute() else repo / requested
    if path.exists():
        return path
    generated = repo / "posetnepa_mask_order_preflight" / "generated"
    candidates = sorted(
        generated.glob("stage1_*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"[info] manifest not found at {path}; using latest Stage 1 manifest: {candidates[0]}")
        return candidates[0]
    raise SystemExit(f"[error] manifest not found: {path}")


def exp_root(pointgpt: Path, cfg_rel: str) -> Path:
    cfg = Path(cfg_rel)
    return pointgpt / "experiments" / cfg.stem / cfg.parent.name


def find_pretrain_dir(pointgpt: Path, cfg_rel: str, exp_name: str, run_tag: str) -> Path | None:
    root = exp_root(pointgpt, cfg_rel)
    if not root.exists():
        return None
    candidates: list[Path]
    if run_tag:
        exact = root / f"{exp_name}_{run_tag}"
        candidates = [exact] if exact.exists() else []
        candidates += [p for p in root.glob(exp_name + f"*{run_tag}*") if p.is_dir()]
    else:
        candidates = [p for p in root.glob(exp_name + "*") if p.is_dir()]
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def latest_log(exp_dir: Path | None) -> Path | None:
    if exp_dir is None or not exp_dir.exists():
        return None
    logs = sorted(exp_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def parse_log(log_path: Path | None) -> list[dict[str, float | int | None]]:
    if log_path is None:
        return []
    rows: list[dict[str, float | int | None]] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "diag(" not in line or "cos_tgt" not in line:
            continue
        epoch_match = EPOCH_RE.search(line)
        row: dict[str, float | int | None] = {
            "epoch": int(epoch_match.group(1)) if epoch_match else None,
            "max_epoch": int(epoch_match.group(2)) if epoch_match else None,
            "batch": int(epoch_match.group(3)) if epoch_match else None,
            "max_batch": int(epoch_match.group(4)) if epoch_match else None,
        }
        for key, pattern in DIAG_PATTERNS.items():
            match = pattern.search(line)
            row[key] = parse_float(match.group(1) if match else None)
        if row["cos_tgt"] is not None and row["cos_prev"] is not None:
            rows.append(row)
    return rows


def classify(model_better_by_loss: float | None, copy_win: float | None) -> str:
    if model_better_by_loss is None or copy_win is None:
        return "missing_diag"
    if model_better_by_loss <= -0.002 or copy_win >= 0.55:
        return "previous_token_competitive"
    if abs(model_better_by_loss) < 0.002 or copy_win >= 0.48:
        return "near_tie"
    return "model_beats_previous_token"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--run-tag", default="", help="Optional Stage tag such as stage1_20260513_025903.")
    ap.add_argument("--final-epoch-only", action="store_true", help="Aggregate only diagnostics from the final logged epoch.")
    ap.add_argument("--tail-batches", type=int, default=64, help="Use the last N diagnostic batches per run; 0 means all.")
    ap.add_argument("--out-csv", default="posetnepa_mask_order_preflight/generated/copy_baseline_diag.csv")
    ap.add_argument("--out-md", default="posetnepa_mask_order_preflight/generated/copy_baseline_diag.md")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    manifest_path = resolve_manifest(repo, args.manifest)
    manifest = json.loads(manifest_path.read_text())
    run_tag = args.run_tag or Path(manifest["runs"][0]["pretrain_config"]).parent.name

    rows: list[dict[str, str]] = []
    for entry in manifest["runs"]:
        exp_dir = find_pretrain_dir(pointgpt, entry["pretrain_config"], entry["pretrain_exp"], run_tag)
        log_path = latest_log(exp_dir)
        samples = parse_log(log_path)
        if args.final_epoch_only:
            epochs = [sample["epoch"] for sample in samples if isinstance(sample["epoch"], int)]
            if epochs:
                final_epoch = max(epochs)
                samples = [sample for sample in samples if sample["epoch"] == final_epoch]
        used = samples if args.tail_batches <= 0 else samples[-args.tail_batches :]

        cos_tgt = mean(sample["cos_tgt"] for sample in used)
        cos_prev = mean(sample["cos_prev"] for sample in used)
        gap_logged = mean(sample["gap"] for sample in used)
        copy_win = mean(sample["copy_win"] for sample in used)
        loss_main = mean(sample["loss_main"] for sample in used)
        model_loss = None if cos_tgt is None else 1.0 - cos_tgt
        prev_loss = None if cos_prev is None else 1.0 - cos_prev
        model_better = None if model_loss is None or prev_loss is None else prev_loss - model_loss
        prev_better = None if model_better is None else -model_better
        epochs = [sample["epoch"] for sample in used if isinstance(sample["epoch"], int)]

        rows.append({
            "run_id": entry["run_id"],
            "order": entry["order"],
            "mask_ratio": str(entry["mask_ratio"]),
            "group_mode": entry["group_mode"],
            "diag_samples_total": str(len(samples)),
            "diag_samples_used": str(len(used)),
            "epoch_min_used": str(min(epochs)) if epochs else "",
            "epoch_max_used": str(max(epochs)) if epochs else "",
            "loss_main": fmt(loss_main),
            "cos_tgt": fmt(cos_tgt),
            "cos_prev": fmt(cos_prev),
            "gap_logged": fmt(gap_logged),
            "model_loss_1_minus_cos_tgt": fmt(model_loss),
            "prev_token_loss_1_minus_cos_prev": fmt(prev_loss),
            "model_better_by_loss": fmt(model_better),
            "prev_token_better_by_loss": fmt(prev_better),
            "copy_win": fmt(copy_win),
            "copy_win_minus_half": fmt(None if copy_win is None else copy_win - 0.5),
            "status": classify(model_better, copy_win),
            "log_path": str(log_path or ""),
        })

    fields = list(rows[0]) if rows else []
    out_csv = Path(args.out_csv)
    out_csv = out_csv if out_csv.is_absolute() else repo / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out_md = Path(args.out_md)
    out_md = out_md if out_md.is_absolute() else repo / out_md
    with out_md.open("w") as f:
        f.write("# Previous-Token Copy Baseline Diagnostic\n\n")
        f.write(f"Manifest: `{manifest_path}`\n\n")
        if args.final_epoch_only:
            tail_desc = "all" if args.tail_batches <= 0 else f"last {args.tail_batches}"
            f.write(f"Aggregation: {tail_desc} diagnostic batches from the final logged epoch per run.\n\n")
        else:
            f.write(f"Aggregation: last {args.tail_batches} diagnostic batches per run.\n\n")
        f.write("| run | order | mask | cos_tgt | cos_prev | model_better_by_loss | copy_win | status |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['run_id']} | {row['order']} | {row['mask_ratio']} | "
                f"{row['cos_tgt']} | {row['cos_prev']} | {row['model_better_by_loss']} | "
                f"{row['copy_win']} | {row['status']} |\n"
            )
        f.write("\n`model_better_by_loss = (1 - cos_prev) - (1 - cos_tgt)`. Negative means the previous-token baseline is better under this cosine proxy.\n")

    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
