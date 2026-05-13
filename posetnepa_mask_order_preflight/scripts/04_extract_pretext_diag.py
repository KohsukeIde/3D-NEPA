#!/usr/bin/env python3
"""Extract NEPA pretext diagnostics from PointGPT logs.

This is intentionally permissive. It scans likely experiment directories for lines
containing loss/cos_tgt/cos_prev/gap/copy_win. The primary reported values are
means over the final logged epoch with diagnostics, so one noisy last batch does
not stand in for the run. If no values are found, it still outputs rows from the
manifest for manual filling.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
METRICS = ["loss", "cos_tgt", "cos_prev", "gap", "copy_win"]
PATTERNS = {
    "loss": re.compile(rf"(?<![A-Za-z0-9_])(?:loss_main|loss_total|loss)\s*[:=]\s*({FLOAT})", re.IGNORECASE),
    "cos_tgt": re.compile(rf"(?<![A-Za-z0-9_])(?:cos_tgt|cos target|cos_target)\s*[:=]\s*({FLOAT})", re.IGNORECASE),
    "cos_prev": re.compile(rf"(?<![A-Za-z0-9_])(?:cos_prev|cos previous|cos_previous)\s*[:=]\s*({FLOAT})", re.IGNORECASE),
    "gap": re.compile(rf"(?<![A-Za-z0-9_])gap\s*[:=]\s*({FLOAT})", re.IGNORECASE),
    "copy_win": re.compile(rf"(?<![A-Za-z0-9_])(?:copy_win|copy win|copywin)\s*[:=]\s*({FLOAT})", re.IGNORECASE),
}
EPOCH_BATCH_RE = re.compile(r"\[Epoch\s+(\d+)(?:/\d+)?\]\[Batch\s+(\d+)(?:/\d+)?\]", re.IGNORECASE)
EPOCH_RE = re.compile(r"(?<![A-Za-z0-9_])epoch\s*[:=]\s*(\d+)", re.IGNORECASE)
BATCH_RE = re.compile(r"(?<![A-Za-z0-9_])batch\s*[:=]\s*(\d+)", re.IGNORECASE)


def latest_matching_exp(pointgpt: Path, cfg_rel: str, base_exp: str, run_tag: str = "") -> Path | None:
    cfg_stem = Path(cfg_rel).stem
    cfg_parent = Path(cfg_rel).parent.name
    root = pointgpt / "experiments" / cfg_stem / cfg_parent
    if not root.exists():
        return None
    if run_tag:
        exact = root / f"{base_exp}_{run_tag}"
        cands = [exact] if exact.exists() else []
        cands += [p for p in root.glob(base_exp + f"*{run_tag}*") if p.is_dir()]
    else:
        cands = [p for p in root.glob(base_exp + "_*") if p.is_dir()]
    cands = sorted({p for p in cands if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def empty_diag() -> dict:
    vals = {k: None for k in METRICS}
    vals.update({
        "n_diag": 0,
        "epoch": None,
        "n_diag_total": 0,
        "batch_last": None,
    })
    for key in METRICS:
        vals[f"{key}_mean"] = None
        vals[f"{key}_last"] = None
    return vals


def candidate_log_files(exp_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pat in ["*.log", "log.txt", "*.out", "stdout*", "stderr*"]:
        files += list(exp_dir.rglob(pat))
    # Also scan small text-like files in case logger uses another name.
    for f in exp_dir.rglob("*"):
        if f.is_file() and f.suffix in {".txt", ".log", ".out"} and f not in files:
            files.append(f)
    unique = {f.resolve(): f for f in files}
    return sorted(unique.values(), key=lambda p: (p.stat().st_mtime, str(p)))


def parse_epoch_batch(line: str) -> tuple[int | None, int | None]:
    match = EPOCH_BATCH_RE.search(line)
    if match:
        return int(match.group(1)), int(match.group(2))
    epoch = None
    batch = None
    match = EPOCH_RE.search(line)
    if match:
        epoch = int(match.group(1))
    match = BATCH_RE.search(line)
    if match:
        batch = int(match.group(1))
    return epoch, batch


def mean_metric(records: list[dict], key: str) -> float | None:
    values = [r["values"][key] for r in records if key in r["values"]]
    return statistics.mean(values) if values else None


def scan_logs(exp_dir: Path | None) -> dict:
    vals = empty_diag()
    if exp_dir is None or not exp_dir.exists():
        return vals
    records: list[dict] = []
    for f in candidate_log_files(exp_dir):
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for line in text.splitlines():
            found = {}
            for key, pat in PATTERNS.items():
                m = pat.search(line)
                if m:
                    try:
                        found[key] = float(m.group(1))
                    except Exception:
                        pass
            if found:
                epoch, batch = parse_epoch_batch(line)
                records.append({
                    "epoch": epoch,
                    "batch": batch,
                    "seq": len(records),
                    "values": found,
                })
    if not records:
        return vals

    vals["n_diag_total"] = len(records)
    epoch_records = [r for r in records if r["epoch"] is not None]
    if epoch_records:
        final_epoch = max(r["epoch"] for r in epoch_records)
        selected = [r for r in epoch_records if r["epoch"] == final_epoch]
        vals["epoch"] = final_epoch
    else:
        selected = records

    vals["n_diag"] = len(selected)
    for key in METRICS:
        value = mean_metric(selected, key)
        vals[key] = value
        vals[f"{key}_mean"] = value

    def last_key(record: dict) -> tuple[int, int]:
        batch = record["batch"] if record["batch"] is not None else -1
        return batch, record["seq"]

    last = max(selected, key=last_key)
    vals["batch_last"] = last["batch"]
    for key in METRICS:
        vals[f"{key}_last"] = last["values"].get(key)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--run-tag", default="", help="Optional Stage tag, e.g. stage1_20260513_025903.")
    ap.add_argument("--out-csv", default="posetnepa_mask_order_preflight/generated/pretext_diag.csv")
    ap.add_argument("--out-md", default="posetnepa_mask_order_preflight/generated/pretext_diag.md")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    m = json.loads((repo / args.manifest).read_text())
    run_tag = args.run_tag or Path(m["runs"][0]["pretrain_config"]).parent.name
    rows = []
    for e in m["runs"]:
        exp_dir = latest_matching_exp(pointgpt, e["pretrain_config"], e["pretrain_exp"], run_tag)
        vals = scan_logs(exp_dir)
        row = {
            "run_id": e["run_id"],
            "order": e["order"],
            "mask_ratio": e["mask_ratio"],
            "group_mode": e["group_mode"],
            "exp_dir": str(exp_dir) if exp_dir else "",
            **vals,
        }
        rows.append(row)

    out_csv = repo / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id", "order", "mask_ratio", "group_mode",
        "loss", "cos_tgt", "cos_prev", "gap", "copy_win", "exp_dir",
        "n_diag", "epoch",
        "loss_mean", "cos_tgt_mean", "cos_prev_mean", "gap_mean", "copy_win_mean",
        "batch_last",
        "loss_last", "cos_tgt_last", "cos_prev_last", "gap_last", "copy_win_last",
        "n_diag_total",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_md = repo / args.out_md
    def fmt(x):
        if x is None or x == "":
            return ""
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)
    with out_md.open("w") as f:
        f.write("# Pretext diagnostics\n\n")
        f.write("Values are means over the final logged epoch with diagnostics. The CSV also includes last-batch diagnostic values.\n\n")
        f.write("| run | order | mask | epoch | n_diag | loss_mean | cos_tgt_mean | cos_prev_mean | gap_mean | copy_win_mean |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[k]) for k in [
                "run_id", "order", "mask_ratio", "epoch", "n_diag",
                "loss_mean", "cos_tgt_mean", "cos_prev_mean", "gap_mean", "copy_win_mean",
            ]) + " |\n")
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
