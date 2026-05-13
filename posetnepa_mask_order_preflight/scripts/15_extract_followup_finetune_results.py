#!/usr/bin/env python3
"""Extract fine-tune results for mismatch and scratch/early follow-up manifests."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
VAL_RE = re.compile(rf"\[Validation\].*acc\s*=\s*({FLOAT})")


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


def find_exp_dir(pointgpt: Path, cfg_rel: str, exp_name: str) -> Path | None:
    root = exp_root(pointgpt, cfg_rel)
    exact = root / exp_name
    if exact.exists():
        return exact
    if not root.exists():
        return None
    candidates = sorted(root.glob(exp_name + "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def rows_from_manifest(manifest: dict) -> list[dict]:
    if "rows" in manifest:
        rows = []
        for row in manifest["rows"]:
            rows.append({
                "job_id": row["row_id"],
                "init_kind": "stage1_maskoff",
                "pretrain_order": row["pretrain_order"],
                "finetune_order": row["finetune_order"],
                "mask_ratio": row["pretrain_mask_ratio"],
                "source_epoch": "",
                "freeze_backbone": False,
                "split": row["split"],
                "finetune_config": row["finetune_config"],
                "exp_name": row["finetune_exp"],
                "ckpt_path": row["ckpt_path"],
            })
        return rows
    if "finetune_jobs" in manifest:
        rows = []
        for job in manifest["finetune_jobs"]:
            rows.append({
                "job_id": job["job_id"],
                "init_kind": job["init_kind"],
                "pretrain_order": job["order"],
                "finetune_order": job["finetune_order"],
                "mask_ratio": job["mask_ratio"],
                "source_epoch": job["source_epoch"],
                "freeze_backbone": job.get("freeze_backbone", False),
                "split": job["split"],
                "finetune_config": job["finetune_config"],
                "exp_name": job["exp_name"],
                "ckpt_path": job["ckpt_path"],
            })
        return rows
    raise SystemExit("[error] unsupported follow-up manifest shape")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    manifest_path = Path(args.manifest)
    manifest_path = manifest_path if manifest_path.is_absolute() else repo / manifest_path
    manifest = json.loads(manifest_path.read_text())

    output_rows: list[dict[str, str]] = []
    for job in rows_from_manifest(manifest):
        exp_dir = find_exp_dir(pointgpt, job["finetune_config"], job["exp_name"])
        acc = None
        status = "missing"
        if (
            job["init_kind"] == "stage1_maskoff"
            and str(job["pretrain_order"]) == str(job["finetune_order"])
            and exp_dir is None
        ):
            status = "skipped_diagonal"
        if exp_dir is not None:
            acc = acc_from_checkpoint(exp_dir / "ckpt-best.pth")
            if acc is None:
                acc = acc_from_checkpoint(exp_dir / "ckpt-last.pth")
            if acc is None:
                acc = best_acc_from_logs(exp_dir)
            status = "ok" if acc is not None else "no_metric"
        output_rows.append({
            "job_id": str(job["job_id"]),
            "init_kind": str(job["init_kind"]),
            "pretrain_order": str(job["pretrain_order"]),
            "finetune_order": str(job["finetune_order"]),
            "mask_ratio": str(job["mask_ratio"]),
            "source_epoch": str(job["source_epoch"]),
            "freeze_backbone": "1" if job.get("freeze_backbone", False) else "0",
            "split": str(job["split"]),
            "best_acc": "" if acc is None else f"{acc:.6f}",
            "status": status,
            "exp_dir": str(exp_dir or ""),
            "ckpt_path": str(job["ckpt_path"]),
        })

    default_csv = manifest_path.with_name(manifest_path.stem.replace("_manifest", "") + "_results.csv")
    default_md = manifest_path.with_name(manifest_path.stem.replace("_manifest", "") + "_results.md")
    out_csv = Path(args.out_csv) if args.out_csv else default_csv
    out_md = Path(args.out_md) if args.out_md else default_md
    out_csv = out_csv if out_csv.is_absolute() else repo / out_csv
    out_md = out_md if out_md.is_absolute() else repo / out_md

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(output_rows[0]) if output_rows else []
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    with out_md.open("w") as f:
        f.write("# Follow-Up Fine-Tune Results\n\n")
        f.write("| job | init | pre order | ft order | mask | epoch | frozen | split | best acc | status |\n")
        f.write("|---|---|---|---|---:|---:|---:|---|---:|---|\n")
        for row in output_rows:
            f.write(
                f"| {row['job_id']} | {row['init_kind']} | {row['pretrain_order']} | "
                f"{row['finetune_order']} | {row['mask_ratio']} | {row['source_epoch']} | "
                f"{row['freeze_backbone']} | {row['split']} | {row['best_acc']} | {row['status']} |\n"
            )
    print(f"[done] wrote {out_csv}")
    print(f"[done] wrote {out_md}")


if __name__ == "__main__":
    main()
