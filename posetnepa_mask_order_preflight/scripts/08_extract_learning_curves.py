#!/usr/bin/env python3
"""Extract pretrain and fine-tune learning curves for mask/order pre-flight."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


FLOAT = r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?"
PRETRAIN_EPOCH_RE = re.compile(
    rf"\[Training\] EPOCH: (\d+).*Losses = \['({FLOAT})'\]"
)
FINETUNE_TRAIN_RE = re.compile(
    rf"\[Training\] EPOCH: (\d+).*Losses = \['({FLOAT})', '({FLOAT})', '({FLOAT})'\]"
)
FINETUNE_VAL_RE = re.compile(rf"\[Validation\] EPOCH: (\d+)\s+acc = ({FLOAT})")


def exp_root(pointgpt: Path, cfg_rel: str) -> Path:
    cfg = Path(cfg_rel)
    return pointgpt / "experiments" / cfg.stem / cfg.parent.name


def latest_log(exp_dir: Path | None) -> Path | None:
    if exp_dir is None or not exp_dir.exists():
        return None
    logs = sorted(exp_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def find_pretrain_dir(pointgpt: Path, cfg_rel: str, exp_name: str, run_tag: str) -> Path | None:
    name = f"{exp_name}_{run_tag}" if run_tag else exp_name
    path = exp_root(pointgpt, cfg_rel) / name
    if path.exists():
        return path
    cands = sorted(exp_root(pointgpt, cfg_rel).glob(exp_name + "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def find_finetune_dir(pointgpt: Path, cfg_rel: str, exp_prefix: str, run_tag: str) -> Path | None:
    root = exp_root(pointgpt, cfg_rel)
    if not root.exists():
        return None
    pat = exp_prefix + (f"*{run_tag}*" if run_tag else "*")
    cands = sorted(root.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def parse_pretrain(log: Path | None) -> list[tuple[int, float]]:
    if log is None:
        return []
    by_epoch: dict[int, float] = {}
    for line in log.read_text(errors="ignore").splitlines():
        match = PRETRAIN_EPOCH_RE.search(line)
        if match:
            by_epoch[int(match.group(1))] = float(match.group(2))
    return sorted(by_epoch.items())


def parse_finetune(log: Path | None) -> tuple[list[tuple[int, float, float]], list[tuple[int, float]]]:
    if log is None:
        return [], []
    train: dict[int, tuple[float, float]] = {}
    val: dict[int, float] = {}
    for line in log.read_text(errors="ignore").splitlines():
        match = FINETUNE_TRAIN_RE.search(line)
        if match:
            train[int(match.group(1))] = (float(match.group(2)), float(match.group(4)))
        match = FINETUNE_VAL_RE.search(line)
        if match:
            val[int(match.group(1))] = float(match.group(2))
    train_rows = [(epoch, loss, acc) for epoch, (loss, acc) in sorted(train.items())]
    val_rows = sorted(val.items())
    return train_rows, val_rows


def mean_last(values: list[float], n: int = 5) -> float | None:
    if not values:
        return None
    return statistics.mean(values[-n:])


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--run-tag", default="")
    ap.add_argument("--splits", default="hardest")
    ap.add_argument("--out-dir", default="posetnepa_mask_order_preflight/generated")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    manifest = json.loads((repo / args.manifest).read_text())
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    pre_summary = []
    pre_points = []
    ft_summary = []
    ft_points = []

    for entry in manifest["runs"]:
        run_id = entry["run_id"]
        pre_dir = find_pretrain_dir(pointgpt, entry["pretrain_config"], entry["pretrain_exp"], args.run_tag)
        pre_log = latest_log(pre_dir)
        pre_curve = parse_pretrain(pre_log)
        for epoch, loss in pre_curve:
            pre_points.append({
                "run_id": run_id,
                "order": entry["order"],
                "mask_ratio": entry["mask_ratio"],
                "epoch": epoch,
                "loss": f"{loss:.6f}",
            })
        if pre_curve:
            losses = [loss for _, loss in pre_curve]
            pre_summary.append({
                "run_id": run_id,
                "order": entry["order"],
                "mask_ratio": entry["mask_ratio"],
                "epochs_seen": len(pre_curve),
                "first_epoch": pre_curve[0][0],
                "last_epoch": pre_curve[-1][0],
                "first_loss": fmt(losses[0]),
                "last_loss": fmt(losses[-1]),
                "min_loss": fmt(min(losses)),
                "last5_loss_mean": fmt(mean_last(losses)),
                "drop_first_to_last": fmt(losses[0] - losses[-1]),
                "log_path": str(pre_log or ""),
            })

        for split in splits:
            key = {"obj_bg": "objbg", "obj_only": "objonly", "pb_t50_rs": "hardest"}.get(split, split)
            if key not in entry["finetune_configs"]:
                continue
            exp_prefix = f"{entry['finetune_exp_prefix']}_{key}_from_"
            ft_dir = find_finetune_dir(pointgpt, entry["finetune_configs"][key], exp_prefix, args.run_tag)
            ft_log = latest_log(ft_dir)
            train_curve, val_curve = parse_finetune(ft_log)
            train_by_epoch = {epoch: (loss, acc) for epoch, loss, acc in train_curve}
            for epoch, val_acc in val_curve:
                loss, train_acc = train_by_epoch.get(epoch, ("", ""))
                ft_points.append({
                    "run_id": run_id,
                    "order": entry["order"],
                    "mask_ratio": entry["mask_ratio"],
                    "split": key,
                    "epoch": epoch,
                    "train_loss": "" if loss == "" else f"{loss:.6f}",
                    "train_acc": "" if train_acc == "" else f"{train_acc:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                })
            if val_curve:
                vals = [acc for _, acc in val_curve]
                best_epoch, best_acc = max(val_curve, key=lambda x: x[1])
                last_epoch, last_acc = val_curve[-1]
                tr_loss = tr_acc = None
                if train_curve:
                    _, tr_loss, tr_acc = train_curve[-1]
                status = "running" if last_epoch < 50 else "complete"
                best_minus_last = best_acc - last_acc
                if status == "complete" and best_epoch >= last_epoch - 5 and best_minus_last <= 0.5:
                    status = "late_or_still_improving"
                elif status == "complete" and best_minus_last <= 1.0:
                    status = "plateau"
                elif status == "complete":
                    status = "peaked_early_or_unstable"
                ft_summary.append({
                    "run_id": run_id,
                    "order": entry["order"],
                    "mask_ratio": entry["mask_ratio"],
                    "split": key,
                    "val_epochs_seen": len(val_curve),
                    "best_epoch": best_epoch,
                    "best_val_acc": fmt(best_acc),
                    "last_epoch": last_epoch,
                    "last_val_acc": fmt(last_acc),
                    "last5_val_acc_mean": fmt(mean_last(vals)),
                    "best_minus_last": fmt(best_minus_last),
                    "last_train_loss": fmt(tr_loss),
                    "last_train_acc": fmt(tr_acc),
                    "curve_status": status,
                    "log_path": str(ft_log or ""),
                })
            else:
                ft_summary.append({
                    "run_id": run_id,
                    "order": entry["order"],
                    "mask_ratio": entry["mask_ratio"],
                    "split": key,
                    "val_epochs_seen": 0,
                    "best_epoch": "",
                    "best_val_acc": "",
                    "last_epoch": "",
                    "last_val_acc": "",
                    "last5_val_acc_mean": "",
                    "best_minus_last": "",
                    "last_train_loss": "",
                    "last_train_acc": "",
                    "curve_status": "missing_or_not_validated_yet",
                    "log_path": str(ft_log or ""),
                })

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("")
            return
        fields = list(rows[0])
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "pretrain_curve_points.csv", pre_points)
    write_csv(out_dir / "pretrain_curve_summary.csv", pre_summary)
    write_csv(out_dir / "finetune_curve_points.csv", ft_points)
    write_csv(out_dir / "finetune_curve_summary.csv", ft_summary)

    md = out_dir / "learning_curve_summary.md"
    with md.open("w") as f:
        f.write("# Learning Curve Summary\n\n")
        f.write("## Pretrain\n\n")
        f.write("| run | mask | epochs | first loss | last loss | min loss | last5 mean | drop |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in pre_summary:
            f.write(
                f"| {row['run_id']} | {row['mask_ratio']} | {row['epochs_seen']} | "
                f"{row['first_loss']} | {row['last_loss']} | {row['min_loss']} | "
                f"{row['last5_loss_mean']} | {row['drop_first_to_last']} |\n"
            )
        f.write("\n## Fine-tune\n\n")
        f.write("| run | mask | split | vals | best | last | last5 mean | best-last | status |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---|\n")
        for row in ft_summary:
            best = f"{row['best_val_acc']}@{row['best_epoch']}" if row["best_val_acc"] else ""
            last = f"{row['last_val_acc']}@{row['last_epoch']}" if row["last_val_acc"] else ""
            f.write(
                f"| {row['run_id']} | {row['mask_ratio']} | {row['split']} | "
                f"{row['val_epochs_seen']} | {best} | {last} | {row['last5_val_acc_mean']} | "
                f"{row['best_minus_last']} | {row['curve_status']} |\n"
            )
    print(f"[done] wrote {md}")


if __name__ == "__main__":
    main()
