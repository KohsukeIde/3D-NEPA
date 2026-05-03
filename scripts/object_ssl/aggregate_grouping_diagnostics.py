#!/usr/bin/env python
"""Aggregate eval-time grouping diagnostics for Point-MAE / PCP-MAE."""

from __future__ import annotations

import argparse
from pathlib import Path

from object_ssl_common import git_commit, markdown_table, read_json, repo_root_from_script, write_csv, write_json, write_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Aggregate Point-MAE / PCP-MAE eval-time grouping diagnostics")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--result-dir", required=True)
    return p.parse_args()


def rows_from_payload(path: Path) -> list[dict]:
    payload = read_json(path)
    meta = payload["metadata"]
    out = []
    for condition in payload["conditions"]:
        row = {
            "model": meta["model"],
            "task": meta["task"],
            "split": meta["split"],
            "selection_protocol": meta["selection_protocol"],
            "grouping_mode": meta.get("grouping_mode", "fps_knn"),
            "condition": condition["condition"],
            "checkpoint_path": meta["checkpoint_path"],
            "n_samples": condition.get("n_samples", condition.get("n_shapes", meta.get("n_samples", ""))),
            "seed": meta["seed"],
            "script": meta["script"],
            "git_commit": meta["git_commit"],
            "notes": meta.get("notes", ""),
        }
        if meta["task"] == "scanobjectnn":
            row.update(
                {
                    "metric_name": "Top-1 (%)",
                    "score": condition.get("top1"),
                    "top2_hit": condition.get("top2_hit"),
                    "top5_hit": condition.get("top5_hit"),
                    "damage_pp": condition.get("damage_pp"),
                    "strong_stress_score": condition.get("top1") if condition["condition"] == "xyz_zero" else "",
                }
            )
        elif meta["task"] == "shapenetpart":
            row.update(
                {
                    "metric_name": "Instance mIoU (%)",
                    "score": condition.get("instance_avg_miou"),
                    "top2_hit": condition.get("point_top2_hit"),
                    "top5_hit": condition.get("point_top5_hit"),
                    "damage_pp": condition.get("damage_pp"),
                    "strong_stress_score": (
                        condition.get("instance_avg_miou")
                        if condition["condition"] in {"largest_part_removed", "xyz_zero"}
                        else ""
                    ),
                }
            )
        else:
            continue
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        rows.extend(rows_from_payload(path))

    fields = [
        "model",
        "task",
        "split",
        "selection_protocol",
        "grouping_mode",
        "condition",
        "metric_name",
        "score",
        "damage_pp",
        "top2_hit",
        "top5_hit",
        "n_samples",
        "checkpoint_path",
        "seed",
        "git_commit",
        "notes",
    ]
    summary_fields = [
        "model",
        "task",
        "split",
        "selection_protocol",
        "grouping_mode",
        "condition",
        "metric_name",
        "score",
        "damage_pp",
    ]
    write_json(result_dir / "object_ssl_pointmae_pcpmae_grouping.json", rows)
    write_csv(result_dir / "object_ssl_pointmae_pcpmae_grouping.csv", rows, fields)
    lines = [
        "# Point-MAE / PCP-MAE Eval-Time Grouping Diagnostics",
        "",
        "Checkpoint and readout are fixed. These rows perturb grouping only at inference time; no retraining is used.",
        "",
        f"- git commit: `{git_commit(root)}`",
        f"- raw dir: `{input_dir.resolve()}`",
        "",
        markdown_table(rows, summary_fields),
    ]
    write_text(result_dir / "object_ssl_pointmae_pcpmae_grouping.md", "\n".join(lines))
    print(f"[done] wrote grouping diagnostics from {len(rows)} rows")


if __name__ == "__main__":
    main()
