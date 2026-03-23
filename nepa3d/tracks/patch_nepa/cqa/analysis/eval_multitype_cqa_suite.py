from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import run_token_eval

DEFAULT_TASKS = "udf_distance,mesh_normal"
DEFAULT_CONTROLS = (
    "correct",
    "no_context",
    "wrong_shape_same_synset",
    "wrong_shape_other_synset",
    "shuffled_query",
)


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _run_controls(
    *,
    ckpt: str,
    mix_config_path: str,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    n_ctx: int,
    n_qry: int,
    max_samples_per_task: int,
    split_override: str | None,
    task_filter: set[str],
    eval_sample_mode: str,
    controls: Iterable[str],
    query_order: str,
) -> dict[str, Any]:
    by_control = {}
    for control in controls:
        by_control[str(control)] = run_token_eval(
            ckpt_path=str(ckpt),
            mix_config_path=str(mix_config_path),
            device=str(device),
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            seed=int(seed),
            n_ctx=int(n_ctx),
            n_qry=int(n_qry),
            max_samples_per_task=int(max_samples_per_task),
            split_override=split_override,
            task_filter=set(task_filter),
            control=str(control),
            eval_sample_mode=str(eval_sample_mode),
            query_order=str(query_order),
        )
    baseline = by_control.get("correct")
    deltas = {}
    if baseline is not None:
        base_by_task = {r["task_name"]: r for r in baseline["results"]}
        for control, summary in by_control.items():
            if control == "correct":
                continue
            task_delta = {}
            for row in summary["results"]:
                base = base_by_task.get(row["task_name"])
                if base is None:
                    continue
                task_delta[row["task_name"]] = {
                    "delta_ce": float(row["ce"]) - float(base["ce"]),
                    "delta_token_acc": float(row["token_acc"]) - float(base["token_acc"]),
                    "delta_answer_entropy": float(row["answer_entropy"]) - float(base["answer_entropy"]),
                }
            deltas[control] = {
                "overall": {
                    "delta_ce": float(summary["overall"]["ce"]) - float(baseline["overall"]["ce"]),
                    "delta_token_acc": float(summary["overall"]["token_acc"]) - float(baseline["overall"]["token_acc"]),
                    "delta_answer_entropy": float(summary["overall"]["answer_entropy"]) - float(baseline["overall"]["answer_entropy"]),
                },
                "per_task": task_delta,
            }
    return {"by_control": by_control, "deltas_vs_correct": deltas}


def _flatten(tag: str, payload: dict[str, Any], controls: list[str]) -> list[dict[str, Any]]:
    base = payload["by_control"]["correct"]
    rows: list[dict[str, Any]] = []
    for row in base["results"]:
        task = str(row["task_name"])
        item = {
            "suite": str(tag),
            "task": task,
            "query_type": row["query_type_name"],
            "context_source": row["context_source"],
            "n_samples": int(row["n_samples"]),
            "token_acc": float(row["token_acc"]),
            "majority_acc": float(row["majority_baseline_acc"]),
            "ce": float(row["ce"]),
            "answer_entropy": float(row["answer_entropy"]),
            "pred_top1_share": float(row["pred_top1_share"]),
            "pred_unique_codes": int(row["pred_unique_codes"]),
        }
        for c in controls:
            if c == "correct":
                continue
            delta = payload["deltas_vs_correct"].get(c, {}).get("per_task", {}).get(task)
            if delta is not None:
                item[f"delta_ce({c})"] = float(delta["delta_ce"])
                item[f"delta_acc({c})"] = float(delta["delta_token_acc"])
            else:
                item[f"delta_ce({c})"] = float("nan")
                item[f"delta_acc({c})"] = float("nan")
        rows.append(item)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval_multitype_cqa_suite")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--same_mix_config", type=str, required=True)
    p.add_argument("--offdiag_mix_config", type=str, default="")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=-1)
    p.add_argument("--n_qry", type=int, default=-1)
    p.add_argument("--max_samples_per_task", type=int, default=256)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    p.add_argument("--controls", type=str, default=",".join(DEFAULT_CONTROLS))
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--query_order", type=str, default="sampled")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_csv", type=str, default="")
    p.add_argument("--output_md", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    task_filter = set(_parse_csv(args.tasks))
    controls = _parse_csv(args.controls)
    split_override = str(args.split_override).strip() or None

    payload: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "tasks": sorted(task_filter),
        "controls": controls,
        "query_order": str(args.query_order),
        "same": _run_controls(
            ckpt=str(args.ckpt),
            mix_config_path=str(args.same_mix_config),
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            n_ctx=int(args.n_ctx),
            n_qry=int(args.n_qry),
            max_samples_per_task=int(args.max_samples_per_task),
            split_override=split_override,
            task_filter=task_filter,
            eval_sample_mode=str(args.eval_sample_mode),
            controls=controls,
            query_order=str(args.query_order),
        ),
    }
    rows = _flatten("same", payload["same"], controls)
    if str(args.offdiag_mix_config).strip():
        payload["offdiag"] = _run_controls(
            ckpt=str(args.ckpt),
            mix_config_path=str(args.offdiag_mix_config),
            device=str(args.device),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            seed=int(args.seed),
            n_ctx=int(args.n_ctx),
            n_qry=int(args.n_qry),
            max_samples_per_task=int(args.max_samples_per_task),
            split_override=split_override,
            task_filter=task_filter,
            eval_sample_mode=str(args.eval_sample_mode),
            controls=controls,
            query_order=str(args.query_order),
        )
        rows.extend(_flatten("offdiag", payload["offdiag"], controls))

    out_json = Path(str(args.output_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    if str(args.output_csv).strip():
        out_csv = Path(str(args.output_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(out_csv, rows)
    if str(args.output_md).strip():
        out_md = Path(str(args.output_md))
        out_md.parent.mkdir(parents=True, exist_ok=True)
        _write_md(out_md, rows)


if __name__ == "__main__":
    main()
