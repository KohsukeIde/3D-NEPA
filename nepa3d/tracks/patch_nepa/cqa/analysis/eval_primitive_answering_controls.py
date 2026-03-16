from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import (
    CONTROL_MODES,
    _parse_task_filter,
    run_token_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval_primitive_answering_controls")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=-1)
    p.add_argument("--n_qry", type=int, default=-1)
    p.add_argument("--max_samples_per_task", type=int, default=256)
    p.add_argument("--split_override", type=str, default="")
    p.add_argument("--task_filter", type=str, default="")
    p.add_argument("--eval_sample_mode", type=str, default="head", choices=["head", "random"])
    p.add_argument(
        "--controls",
        type=str,
        default="correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query",
        help="Comma-separated subset of controls.",
    )
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    requested = [x.strip() for x in str(args.controls).split(",") if x.strip()]
    for control in requested:
        if control not in CONTROL_MODES:
            raise KeyError(f"unknown control={control}")

    common = dict(
        ckpt_path=str(args.ckpt),
        mix_config_path=str(args.mix_config_path),
        device=str(args.device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        max_samples_per_task=int(args.max_samples_per_task),
        split_override=(str(args.split_override).strip() or None),
        task_filter=_parse_task_filter(str(args.task_filter)),
        eval_sample_mode=str(args.eval_sample_mode),
    )
    by_control = {control: run_token_eval(control=control, **common) for control in requested}
    baseline = by_control.get("correct", None)

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
                    "delta_answer_entropy": float(summary["overall"]["answer_entropy"]) - float(
                        baseline["overall"]["answer_entropy"]
                    ),
                },
                "per_task": task_delta,
            }

    payload = {
        "controls": requested,
        "by_control": by_control,
        "deltas_vs_correct": deltas,
    }
    print(json.dumps(payload, indent=2))
    if str(args.output_json).strip():
        out_path = Path(str(args.output_json))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
