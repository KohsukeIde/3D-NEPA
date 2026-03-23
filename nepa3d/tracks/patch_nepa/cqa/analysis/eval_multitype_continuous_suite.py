from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nepa3d.data.mixed_pretrain import load_mix_config
from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import apply_control
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import (
    ASK_DISTANCE,
    ASK_NORMAL,
    encode_answers_from_fields,
    quantize_normals_unsigned_to_vocab,
)
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa_continuous import (
    V2PrimitiveCQAContinuousDataset,
    cqa_continuous_collate_fn,
)
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering_distnorm_continuous import (
    load_distnorm_continuous_model,
)

DEFAULT_TASKS = "udf_distance,mesh_normal"
DEFAULT_CONTROLS = (
    "correct",
    "no_context",
    "wrong_shape_same_synset",
    "wrong_shape_other_synset",
    "shuffled_query",
)


@dataclass(frozen=True)
class EvalDatasetSpec:
    name: str
    split: str
    cache_root: str
    context_source: str
    eval_sample_mode: str
    dataset: V2PrimitiveCQAContinuousDataset


def _parse_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _sample_eval_paths(paths: List[str], *, max_samples: int, seed: int, mode: str) -> List[str]:
    if int(max_samples) <= 0 or len(paths) <= int(max_samples):
        return list(paths)
    mode = str(mode)
    if mode == "head":
        return list(paths[: int(max_samples)])
    if mode == "random":
        rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        take = rng.choice(len(paths), size=int(max_samples), replace=False)
        return [paths[int(i)] for i in take.tolist()]
    raise KeyError(f"unknown eval_sample_mode={mode}")


def _build_eval_datasets(
    mix_config_path: str,
    *,
    seed: int,
    n_ctx: int,
    n_qry: int,
    max_samples_per_task: int,
    split_override: str | None,
    task_filter: set[str],
    eval_sample_mode: str,
    query_order: str,
) -> List[EvalDatasetSpec]:
    specs, _cfg = load_mix_config(mix_config_path)
    out: List[EvalDatasetSpec] = []
    for i, s in enumerate(specs):
        task_name = str(s.extra.get("task_name", s.name))
        if task_filter and task_name not in task_filter:
            continue
        split = str(split_override).strip() if split_override else str(s.split)
        paths = list_npz(s.cache_root, split)
        if len(paths) == 0:
            continue
        picked = _sample_eval_paths(
            paths,
            max_samples=int(max_samples_per_task),
            seed=int(seed) + 97 * i,
            mode=str(eval_sample_mode),
        )
        ds = V2PrimitiveCQAContinuousDataset(
            picked,
            task_name=task_name,
            context_source=str(s.extra.get("context_source", "surf")),
            n_ctx=int(s.extra.get("n_ctx", n_ctx)),
            n_qry=int(s.extra.get("n_qry", n_qry)),
            seed=int(seed),
            mode="eval",
            query_src_filter=s.extra.get("query_src_filter", None),
            query_dist_min=s.extra.get("query_dist_min", None),
            query_dist_max=s.extra.get("query_dist_max", None),
            query_order=query_order,
        )
        out.append(
            EvalDatasetSpec(
                name=task_name,
                split=split,
                cache_root=str(s.cache_root),
                context_source=str(s.extra.get("context_source", "surf")),
                eval_sample_mode=str(eval_sample_mode),
                dataset=ds,
            )
        )
    return out


def _distance_metrics(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    pred_code = encode_answers_from_fields(ASK_DISTANCE, {"distance": y_pred})
    tgt_code = encode_answers_from_fields(ASK_DISTANCE, {"distance": y_true})
    pred_code = np.asarray(pred_code, dtype=np.int64).reshape(-1)
    tgt_code = np.asarray(tgt_code, dtype=np.int64).reshape(-1)
    tgt_hist = np.bincount(tgt_code, minlength=640)
    pred_hist = np.bincount(pred_code, minlength=640)
    gt = y_true <= float(tau)
    pr = y_pred <= float(tau)
    inter = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum()) + 1e-9
    return {
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        f"iou@{float(tau):g}": float(inter / union),
        "code_acc_proxy": float(np.mean(pred_code == tgt_code)),
        "majority_code_acc": float(tgt_hist.max()) / float(max(1, tgt_code.size)),
        "pred_top1_share": float(pred_hist.max()) / float(max(1, pred_code.size)),
        "pred_unique_codes": int(np.sum(pred_hist > 0)),
        "n_tokens": int(y_true.size),
    }


def _normal_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, unsigned: bool = False) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1, 3)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1, 3)
    y_true = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-8)
    y_pred = y_pred / (np.linalg.norm(y_pred, axis=1, keepdims=True) + 1e-8)
    cos = np.sum(y_true * y_pred, axis=1)
    if bool(unsigned):
        cos = np.abs(cos)
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    if bool(unsigned):
        pred_code = quantize_normals_unsigned_to_vocab(y_pred)
        tgt_code = quantize_normals_unsigned_to_vocab(y_true)
    else:
        pred_code = encode_answers_from_fields(ASK_NORMAL, {"normal": y_pred})
        tgt_code = encode_answers_from_fields(ASK_NORMAL, {"normal": y_true})
    pred_code = np.asarray(pred_code, dtype=np.int64).reshape(-1)
    tgt_code = np.asarray(tgt_code, dtype=np.int64).reshape(-1)
    tgt_hist = np.bincount(tgt_code, minlength=640)
    pred_hist = np.bincount(pred_code, minlength=640)
    return {
        "mean_cos": float(np.mean(cos)),
        "angle_deg": float(np.mean(angle)),
        "code_acc_proxy": float(np.mean(pred_code == tgt_code)),
        "majority_code_acc": float(tgt_hist.max()) / float(max(1, tgt_code.size)),
        "pred_top1_share": float(pred_hist.max()) / float(max(1, pred_code.size)),
        "pred_unique_codes": int(np.sum(pred_hist > 0)),
        "n_tokens": int(y_true.shape[0]),
    }


def _evaluate_dataset(
    model,
    spec: EvalDatasetSpec,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    control: str,
    tau: float,
) -> Dict[str, Any]:
    loader = DataLoader(
        spec.dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=cqa_continuous_collate_fn,
    )
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_effective: List[np.ndarray] = []
    for batch in loader:
        batch = apply_control(batch, str(control))
        ctx_xyz = batch["ctx_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
        qry_xyz = batch["qry_xyz"].to(device=device, dtype=torch.float32, non_blocking=True)
        qry_type = batch["qry_type"].to(device=device, dtype=torch.long, non_blocking=True)
        target_vec = batch["target_vec"].cpu().numpy().astype(np.float32, copy=False)
        effective = batch["control_effective_mask"].cpu().numpy().astype(np.bool_, copy=False)
        with torch.no_grad():
            pred = model.predict(ctx_xyz, qry_xyz, qry_type).cpu().numpy().astype(np.float32, copy=False)
        all_true.append(target_vec)
        all_pred.append(pred)
        all_effective.append(effective)
    if not all_true:
        raise RuntimeError(f"empty eval set for {spec.name}")
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    eff = np.concatenate(all_effective, axis=0)
    if spec.name == "udf_distance":
        y_true = y_true[eff, :, 0]
        y_pred = y_pred[eff, :, 0]
        out = _distance_metrics(y_true, y_pred, float(tau))
    elif spec.name == "mesh_normal":
        y_true = y_true[eff]
        y_pred = y_pred[eff]
        out = _normal_metrics(y_true, y_pred)
    elif spec.name == "mesh_normal_unsigned":
        y_true = y_true[eff]
        y_pred = y_pred[eff]
        out = _normal_metrics(y_true, y_pred, unsigned=True)
    else:
        raise KeyError(f"unsupported continuous task={spec.name}")
    out.update(
        {
            "task_name": spec.name,
            "split": spec.split,
            "cache_root": spec.cache_root,
            "context_source": spec.context_source,
            "control": str(control),
        }
    )
    return out


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
    tau: float,
) -> dict[str, Any]:
    model, _ckpt, _args = load_distnorm_continuous_model(str(ckpt), torch.device(str(device)))
    specs = _build_eval_datasets(
        mix_config_path,
        seed=int(seed),
        n_ctx=int(n_ctx),
        n_qry=int(n_qry),
        max_samples_per_task=int(max_samples_per_task),
        split_override=split_override,
        task_filter=task_filter,
        eval_sample_mode=str(eval_sample_mode),
        query_order=str(query_order),
    )
    by_control = {}
    for control in controls:
        rows = [
            _evaluate_dataset(
                model,
                spec,
                device=torch.device(str(device)),
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                control=str(control),
                tau=float(tau),
            )
            for spec in specs
        ]
        by_control[str(control)] = {"results": rows}
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
                if row["task_name"] == "udf_distance":
                    task_delta[row["task_name"]] = {
                        "delta_mae": float(row["mae"]) - float(base["mae"]),
                        "delta_rmse": float(row["rmse"]) - float(base["rmse"]),
                        f"delta_iou@{float(tau):g}": float(row[f"iou@{float(tau):g}"]) - float(base[f"iou@{float(tau):g}"]),
                        "delta_code_acc_proxy": float(row["code_acc_proxy"]) - float(base["code_acc_proxy"]),
                    }
                elif row["task_name"] in {"mesh_normal", "mesh_normal_unsigned"}:
                    task_delta[row["task_name"]] = {
                        "delta_mean_cos": float(row["mean_cos"]) - float(base["mean_cos"]),
                        "delta_angle_deg": float(row["angle_deg"]) - float(base["angle_deg"]),
                        "delta_code_acc_proxy": float(row["code_acc_proxy"]) - float(base["code_acc_proxy"]),
                    }
            deltas[control] = {"per_task": task_delta}
    return {"by_control": by_control, "deltas_vs_correct": deltas}


def _flatten(tag: str, payload: dict[str, Any], controls: list[str]) -> list[dict[str, Any]]:
    base = payload["by_control"]["correct"]
    rows: list[dict[str, Any]] = []
    for row in base["results"]:
        task = str(row["task_name"])
        item = {
            "suite": str(tag),
            "task": task,
            "context_source": row["context_source"],
            "n_tokens": int(row["n_tokens"]),
            "code_acc_proxy": float(row["code_acc_proxy"]),
            "majority_code_acc": float(row["majority_code_acc"]),
            "pred_top1_share": float(row["pred_top1_share"]),
            "pred_unique_codes": int(row["pred_unique_codes"]),
        }
        if task == "udf_distance":
            item.update(
                {
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "iou@0.05": float(row["iou@0.05"]),
                }
            )
        elif task == "mesh_normal":
            item.update(
                {
                    "mean_cos": float(row["mean_cos"]),
                    "angle_deg": float(row["angle_deg"]),
                }
            )
        for c in controls:
            if c == "correct":
                continue
            delta = payload["deltas_vs_correct"].get(c, {}).get("per_task", {}).get(task)
            if delta is None:
                continue
            for k, v in delta.items():
                item[f"{k}({c})"] = float(v)
        rows.append(item)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
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
    cols = sorted({k for r in rows for k in r.keys()})
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval_multitype_continuous_suite")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--same_mix_config", type=str, required=True)
    p.add_argument("--offdiag_mix_config", type=str, default="")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--max_samples_per_task", type=int, default=256)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--tasks", type=str, default=DEFAULT_TASKS)
    p.add_argument("--controls", type=str, default=",".join(DEFAULT_CONTROLS))
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--query_order", type=str, default="sampled")
    p.add_argument("--tau", type=float, default=0.05)
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
        "tau": float(args.tau),
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
            tau=float(args.tau),
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
            tau=float(args.tau),
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
