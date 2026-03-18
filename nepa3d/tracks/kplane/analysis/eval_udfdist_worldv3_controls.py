from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from nepa3d.tracks.kplane.data.udfdist_worldv3_dataset import build_worldv3_udfdist_eval_specs
from nepa3d.tracks.kplane.models.kplane import build_kplane_from_ckpt
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ASK_DISTANCE, encode_answers_from_fields

CONTROL_MODES = (
    "correct",
    "no_context",
    "wrong_shape_same_synset",
    "wrong_shape_other_synset",
    "shuffled_query",
)


def _code_stats(pred_dist: np.ndarray, target_dist: np.ndarray) -> Dict[str, Any]:
    pred_code = encode_answers_from_fields(ASK_DISTANCE, {"distance": np.asarray(pred_dist, dtype=np.float32)})
    tgt_code = encode_answers_from_fields(ASK_DISTANCE, {"distance": np.asarray(target_dist, dtype=np.float32)})
    pred_code = np.asarray(pred_code, dtype=np.int64).reshape(-1)
    tgt_code = np.asarray(tgt_code, dtype=np.int64).reshape(-1)
    pred_hist = np.bincount(pred_code, minlength=640)
    tgt_hist = np.bincount(tgt_code, minlength=640)
    pred_top1 = int(pred_hist.argmax())
    tgt_top1 = int(tgt_hist.argmax())
    return {
        "code_acc_proxy": float(np.mean(pred_code == tgt_code)),
        "majority_code_acc": float(tgt_hist.max()) / float(max(1, tgt_code.size)),
        "pred_top1_code": pred_top1,
        "pred_top1_share": float(pred_hist[pred_top1]) / float(max(1, pred_code.size)),
        "pred_unique_codes": int(np.sum(pred_hist > 0)),
        "target_top1_code": tgt_top1,
        "target_top1_share": float(tgt_hist[tgt_top1]) / float(max(1, tgt_code.size)),
        "target_unique_codes": int(np.sum(tgt_hist > 0)),
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    gt = y_true <= float(tau)
    pr = y_pred <= float(tau)
    inter = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum()) + 1e-9
    iou = float(inter / union)
    out = {"mae": mae, "rmse": rmse, f"iou@{float(tau):g}": iou}
    out.update(_code_stats(y_pred, y_true))
    return out


def _pick_donor_indices(samples: List[Dict[str, Any]], mode: str) -> List[int | None]:
    synsets = [str(s.get("synset", "")) for s in samples]
    donors: List[int | None] = []
    for i, syn in enumerate(synsets):
        donor = None
        for j in range(len(samples)):
            if i == j:
                continue
            same = synsets[j] == syn
            if mode == "wrong_shape_same_synset" and same:
                donor = j
                break
            if mode == "wrong_shape_other_synset" and not same:
                donor = j
                break
        donors.append(donor)
    return donors


def _predict_sample(
    model,
    *,
    ctx_xyz: np.ndarray,
    qry_xyz: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    ctx_t = torch.from_numpy(np.asarray(ctx_xyz, dtype=np.float32)).unsqueeze(0).to(device)
    ctx_d = torch.zeros((1, int(ctx_t.shape[1])), device=device, dtype=torch.float32)
    qry_t = torch.from_numpy(np.asarray(qry_xyz, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, *_ = model(ctx_t, ctx_d, qry_t)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def evaluate_control(
    model,
    samples: List[Dict[str, Any]],
    *,
    device: torch.device,
    control: str,
    tau: float,
) -> Dict[str, Any]:
    if control not in CONTROL_MODES:
        raise KeyError(f"unknown control={control}")
    donors = None
    if control in {"wrong_shape_same_synset", "wrong_shape_other_synset"}:
        donors = _pick_donor_indices(samples, control)
    all_true, all_pred, all_src = [], [], []
    for i, s in enumerate(tqdm(samples, desc=f"eval:{control}")):
        ctx_xyz = np.asarray(s["ctx_xyz"].numpy(), dtype=np.float32)
        qry_xyz = np.asarray(s["qry_xyz"].numpy(), dtype=np.float32)
        qry_dist = np.asarray(s["qry_dist"].numpy(), dtype=np.float32)
        qry_src = np.asarray(s["qry_src_code"].numpy(), dtype=np.int64)
        if control == "no_context":
            ctx_use = np.zeros((0, 3), dtype=np.float32)
            qry_use = qry_xyz
        elif control == "shuffled_query":
            perm = np.random.RandomState(i).permutation(int(qry_xyz.shape[0]))
            ctx_use = ctx_xyz
            qry_use = qry_xyz[perm]
        elif control in {"wrong_shape_same_synset", "wrong_shape_other_synset"}:
            donor = donors[i]
            if donor is None:
                continue
            ctx_use = np.asarray(samples[int(donor)]["ctx_xyz"].numpy(), dtype=np.float32)
            qry_use = qry_xyz
        else:
            ctx_use = ctx_xyz
            qry_use = qry_xyz
        pred = _predict_sample(model, ctx_xyz=ctx_use, qry_xyz=qry_use, device=device)
        all_true.append(qry_dist.reshape(-1))
        all_pred.append(pred.reshape(-1))
        all_src.append(qry_src.reshape(-1))
    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0,), dtype=np.float32)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,), dtype=np.float32)
    src = np.concatenate(all_src, axis=0) if all_src else np.zeros((0,), dtype=np.int64)
    out = _metrics(y_true, y_pred, tau=float(tau))
    breakdown = {}
    for code in sorted(np.unique(src).tolist()):
        m = src == int(code)
        if int(m.sum()) <= 0:
            continue
        row = _metrics(y_true[m], y_pred[m], tau=float(tau))
        row["query_src_code"] = int(code)
        row["n_tokens"] = int(m.sum())
        breakdown[str(code)] = row
    out["query_src_breakdown"] = breakdown
    out["n_tokens"] = int(y_true.size)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("eval_udfdist_worldv3_controls")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--max_samples_per_task", type=int, default=256)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--controls", type=str, default="correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    model, ckpt = build_kplane_from_ckpt(str(args.ckpt), device)
    controls = [x.strip() for x in str(args.controls).split(",") if x.strip()]
    specs = build_worldv3_udfdist_eval_specs(
        str(args.mix_config_path),
        seed=int(args.seed),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        max_samples_per_task=int(args.max_samples_per_task),
        split_override=(str(args.split_override).strip() or None),
        eval_sample_mode=str(args.eval_sample_mode),
    )
    by_control = {}
    for control in controls:
        if control not in CONTROL_MODES:
            raise KeyError(f"unknown control={control}")
        results = []
        for spec in specs:
            samples = [spec.dataset[i] for i in range(len(spec.dataset))]
            row = evaluate_control(model, samples, device=device, control=control, tau=float(args.tau))
            row.update({
                "task_name": spec.name,
                "context_source": spec.context_source,
                "split": spec.split,
                "cache_root": spec.cache_root,
            })
            results.append(row)
        overall_n = sum(int(r["n_tokens"]) for r in results)
        overall = {
            "mae": sum(float(r["mae"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_n)),
            "rmse": sum(float(r["rmse"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_n)),
            f"iou@{float(args.tau):g}": sum(float(r[f"iou@{float(args.tau):g}"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_n)),
            "code_acc_proxy": sum(float(r["code_acc_proxy"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_n)),
            "majority_code_acc": sum(float(r["majority_code_acc"]) * int(r["n_tokens"]) for r in results) / float(max(1, overall_n)),
            "n_tokens": int(overall_n),
        }
        by_control[control] = {"results": results, "overall": overall}
    baseline = by_control.get("correct")
    deltas = {}
    if baseline is not None:
        for control, summary in by_control.items():
            if control == "correct":
                continue
            deltas[control] = {
                "overall": {
                    "delta_mae": float(summary["overall"]["mae"]) - float(baseline["overall"]["mae"]),
                    "delta_rmse": float(summary["overall"]["rmse"]) - float(baseline["overall"]["rmse"]),
                    f"delta_iou@{float(args.tau):g}": float(summary["overall"][f"iou@{float(args.tau):g}"]) - float(baseline["overall"][f"iou@{float(args.tau):g}"]),
                    "delta_code_acc_proxy": float(summary["overall"]["code_acc_proxy"]) - float(baseline["overall"]["code_acc_proxy"]),
                }
            }
    payload = {
        "ckpt": str(args.ckpt),
        "train_global_step": int(ckpt.get("global_step", -1)),
        "mix_config_path": str(args.mix_config_path),
        "tau": float(args.tau),
        "controls": controls,
        "by_control": by_control,
        "deltas_vs_correct": deltas,
    }
    print(json.dumps(payload, indent=2))
    if str(args.output_json).strip():
        out = Path(str(args.output_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
