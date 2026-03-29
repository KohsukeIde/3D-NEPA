from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from nepa3d.analysis.mesh_metrics import mesh_metrics
from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import (
    _aggregate,
    _closest_distance,
    _export_mesh,
    _export_point_cloud,
    _load_normalized_mesh,
    _predict_distance_codes,
    _safe_mesh_from_udf_grid,
)
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import (
    build_eval_datasets,
    load_cqa_model,
)
from nepa3d.tracks.patch_nepa.cqa.analysis.open3d_mesh_baselines import (
    bpa_from_points,
    poisson_from_points,
)
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import (
    CQA_VOCAB_VERSION,
    query_name_to_id,
)
from nepa3d.utils import grid as grid_utils


def _parse_tau_list(text: str) -> List[float]:
    vals: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        vals = [0.01]
    return vals


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _stable_int_seed(text: str) -> int:
    x = 0
    for ch in str(text):
        x = (x * 131 + ord(ch)) % 2147483647
    return int(x)


def _corruption_conditions(
    *,
    dropout_keep_list: List[float],
    gaussian_sigma_list: List[float],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"name": "clean", "family": "clean"}]
    for keep in dropout_keep_list:
        out.append(
            {
                "name": f"dropout@{keep:.2f}",
                "family": "dropout",
                "keep_ratio": float(keep),
            }
        )
    for sigma in gaussian_sigma_list:
        out.append(
            {
                "name": f"gaussian@{sigma:.3f}",
                "family": "gaussian",
                "sigma": float(sigma),
            }
        )
    return out


def _apply_corruption(
    ctx_xyz: np.ndarray,
    *,
    condition: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    pts = np.asarray(ctx_xyz, dtype=np.float32).reshape(-1, 3)
    family = str(condition["family"])
    if family == "clean":
        return pts
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    if family == "dropout":
        keep_ratio = float(condition["keep_ratio"])
        keep_n = max(1, int(round(float(pts.shape[0]) * keep_ratio)))
        idx = rng.choice(int(pts.shape[0]), size=keep_n, replace=False)
        return np.asarray(pts[idx], dtype=np.float32)
    if family == "gaussian":
        sigma = float(condition["sigma"])
        noise = rng.normal(loc=0.0, scale=sigma, size=pts.shape).astype(np.float32)
        out = pts + noise
        np.clip(out, -1.0, 1.0, out=out)
        return out
    raise KeyError(f"unknown corruption family={family}")


def _record_field_metrics(
    rec: Dict[str, Any],
    *,
    pred_grid: np.ndarray,
    gt_grid: np.ndarray,
    tau_list: List[float],
) -> None:
    err = np.asarray(pred_grid, dtype=np.float32) - np.asarray(gt_grid, dtype=np.float32)
    rec["mae"] = float(np.mean(np.abs(err)))
    rec["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    for tau in tau_list:
        pred_occ = pred_grid <= float(tau)
        gt_occ = gt_grid <= float(tau)
        inter = int(np.logical_and(pred_occ, gt_occ).sum())
        union = int(np.logical_or(pred_occ, gt_occ).sum())
        rec[f"iou@{tau:g}"] = float(inter / max(union, 1))


def _record_mesh_metrics(
    rec: Dict[str, Any],
    *,
    pred_mesh_vf: tuple[np.ndarray, np.ndarray],
    gt_mesh_source,
    mesh_num_samples: int,
    tau_list: List[float],
) -> None:
    mm = mesh_metrics(
        pred_mesh_vf[0],
        pred_mesh_vf[1],
        np.asarray(gt_mesh_source.vertices, dtype=np.float32),
        np.asarray(gt_mesh_source.faces, dtype=np.int64),
        n_samples=int(mesh_num_samples),
        taus=(float(tau_list[0]),),
    )
    rec["mesh_chamfer_l2"] = float(mm.get("chamfer_l2", float("nan")))
    rec["mesh_chamfer_l1"] = float(mm.get("chamfer_l1", float("nan")))
    rec["mesh_fscore"] = float(mm.get(f"fscore@{float(tau_list[0])}", float("nan")))
    rec["mesh_target"] = "gt_source_mesh"


def _asset_dir(
    assets_root: Path,
    *,
    context_name: str,
    condition_name: str,
    method: str,
    sample_path: str,
) -> Path:
    stem = Path(str(sample_path)).stem
    out = assets_root / context_name / condition_name / method / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _export_common_assets(asset_dir: Path, *, ctx_xyz: np.ndarray, gt_mesh_source) -> None:
    np.save(asset_dir / "context_xyz.npy", np.asarray(ctx_xyz, dtype=np.float32))
    _export_point_cloud(asset_dir / "context_points.ply", ctx_xyz)
    _export_mesh(
        asset_dir / "gt_mesh.obj",
        np.asarray(gt_mesh_source.vertices, dtype=np.float32),
        np.asarray(gt_mesh_source.faces, dtype=np.int64),
    )


def _export_cqa_assets(
    asset_dir: Path,
    *,
    pred_grid: np.ndarray,
    gt_grid: np.ndarray,
    pred_mesh_vf: tuple[np.ndarray, np.ndarray] | None,
    mc_level: float,
) -> None:
    np.savez_compressed(
        asset_dir / "fields.npz",
        pred_udf=np.asarray(pred_grid, dtype=np.float32),
        gt_udf=np.asarray(gt_grid, dtype=np.float32),
        mc_level=np.asarray([float(mc_level)], dtype=np.float32),
    )
    if pred_mesh_vf is not None:
        _export_mesh(asset_dir / "pred_mesh.obj", pred_mesh_vf[0], pred_mesh_vf[1])


def _export_baseline_assets(
    asset_dir: Path,
    *,
    pred_mesh_vf: tuple[np.ndarray, np.ndarray] | None,
) -> None:
    if pred_mesh_vf is not None:
        _export_mesh(asset_dir / "pred_mesh.obj", pred_mesh_vf[0], pred_mesh_vf[1])


def _summary_from_per_shape(
    per_shape: List[Dict[str, Any]],
    *,
    include_field_metrics: bool,
    tau_list: List[float],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if include_field_metrics:
        metrics["mae"] = _aggregate([float(r["mae"]) for r in per_shape if "mae" in r])
        metrics["rmse"] = _aggregate([float(r["rmse"]) for r in per_shape if "rmse" in r])
        for tau in tau_list:
            key = f"iou@{tau:g}"
            metrics[key] = _aggregate([float(r[key]) for r in per_shape if key in r])
    for key in ["mesh_chamfer_l2", "mesh_chamfer_l1", "mesh_fscore"]:
        metrics[key] = _aggregate([float(r[key]) for r in per_shape if key in r])
    return {
        "num_shapes": int(len(per_shape)),
        "metrics": metrics,
        "per_shape": per_shape,
    }


def _write_table(rows: List[Dict[str, Any]], csv_path: Path, md_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "context",
        "condition",
        "method",
        "num_shapes",
        "mesh_chamfer_l1_mean",
        "mesh_chamfer_l1_std",
        "mesh_chamfer_l2_mean",
        "mesh_chamfer_l2_std",
        "mesh_fscore_mean",
        "mesh_fscore_std",
        "mae_mean",
        "rmse_mean",
        "iou_at_tau_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    lines = [
        "| context | condition | method | num_shapes | chamfer_l1 | chamfer_l2 | fscore | mae | rmse | iou |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {context} | {condition} | {method} | {num_shapes} | "
            "{mesh_chamfer_l1_mean:.6f} | {mesh_chamfer_l2_mean:.6f} | {mesh_fscore_mean:.6f} | "
            "{mae_mean:.6f} | {rmse_mean:.6f} | {iou_at_tau_mean:.6f} |".format(**row)
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("completion_udfdist_degraded_cqa")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--same_mix_config_path", type=str, required=True)
    p.add_argument("--offdiag_mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_shapes", type=int, default=64)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--query_order", type=str, default="sampled")
    p.add_argument("--grid_res", type=int, default=16)
    p.add_argument("--chunk_n_query", type=int, default=64)
    p.add_argument("--tau_list", type=str, default="0.01,0.02,0.05")
    p.add_argument("--mesh_num_samples", type=int, default=10000)
    p.add_argument("--mc_level", type=float, default=0.05)
    p.add_argument("--dropout_keep_list", type=str, default="0.50,0.25,0.10,0.05")
    p.add_argument("--gaussian_sigma_list", type=str, default="0.01,0.02,0.05")
    p.add_argument("--export_assets", type=int, default=1, choices=[0, 1])
    p.add_argument("--max_asset_shapes", type=int, default=8)
    p.add_argument("--assets_root", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--output_md", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    model, ckpt, model_args = load_cqa_model(str(args.ckpt), device)
    codec_version = str(model_args.get("codec_version", CQA_VOCAB_VERSION))
    distance_query_type = int(query_name_to_id(codec_version)["udf_distance"])
    tau_list = _parse_tau_list(str(args.tau_list))
    conditions = _corruption_conditions(
        dropout_keep_list=_parse_float_list(str(args.dropout_keep_list)),
        gaussian_sigma_list=_parse_float_list(str(args.gaussian_sigma_list)),
    )
    contexts = [
        ("same", str(args.same_mix_config_path)),
        ("offdiag", str(args.offdiag_mix_config_path)),
    ]
    centers = grid_utils.make_grid_centers_np(int(args.grid_res)).reshape(-1, 3).astype(np.float32, copy=False)
    export_assets = int(args.export_assets) == 1
    assets_root = Path(str(args.assets_root))
    if export_assets:
        assets_root.mkdir(parents=True, exist_ok=True)

    context_payloads: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    for context_name, mix_config in contexts:
        specs = build_eval_datasets(
            mix_config,
            seed=int(args.seed),
            n_ctx=int(args.n_ctx),
            n_qry=int(args.n_qry),
            max_samples_per_task=int(args.max_shapes),
            split_override=(str(args.split_override).strip() or None),
            task_filter={"udf_distance"},
            eval_sample_mode=str(args.eval_sample_mode),
            query_order=str(args.query_order),
            codec_version=codec_version,
        )
        if len(specs) != 1:
            raise RuntimeError(f"expected exactly one udf_distance spec for {context_name}, got {len(specs)}")
        spec = specs[0]
        samples = [spec.dataset[i] for i in range(len(spec.dataset))]
        asset_paths = {str(samples[i]["path"]) for i in range(min(int(args.max_asset_shapes), len(samples)))}
        condition_payloads: List[Dict[str, Any]] = []
        for condition in conditions:
            method_records: Dict[str, List[Dict[str, Any]]] = {"cqa": [], "poisson": [], "bpa": []}
            pbar = tqdm(
                range(0, len(samples), int(args.batch_size)),
                desc=f"{context_name}:{condition['name']}",
            )
            for start in pbar:
                batch_samples = samples[start : start + int(args.batch_size)]
                ctx_np_batch: List[np.ndarray] = []
                for sample in batch_samples:
                    seed_text = f"{sample['path']}::{context_name}::{condition['name']}::{args.seed}"
                    degraded = _apply_corruption(
                        np.asarray(sample["ctx_xyz"].cpu().numpy(), dtype=np.float32),
                        condition=condition,
                        seed=_stable_int_seed(seed_text),
                    )
                    ctx_np_batch.append(degraded)
                ctx_sizes = {int(x.shape[0]) for x in ctx_np_batch}
                if len(ctx_sizes) != 1:
                    raise RuntimeError(
                        f"mixed degraded ctx sizes inside one condition batch: {sorted(ctx_sizes)} "
                        f"context={context_name} condition={condition['name']}"
                    )
                ctx_batch = torch.from_numpy(np.stack(ctx_np_batch, axis=0)).to(device)
                pred_flat = _predict_distance_codes(
                    model,
                    ctx_xyz=ctx_batch,
                    centers=centers,
                    chunk_n_query=int(args.chunk_n_query),
                    query_type_id=distance_query_type,
                    codec_version=codec_version,
                )
                for bi, sample in enumerate(batch_samples):
                    path = str(sample["path"])
                    ctx_np = ctx_np_batch[bi]
                    gt_mesh = _load_normalized_mesh(path)
                    gt_flat = _closest_distance(gt_mesh, centers)
                    pred_grid = pred_flat[bi].reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                    gt_grid = gt_flat.reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))

                    cqa_rec: Dict[str, Any] = {
                        "path": path,
                        "synset": str(sample.get("synset", "")),
                        "context_source": str(sample.get("context_source", spec.context_source)),
                        "context_name": context_name,
                        "condition": str(condition["name"]),
                        "method": "cqa",
                        "codec_version": codec_version,
                        "n_ctx_effective": int(ctx_np.shape[0]),
                    }
                    _record_field_metrics(cqa_rec, pred_grid=pred_grid, gt_grid=gt_grid, tau_list=tau_list)
                    pred_mesh_vf: tuple[np.ndarray, np.ndarray] | None = None
                    try:
                        pred_mesh_vf = _safe_mesh_from_udf_grid(pred_grid, level=float(args.mc_level))
                        _record_mesh_metrics(
                            cqa_rec,
                            pred_mesh_vf=pred_mesh_vf,
                            gt_mesh_source=gt_mesh,
                            mesh_num_samples=int(args.mesh_num_samples),
                            tau_list=tau_list,
                        )
                    except Exception as e:
                        cqa_rec["mesh_error"] = str(e)
                    if export_assets and path in asset_paths:
                        asset_dir = _asset_dir(
                            assets_root,
                            context_name=context_name,
                            condition_name=str(condition["name"]),
                            method="cqa",
                            sample_path=path,
                        )
                        _export_common_assets(asset_dir, ctx_xyz=ctx_np, gt_mesh_source=gt_mesh)
                        _export_cqa_assets(
                            asset_dir,
                            pred_grid=pred_grid,
                            gt_grid=gt_grid,
                            pred_mesh_vf=pred_mesh_vf,
                            mc_level=float(args.mc_level),
                        )
                        cqa_rec["asset_dir"] = str(asset_dir)
                    method_records["cqa"].append(cqa_rec)

                    for method in ("poisson", "bpa"):
                        rec: Dict[str, Any] = {
                            "path": path,
                            "synset": str(sample.get("synset", "")),
                            "context_source": str(sample.get("context_source", spec.context_source)),
                            "context_name": context_name,
                            "condition": str(condition["name"]),
                            "method": method,
                            "n_ctx_effective": int(ctx_np.shape[0]),
                        }
                        pred_mesh_vf = None
                        try:
                            if method == "poisson":
                                pred_mesh_vf = poisson_from_points(ctx_np)
                            else:
                                pred_mesh_vf = bpa_from_points(ctx_np)
                            _record_mesh_metrics(
                                rec,
                                pred_mesh_vf=pred_mesh_vf,
                                gt_mesh_source=gt_mesh,
                                mesh_num_samples=int(args.mesh_num_samples),
                                tau_list=tau_list,
                            )
                        except Exception as e:
                            rec["mesh_error"] = str(e)
                        if export_assets and path in asset_paths:
                            asset_dir = _asset_dir(
                                assets_root,
                                context_name=context_name,
                                condition_name=str(condition["name"]),
                                method=method,
                                sample_path=path,
                            )
                            _export_common_assets(asset_dir, ctx_xyz=ctx_np, gt_mesh_source=gt_mesh)
                            _export_baseline_assets(asset_dir, pred_mesh_vf=pred_mesh_vf)
                            rec["asset_dir"] = str(asset_dir)
                        method_records[method].append(rec)

            method_payloads: List[Dict[str, Any]] = []
            for method in ("cqa", "poisson", "bpa"):
                include_field_metrics = method == "cqa"
                summary = _summary_from_per_shape(
                    method_records[method],
                    include_field_metrics=include_field_metrics,
                    tau_list=tau_list,
                )
                method_payloads.append(
                    {
                        "method": method,
                        **summary,
                    }
                )
                metrics = summary["metrics"]
                aggregate_rows.append(
                    {
                        "context": context_name,
                        "condition": str(condition["name"]),
                        "method": method,
                        "num_shapes": int(summary["num_shapes"]),
                        "mesh_chamfer_l1_mean": float(metrics["mesh_chamfer_l1"]["mean"]),
                        "mesh_chamfer_l1_std": float(metrics["mesh_chamfer_l1"]["std"]),
                        "mesh_chamfer_l2_mean": float(metrics["mesh_chamfer_l2"]["mean"]),
                        "mesh_chamfer_l2_std": float(metrics["mesh_chamfer_l2"]["std"]),
                        "mesh_fscore_mean": float(metrics["mesh_fscore"]["mean"]),
                        "mesh_fscore_std": float(metrics["mesh_fscore"]["std"]),
                        "mae_mean": float(metrics.get("mae", {}).get("mean", float("nan"))),
                        "rmse_mean": float(metrics.get("rmse", {}).get("mean", float("nan"))),
                        "iou_at_tau_mean": float(metrics.get(f"iou@{tau_list[0]:g}", {}).get("mean", float("nan"))),
                    }
                )
            condition_payloads.append(
                {
                    "condition": {
                        k: (float(v) if isinstance(v, (np.floating, float)) else v)
                        for k, v in condition.items()
                    },
                    "methods": method_payloads,
                }
            )
        context_payloads.append(
            {
                "context": context_name,
                "mix_config_path": mix_config,
                "task_name": spec.name,
                "split": spec.split,
                "codec_version": codec_version,
                "conditions": condition_payloads,
            }
        )

    payload = {
        "ckpt": str(args.ckpt),
        "train_global_step": int(ckpt.get("global_step", -1)),
        "same_mix_config_path": str(args.same_mix_config_path),
        "offdiag_mix_config_path": str(args.offdiag_mix_config_path),
        "codec_version": codec_version,
        "grid_res": int(args.grid_res),
        "mc_level": float(args.mc_level),
        "tau_list": tau_list,
        "mesh_num_samples": int(args.mesh_num_samples),
        "dropout_keep_list": _parse_float_list(str(args.dropout_keep_list)),
        "gaussian_sigma_list": _parse_float_list(str(args.gaussian_sigma_list)),
        "results": context_payloads,
        "aggregate_rows": aggregate_rows,
    }
    out_json = Path(str(args.output_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    _write_table(aggregate_rows, Path(str(args.output_csv)), Path(str(args.output_md)))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
