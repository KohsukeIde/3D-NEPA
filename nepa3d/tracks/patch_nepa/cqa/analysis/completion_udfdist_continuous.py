from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from nepa3d.analysis.mesh_metrics import mesh_metrics
from nepa3d.tracks.kplane.data.udfdist_worldv3_dataset import build_worldv3_udfdist_eval_specs
from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import (
    _aggregate,
    _closest_distance,
    _export_shape_assets,
    _load_normalized_mesh,
    _parse_tau_list,
    _safe_mesh_from_udf_grid,
)
from nepa3d.tracks.patch_nepa.cqa.models.primitive_answering_udfdist_continuous import (
    load_udfdist_continuous_model,
)
from nepa3d.utils import grid as grid_utils


def _predict_distance(
    model,
    *,
    ctx_xyz: torch.Tensor,
    centers: np.ndarray,
    chunk_n_query: int,
) -> np.ndarray:
    device = ctx_xyz.device
    b = int(ctx_xyz.shape[0])
    n_total = int(centers.shape[0])
    out = np.zeros((b, n_total), dtype=np.float32)
    qbs = max(1, int(chunk_n_query))
    for s in range(0, n_total, qbs):
        e = min(n_total, s + qbs)
        q_np = np.repeat(centers[None, s:e, :], b, axis=0).astype(np.float32, copy=False)
        q_t = torch.from_numpy(q_np).to(device)
        with torch.no_grad():
            pred = model.predict(ctx_xyz, q_t)
        out[:, s:e] = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("completion_udfdist_continuous")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_shapes", type=int, default=64)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--grid_res", type=int, default=16)
    p.add_argument("--chunk_n_query", type=int, default=128)
    p.add_argument("--tau_list", type=str, default="0.01,0.02,0.05")
    p.add_argument("--mesh_eval", type=int, default=1, choices=[0, 1])
    p.add_argument("--mc_level", type=float, default=0.05)
    p.add_argument("--mesh_num_samples", type=int, default=10000)
    p.add_argument("--export_assets", type=int, default=1, choices=[0, 1])
    p.add_argument("--assets_root", type=str, default="")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    model, ckpt, _model_args = load_udfdist_continuous_model(str(args.ckpt), device)
    tau_list = _parse_tau_list(str(args.tau_list))
    specs = build_worldv3_udfdist_eval_specs(
        str(args.mix_config_path),
        seed=int(args.seed),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        max_samples_per_task=int(args.max_shapes),
        split_override=(str(args.split_override).strip() or None),
        eval_sample_mode=str(args.eval_sample_mode),
    )
    centers = grid_utils.make_grid_centers_np(int(args.grid_res)).reshape(-1, 3).astype(np.float32, copy=False)
    out_path = Path(str(args.output_json))
    export_assets = int(args.export_assets) == 1
    assets_root = None
    if export_assets:
        if str(args.assets_root).strip():
            assets_root = Path(str(args.assets_root))
        else:
            assets_root = out_path.with_suffix("")
            assets_root = assets_root.parent / f"{assets_root.name}_assets"
        assets_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    shape_counter = 0
    for spec in specs:
        samples = [spec.dataset[i] for i in range(len(spec.dataset))]
        per_shape: List[Dict[str, Any]] = []
        pbar = tqdm(range(0, len(samples), int(args.batch_size)), desc=f"{spec.name}:{spec.context_source}")
        for start in pbar:
            batch_samples = samples[start : start + int(args.batch_size)]
            ctx_batch = torch.stack([s["ctx_xyz"] for s in batch_samples], dim=0).to(device)
            pred_flat = _predict_distance(model, ctx_xyz=ctx_batch, centers=centers, chunk_n_query=int(args.chunk_n_query))
            for bi, sample in enumerate(batch_samples):
                path = str(sample["path"])
                gt_mesh = _load_normalized_mesh(path)
                gt_flat = _closest_distance(gt_mesh, centers)
                pred = pred_flat[bi].reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                gt = gt_flat.reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                err = pred - gt
                pred_mesh_vf = None
                gt_levelset_vf = None
                rec: Dict[str, Any] = {
                    "path": path,
                    "synset": str(sample.get("synset", "")),
                    "context_source": str(sample.get("context_source", spec.context_source)),
                    "mae": float(np.mean(np.abs(err))),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                }
                for tau in tau_list:
                    pred_occ = pred <= float(tau)
                    gt_occ = gt <= float(tau)
                    inter = int(np.logical_and(pred_occ, gt_occ).sum())
                    union = int(np.logical_or(pred_occ, gt_occ).sum())
                    rec[f"iou@{tau:g}"] = float(inter / max(union, 1))
                if int(args.mesh_eval) == 1 or export_assets:
                    try:
                        pred_mesh_vf = _safe_mesh_from_udf_grid(pred, level=float(args.mc_level))
                        gt_levelset_vf = _safe_mesh_from_udf_grid(gt, level=float(args.mc_level))
                        if int(args.mesh_eval) == 1:
                            mm = mesh_metrics(
                                pred_mesh_vf[0],
                                pred_mesh_vf[1],
                                gt_levelset_vf[0],
                                gt_levelset_vf[1],
                                n_samples=int(args.mesh_num_samples),
                                taus=(float(tau_list[0]),),
                            )
                            rec["mesh_chamfer_l2"] = float(mm.get("chamfer_l2", float("nan")))
                            rec["mesh_chamfer_l1"] = float(mm.get("chamfer_l1", float("nan")))
                            rec["mesh_fscore"] = float(mm.get(f"fscore@{float(tau_list[0])}", float("nan")))
                    except Exception as e:
                        rec["mesh_error"] = str(e)
                if export_assets and assets_root is not None:
                    from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import _make_asset_dir

                    asset_dir = _make_asset_dir(assets_root, sample, shape_counter)
                    _export_shape_assets(
                        asset_dir=asset_dir,
                        sample=sample,
                        rec=rec,
                        ctx_xyz=np.asarray(sample["ctx_xyz"].cpu().numpy(), dtype=np.float32),
                        pred_udf=pred,
                        gt_udf=gt,
                        mc_level=float(args.mc_level),
                        gt_mesh_source=gt_mesh,
                        pred_mesh_vf=pred_mesh_vf,
                        gt_levelset_vf=gt_levelset_vf,
                    )
                    rec["asset_dir"] = str(asset_dir)
                per_shape.append(rec)
                shape_counter += 1
        summary = {
            "task_name": spec.name,
            "context_source": spec.context_source,
            "split": spec.split,
            "cache_root": spec.cache_root,
            "num_shapes": int(len(per_shape)),
            "grid_res": int(args.grid_res),
            "chunk_n_query": int(args.chunk_n_query),
            "metrics": {
                "mae": _aggregate([float(r["mae"]) for r in per_shape]),
                "rmse": _aggregate([float(r["rmse"]) for r in per_shape]),
            },
            "per_shape": per_shape,
        }
        for tau in tau_list:
            key = f"iou@{tau:g}"
            summary["metrics"][key] = _aggregate([float(r[key]) for r in per_shape if key in r])
        if int(args.mesh_eval) == 1:
            for key in ["mesh_chamfer_l2", "mesh_chamfer_l1", "mesh_fscore"]:
                summary["metrics"][key] = _aggregate([float(r[key]) for r in per_shape if key in r])
        all_results.append(summary)
    payload = {
        "ckpt": str(args.ckpt),
        "train_global_step": int(ckpt.get("global_step", -1)),
        "mix_config_path": str(args.mix_config_path),
        "split_override": str(args.split_override),
        "eval_sample_mode": str(args.eval_sample_mode),
        "grid_res": int(args.grid_res),
        "chunk_n_query": int(args.chunk_n_query),
        "tau_list": tau_list,
        "mesh_eval": int(args.mesh_eval),
        "results": all_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
