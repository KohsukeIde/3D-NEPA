from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from nepa3d.analysis.mesh_metrics import mesh_metrics
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import (
    build_eval_datasets,
    load_cqa_model,
)
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ASK_DISTANCE, decode_distance_from_vocab
from nepa3d.utils import grid as grid_utils


def _parse_task_filter(text: str) -> set[str]:
    s = str(text).strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


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


def _decode_scalar_str(x: Any) -> str:
    arr = np.asarray(x)
    if arr.dtype.kind in {"S", "a"}:
        return bytes(arr.reshape(-1)[0]).decode("utf-8")
    if arr.dtype.kind == "U":
        return str(arr.reshape(-1)[0])
    return str(arr.reshape(-1)[0])


def _load_normalized_mesh(npz_path: str) -> trimesh.Trimesh:
    with np.load(npz_path, allow_pickle=False) as npz:
        mesh_path = _decode_scalar_str(npz["mesh_source_path"])
        center = np.asarray(npz["norm_center"], dtype=np.float32).reshape(1, 3)
        scale = float(np.asarray(npz["norm_scale"], dtype=np.float32).reshape(-1)[0])
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    verts = (np.asarray(mesh.vertices, dtype=np.float32) - center) / max(scale, 1e-8)
    return trimesh.Trimesh(vertices=verts, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)


def _closest_distance(mesh: trimesh.Trimesh, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    try:
        _cp, dist, _tri = trimesh.proximity.closest_point(mesh, xyz)
        return np.asarray(dist, dtype=np.float32).reshape(-1)
    except Exception:
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float32))
        dist, _ = tree.query(xyz, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float32).reshape(-1)


def _mesh_from_udf_grid(udf_grid: np.ndarray, *, level: float) -> tuple[np.ndarray, np.ndarray]:
    from skimage import measure

    udf = np.asarray(udf_grid, dtype=np.float32)
    g = int(udf.shape[0])
    voxel = 2.0 / float(g - 1)
    verts, faces, _normals, _values = measure.marching_cubes(udf, level=float(level), spacing=(voxel, voxel, voxel))
    verts = verts + np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    return verts.astype(np.float32, copy=False), faces.astype(np.int64, copy=False)


def _predict_distance_codes(
    model: torch.nn.Module,
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
        t_t = torch.full((b, int(e - s)), int(ASK_DISTANCE), device=device, dtype=torch.long)
        with torch.no_grad():
            code = model.generate(ctx_xyz, q_t, t_t)
        out[:, s:e] = decode_distance_from_vocab(code.detach().cpu().numpy())
    return out


def _aggregate(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("completion_udfdist_cqa")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mix_config_path", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4, help="Number of shapes per dense-grid decode batch.")
    p.add_argument("--max_shapes", type=int, default=16)
    p.add_argument("--split_override", type=str, default="eval")
    p.add_argument("--task_filter", type=str, default="udf_distance")
    p.add_argument("--eval_sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--grid_res", type=int, default=12)
    p.add_argument("--chunk_n_query", type=int, default=64)
    p.add_argument("--tau_list", type=str, default="0.01,0.02,0.05")
    p.add_argument("--mesh_eval", type=int, default=0, choices=[0, 1])
    p.add_argument("--mc_level", type=float, default=0.02)
    p.add_argument("--mesh_num_samples", type=int, default=10000)
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    model, ckpt, _model_args = load_cqa_model(str(args.ckpt), device)
    tau_list = _parse_tau_list(str(args.tau_list))
    specs = build_eval_datasets(
        str(args.mix_config_path),
        seed=int(args.seed),
        n_ctx=int(args.n_ctx),
        n_qry=int(args.n_qry),
        max_samples_per_task=int(args.max_shapes),
        split_override=(str(args.split_override).strip() or None),
        task_filter=_parse_task_filter(str(args.task_filter)),
        eval_sample_mode=str(args.eval_sample_mode),
    )
    centers = grid_utils.make_grid_centers_np(int(args.grid_res)).reshape(-1, 3).astype(np.float32, copy=False)

    all_results: List[Dict[str, Any]] = []
    for spec in specs:
        if spec.name != "udf_distance":
            continue
        samples = [spec.dataset[i] for i in range(len(spec.dataset))]
        per_shape: List[Dict[str, Any]] = []
        pbar = tqdm(range(0, len(samples), int(args.batch_size)), desc=f"{spec.name}:{spec.context_source}")
        for start in pbar:
            batch_samples = samples[start : start + int(args.batch_size)]
            ctx_batch = torch.stack([s["ctx_xyz"] for s in batch_samples], dim=0).to(device)
            pred_flat = _predict_distance_codes(
                model,
                ctx_xyz=ctx_batch,
                centers=centers,
                chunk_n_query=int(args.chunk_n_query),
            )
            for bi, sample in enumerate(batch_samples):
                path = str(sample["path"])
                gt_mesh = _load_normalized_mesh(path)
                gt_flat = _closest_distance(gt_mesh, centers)
                pred = pred_flat[bi].reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                gt = gt_flat.reshape(int(args.grid_res), int(args.grid_res), int(args.grid_res))
                err = pred - gt
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

                if int(args.mesh_eval) == 1:
                    try:
                        v_pred, f_pred = _mesh_from_udf_grid(pred, level=float(args.mc_level))
                        v_gt, f_gt = _mesh_from_udf_grid(gt, level=float(args.mc_level))
                        mm = mesh_metrics(
                            v_pred,
                            f_pred,
                            v_gt,
                            f_gt,
                            num_samples=int(args.mesh_num_samples),
                            fscore_tau=float(tau_list[0]),
                        )
                        rec["mesh_chamfer_l2"] = float(mm.get("chamfer_l2", float("nan")))
                        rec["mesh_chamfer_l1"] = float(mm.get("chamfer_l1", float("nan")))
                        rec["mesh_fscore"] = float(mm.get("fscore", float("nan")))
                    except Exception as e:
                        rec["mesh_error"] = str(e)
                per_shape.append(rec)

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
        "task_filter": str(args.task_filter),
        "eval_sample_mode": str(args.eval_sample_mode),
        "grid_res": int(args.grid_res),
        "chunk_n_query": int(args.chunk_n_query),
        "tau_list": tau_list,
        "mesh_eval": int(args.mesh_eval),
        "results": all_results,
    }
    out_path = Path(str(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
