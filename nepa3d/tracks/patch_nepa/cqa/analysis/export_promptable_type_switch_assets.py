from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa import (
    _closest_distance,
    _load_normalized_mesh,
    _predict_distance_codes,
    _safe_mesh_from_udf_grid,
)
from nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_tokens import load_cqa_model
from nepa3d.tracks.patch_nepa.cqa.data.cqa_codec import ANSWER_RANGES
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import V2PrimitiveCQADataset
from nepa3d.utils import grid as grid_utils


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def _export_pc(path: Path, xyz: np.ndarray, *, colors: np.ndarray | None = None) -> None:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        trimesh.points.PointCloud(xyz, colors=colors).export(path)
    else:
        trimesh.points.PointCloud(xyz).export(path)


def _normal_colors(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.float32).reshape(-1, 3)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)
    rgb = np.clip((n + 1.0) * 0.5, 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def _pick_paths(cache_root: str, split: str, *, max_shapes: int, seed: int, mode: str) -> list[str]:
    paths = list_npz(cache_root, split)
    if max_shapes <= 0 or len(paths) <= int(max_shapes):
        return list(paths)
    if str(mode) == "head":
        return list(paths[: int(max_shapes)])
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    take = rng.choice(len(paths), size=int(max_shapes), replace=False)
    return [paths[int(i)] for i in take.tolist()]


def _build_single(
    path: str,
    *,
    task_name: str,
    context_source: str,
    n_ctx: int,
    n_qry: int,
    seed: int,
) -> dict[str, Any]:
    ds = V2PrimitiveCQADataset(
        [path],
        task_name=task_name,
        context_source=context_source,
        n_ctx=int(n_ctx),
        n_qry=int(n_qry),
        seed=int(seed),
        mode="eval",
        query_order="sampled",
    )
    return ds[0]


def _decode_normal(code: np.ndarray) -> np.ndarray:
    from nepa3d.tracks.patch_nepa.cqa.data import cqa_codec as codec

    code = np.asarray(code, dtype=np.int64)
    lo, hi = ANSWER_RANGES["mesh_normal"]
    idx = np.clip(code.reshape(-1) - int(lo), 0, int(hi - lo) - 1)
    vec = np.asarray(codec._NORMAL_DIRS, dtype=np.float32)[idx]
    return vec.reshape(*code.shape, 3).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("export_promptable_type_switch_assets")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cache_root", type=str, required=True)
    p.add_argument("--split", type=str, default="eval")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_shapes", type=int, default=8)
    p.add_argument("--sample_mode", type=str, default="random", choices=["head", "random"])
    p.add_argument("--context_source", type=str, default="pc_bank", choices=["surf", "pc_bank"])
    p.add_argument("--n_ctx", type=int, default=2048)
    p.add_argument("--n_qry_surface", type=int, default=1000000)
    p.add_argument("--tasks", type=str, default="udf_distance,mesh_normal")
    p.add_argument("--distance_grid_res", type=int, default=16)
    p.add_argument("--distance_chunk_n_query", type=int, default=64)
    p.add_argument("--distance_mc_level", type=float, default=0.02)
    p.add_argument("--output_root", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(str(args.device))
    model, _ckpt, _train_args = load_cqa_model(str(args.ckpt), device)
    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    paths = _pick_paths(
        str(args.cache_root),
        str(args.split),
        max_shapes=int(args.max_shapes),
        seed=int(args.seed),
        mode=str(args.sample_mode),
    )
    out_root = Path(str(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    centers = grid_utils.make_grid_centers_np(int(args.distance_grid_res)).reshape(-1, 3).astype(np.float32, copy=False)

    for i, path in enumerate(paths):
        shape_id = _safe_name(Path(path).stem)
        shape_dir = out_root / f"{i:04d}_{shape_id}_{_safe_name(args.context_source)}"
        shape_dir.mkdir(parents=True, exist_ok=True)
        sample_ref = _build_single(
            path,
            task_name=tasks[0],
            context_source=str(args.context_source),
            n_ctx=int(args.n_ctx),
            n_qry=int(args.n_qry_surface),
            seed=int(args.seed) + i,
        )
        ctx_xyz = sample_ref["ctx_xyz"].numpy()
        _export_pc(shape_dir / "context_points.ply", ctx_xyz)
        np.save(shape_dir / "context_xyz.npy", ctx_xyz)

        shape_summary: dict[str, Any] = {
            "path": path,
            "context_source": str(args.context_source),
            "tasks": {},
        }
        gt_mesh = _load_normalized_mesh(path)

        for task_name in tasks:
            if task_name == "udf_distance":
                pred_flat = _predict_distance_codes(
                    model,
                    ctx_xyz=torch.from_numpy(ctx_xyz[None]).to(device=device, dtype=torch.float32),
                    centers=centers,
                    chunk_n_query=int(args.distance_chunk_n_query),
                )[0]
                gt_flat = _closest_distance(gt_mesh, centers)
                pred_grid = pred_flat.reshape(int(args.distance_grid_res), int(args.distance_grid_res), int(args.distance_grid_res))
                gt_grid = gt_flat.reshape(int(args.distance_grid_res), int(args.distance_grid_res), int(args.distance_grid_res))
                np.savez_compressed(shape_dir / "udf_distance_fields.npz", pred_udf=pred_grid, gt_udf=gt_grid)
                try:
                    pv, pf = _safe_mesh_from_udf_grid(pred_grid, level=float(args.distance_mc_level))
                    trimesh.Trimesh(vertices=pv, faces=pf, process=False).export(shape_dir / "udf_distance_pred_mesh.obj")
                except Exception:
                    pv = None
                try:
                    gv, gf = _safe_mesh_from_udf_grid(gt_grid, level=float(args.distance_mc_level))
                    trimesh.Trimesh(vertices=gv, faces=gf, process=False).export(shape_dir / "udf_distance_gt_levelset_mesh.obj")
                except Exception:
                    gv = None
                trimesh.Trimesh(vertices=np.asarray(gt_mesh.vertices), faces=np.asarray(gt_mesh.faces), process=False).export(shape_dir / "gt_mesh.obj")
                shape_summary["tasks"][task_name] = {
                    "mae": float(np.mean(np.abs(pred_flat - gt_flat))),
                    "rmse": float(np.sqrt(np.mean((pred_flat - gt_flat) ** 2))),
                    "pred_mesh_ok": pv is not None,
                    "gt_levelset_mesh_ok": gv is not None,
                }
                continue

            sample = _build_single(
                path,
                task_name=task_name,
                context_source=str(args.context_source),
                n_ctx=int(args.n_ctx),
                n_qry=int(args.n_qry_surface),
                seed=int(args.seed) + i,
            )
            q_xyz = sample["qry_xyz"].numpy()
            q_type = sample["qry_type"].unsqueeze(0).to(device=device, dtype=torch.long)
            gt_code = sample["answer_code"].numpy()
            with torch.no_grad():
                pred_code = model.generate(
                    ctx_xyz=sample["ctx_xyz"].unsqueeze(0).to(device=device, dtype=torch.float32),
                    qry_xyz=sample["qry_xyz"].unsqueeze(0).to(device=device, dtype=torch.float32),
                    qry_type=q_type,
                )[0].detach().cpu().numpy()
            np.save(shape_dir / f"{task_name}_qry_xyz.npy", q_xyz)
            np.save(shape_dir / f"{task_name}_pred_code.npy", pred_code)
            np.save(shape_dir / f"{task_name}_gt_code.npy", gt_code)

            if task_name == "mesh_normal":
                gt_val = _decode_normal(gt_code)
                pred_val = _decode_normal(pred_code)
                np.save(shape_dir / f"{task_name}_pred_vec.npy", pred_val)
                np.save(shape_dir / f"{task_name}_gt_vec.npy", gt_val)
                _export_pc(shape_dir / f"{task_name}_pred_points.ply", q_xyz, colors=_normal_colors(pred_val))
                _export_pc(shape_dir / f"{task_name}_gt_points.ply", q_xyz, colors=_normal_colors(gt_val))
                cos = np.sum(pred_val * gt_val, axis=1)
                mean_cos = float(np.mean(np.clip(cos, -1.0, 1.0)))
                shape_summary["tasks"][task_name] = {
                    "token_acc": float(np.mean(pred_code.reshape(-1) == gt_code.reshape(-1))),
                    "mean_cos": mean_cos,
                }
            else:
                shape_summary["tasks"][task_name] = {
                    "token_acc": float(np.mean(pred_code.reshape(-1) == gt_code.reshape(-1))),
                }

        (shape_dir / "summary.json").write_text(json.dumps(shape_summary, indent=2) + "\n")
        index.append({"asset_dir": str(shape_dir), **shape_summary})

    (out_root / "index.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
