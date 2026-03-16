from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np

from .modelnet40_index import list_npz
from .preprocess_modelnet40 import normalize_mesh
from .preprocess_shapenet_v2 import (
    _finite_rate,
    _lt_rate,
    _normalize_params,
    _positive_rate,
    _vis_allzero_rate,
    _visibility_exact_available,
)

try:
    import trimesh
except Exception as e:
    raise RuntimeError("augment_shapenet_world_v3 requires trimesh") from e


WORLD_V3_KEYS = (
    "world_v3",
    "norm_center",
    "norm_scale",
    "visibility_fallback_used",
    "mesh_surf_vis_allzero_rate",
    "mesh_qry_vis_allzero_rate",
    "udf_surf_max_t",
    "udf_surf_hit_out_rate",
    "udf_clear_front_valid_rate",
    "udf_clear_back_valid_rate",
    "udf_probe_valid_rate",
)


def _load_mesh(mesh_path: str):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    return mesh


def _needs_update(npz: "np.lib.npyio.NpzFile") -> bool:
    return not all(k in npz for k in WORLD_V3_KEYS)


def _augment_one(
    task: Tuple[str, float],
    *,
    refresh: bool,
) -> Dict[str, object]:
    npz_path, udf_surf_max_t = task
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if (not bool(refresh)) and (not _needs_update(npz)):
                return {"path": npz_path, "status": "skipped"}
            payload = {k: np.asarray(npz[k]) for k in npz.files}

        mesh_path = payload.get("mesh_source_path")
        if mesh_path is None:
            raise KeyError("missing mesh_source_path")
        mesh_path_str = mesh_path.reshape(-1)[0]
        if isinstance(mesh_path_str, bytes):
            mesh_path_str = mesh_path_str.decode("utf-8", errors="replace")
        mesh = _load_mesh(str(mesh_path_str))
        norm_center, norm_scale = _normalize_params(mesh)
        mesh = normalize_mesh(mesh)

        visibility_fallback_used = np.asarray(
            [0 if _visibility_exact_available(mesh) else 1], dtype=np.int32
        )
        mesh_surf_vis_allzero_rate = _vis_allzero_rate(payload["mesh_surf_vis_sig"])
        mesh_qry_vis_allzero_rate = _vis_allzero_rate(payload["mesh_qry_vis_sig"])
        udf_surf_hit_out_rate = _positive_rate(payload["udf_surf_hit_out"])
        udf_clear_front_valid_rate = _positive_rate(payload["udf_surf_hit_out"])
        udf_clear_back_valid_rate = _lt_rate(
            payload["udf_surf_t_in"], float(udf_surf_max_t) - 1e-4
        )
        udf_probe_valid_rate = _finite_rate(
            payload["udf_surf_probe_front"],
            payload["udf_surf_probe_back"],
            payload["udf_surf_probe_thickness"],
        )

        payload.update(
            {
                "world_v3": np.int32(1),
                "norm_center": norm_center.astype(np.float32),
                "norm_scale": norm_scale.astype(np.float32),
                "visibility_fallback_used": visibility_fallback_used.astype(np.int32),
                "mesh_surf_vis_allzero_rate": mesh_surf_vis_allzero_rate.astype(np.float32),
                "mesh_qry_vis_allzero_rate": mesh_qry_vis_allzero_rate.astype(np.float32),
                "udf_surf_max_t": np.asarray([float(udf_surf_max_t)], dtype=np.float32),
                "udf_surf_hit_out_rate": udf_surf_hit_out_rate.astype(np.float32),
                "udf_clear_front_valid_rate": udf_clear_front_valid_rate.astype(np.float32),
                "udf_clear_back_valid_rate": udf_clear_back_valid_rate.astype(np.float32),
                "udf_probe_valid_rate": udf_probe_valid_rate.astype(np.float32),
            }
        )

        fd, tmp_path = tempfile.mkstemp(prefix=".worldv3_", suffix=".npz", dir=os.path.dirname(npz_path))
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **payload)
            os.replace(tmp_path, npz_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        return {
            "path": npz_path,
            "status": "updated",
            "visibility_fallback_used": int(visibility_fallback_used.reshape(-1)[0]),
        }
    except Exception as e:
        return {"path": npz_path, "status": "error", "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train:test")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--refresh", type=int, default=0, choices=[0, 1])
    ap.add_argument("--udf_surf_max_t", type=float, default=2.0)
    ap.add_argument("--out_summary_json", type=str, default="")
    args = ap.parse_args()

    cache_root = os.path.abspath(args.cache_root)
    splits = [s for s in args.splits.split(":") if s]
    paths: List[str] = []
    for split in splits:
        split_paths = list_npz(cache_root, split)
        if not split_paths:
            raise RuntimeError(f"no npz under {cache_root}/{split}")
        paths.extend(split_paths)

    worker = partial(
        _augment_one,
        refresh=bool(int(args.refresh)),
    )
    tasks = [(p, float(args.udf_surf_max_t)) for p in paths]
    if int(args.num_workers) <= 1:
        results = [worker(t) for t in tasks]
    else:
        with mp.Pool(processes=int(args.num_workers)) as pool:
            results = pool.map(worker, tasks)

    counts: Dict[str, int] = {}
    fallback_used = 0
    errors: List[Dict[str, object]] = []
    for r in results:
        status = str(r["status"])
        counts[status] = counts.get(status, 0) + 1
        if status == "updated":
            fallback_used += int(r.get("visibility_fallback_used", 0))
        if status == "error":
            errors.append(r)

    summary = {
        "cache_root": cache_root,
        "splits": splits,
        "num_paths": len(paths),
        "counts": counts,
        "visibility_fallback_used_count_among_updated": int(fallback_used),
        "errors": errors[:100],
    }

    out_summary_json = args.out_summary_json
    if out_summary_json:
        out_summary_json = os.path.abspath(out_summary_json)
        os.makedirs(os.path.dirname(out_summary_json) or ".", exist_ok=True)
        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
