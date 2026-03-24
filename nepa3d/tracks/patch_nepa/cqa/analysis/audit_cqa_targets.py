from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from nepa3d.data.modelnet40_index import list_npz
from nepa3d.tracks.patch_nepa.cqa.data.dataset_cqa import V2PrimitiveCQADataset

DEFAULT_TASKS = (
    "mesh_visibility",
    "udf_thickness",
    "udf_clearance",
    "udf_distance",
)


def _parse_tasks(text: str) -> List[str]:
    s = str(text).strip()
    if not s:
        return list(DEFAULT_TASKS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _default_split_for_task(task_name: str) -> str:
    if str(task_name).startswith("mesh_"):
        return "train_mesh"
    if str(task_name).startswith("udf_"):
        return "train_udf"
    raise KeyError(f"no default split for task={task_name}")


def _quantiles(arr: np.ndarray) -> Dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    if x.size <= 0:
        return {}
    q = np.quantile(x, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    labels = ("min", "p01", "p05", "p50", "p95", "p99", "max")
    return {k: float(v) for k, v in zip(labels, q.tolist())}


def _entropy_from_counter(cnt: collections.Counter) -> float:
    n = float(sum(cnt.values()))
    if n <= 0.0:
        return 0.0
    probs = np.asarray(list(cnt.values()), dtype=np.float64) / n
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _codec_summary(ds: V2PrimitiveCQADataset) -> Dict[str, Any]:
    code_counter: collections.Counter = collections.Counter()
    per_shape_entropy: List[float] = []
    per_shape_top1_share: List[float] = []
    per_shape_unique: List[float] = []

    for i in range(len(ds)):
        item = ds[i]
        codes = item["answer_code"].numpy().tolist()
        cnt = collections.Counter(codes)
        n = max(sum(cnt.values()), 1)
        code_counter.update(cnt)
        per_shape_entropy.append(_entropy_from_counter(cnt))
        per_shape_top1_share.append(float(max(cnt.values()) / n))
        per_shape_unique.append(float(len(cnt)))

    n_total = int(sum(code_counter.values()))
    top10 = code_counter.most_common(10)
    top1_code, top1_count = top10[0] if top10 else (-1, 0)
    top5_count = sum(v for _k, v in top10[:5])
    return {
        "n_samples": int(len(ds)),
        "n_tokens": n_total,
        "unique_codes": int(len(code_counter)),
        "top1_code": int(top1_code),
        "top1_share": float(top1_count / max(n_total, 1)),
        "top5_cumulative_share": float(top5_count / max(n_total, 1)),
        "entropy_bits": _entropy_from_counter(code_counter),
        "majority_baseline_acc": float(top1_count / max(n_total, 1)),
        "top10": [[int(k), int(v)] for k, v in top10],
        "per_shape_mean_top1_share": float(np.mean(per_shape_top1_share)) if per_shape_top1_share else 0.0,
        "per_shape_mean_unique_codes": float(np.mean(per_shape_unique)) if per_shape_unique else 0.0,
        "per_shape_mean_entropy_bits": float(np.mean(per_shape_entropy)) if per_shape_entropy else 0.0,
    }


def _raw_summary(task_name: str, paths: List[str]) -> Dict[str, Any]:
    arrays: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    vis_sig_all: List[np.ndarray] = []

    for path in paths:
        with np.load(path, allow_pickle=False) as npz:
            if task_name in {"mesh_visibility", "mesh_viscount"}:
                vis = np.asarray(npz["mesh_surf_vis_sig"], dtype=np.float32)
                vis_sig_all.append(vis)
                if "mesh_surf_viscount" in npz:
                    arrays["mesh_surf_viscount"].append(np.asarray(npz["mesh_surf_viscount"], dtype=np.float32))
                if "mesh_surf_ao" in npz:
                    arrays["mesh_surf_ao"].append(np.asarray(npz["mesh_surf_ao"], dtype=np.float32))
            elif task_name in {"udf_thickness", "udf_thickness_valid_qbin"}:
                for key in ("udf_surf_thickness", "udf_surf_t_in", "udf_surf_t_out", "udf_surf_hit_out"):
                    if key in npz:
                        arrays[key].append(np.asarray(npz[key], dtype=np.float32))
            elif task_name == "udf_clearance":
                for key in ("udf_surf_clear_front", "udf_surf_clear_back", "udf_surf_hit_out"):
                    if key in npz:
                        arrays[key].append(np.asarray(npz[key], dtype=np.float32))
            elif task_name == "udf_distance":
                for key in ("udf_qry_dist", "udf_qry_grad", "udf_qry_src_code"):
                    if key in npz:
                        arrays[key].append(np.asarray(npz[key]))
            else:
                raise KeyError(f"unsupported audit task={task_name}")

    out: Dict[str, Any] = {"fields": {}}
    if task_name in {"mesh_visibility", "mesh_viscount"}:
        if vis_sig_all:
            vis = np.concatenate(vis_sig_all, axis=0)
            bits = (vis > 0.5).astype(np.int64)
            packed = np.sum(bits * (1 << np.arange(bits.shape[1], dtype=np.int64)).reshape(1, -1), axis=1)
            out["fields"]["mesh_surf_vis_sig"] = {
                "bit_activation_rate": [float(x) for x in bits.mean(axis=0).tolist()],
                "all_zero_share": float((packed == 0).mean()),
                "unique_packed_codes": int(np.unique(packed).size),
            }
            out["visibility_fallback_recorded_in_cache"] = False
        for key in ("mesh_surf_viscount", "mesh_surf_ao"):
            if arrays.get(key):
                arr = np.concatenate(arrays[key]).reshape(-1)
                out["fields"][key] = {"quantiles": _quantiles(arr)}
        return out

    if task_name == "udf_distance":
        if arrays.get("udf_qry_dist"):
            arr = np.concatenate(arrays["udf_qry_dist"]).reshape(-1)
            out["fields"]["udf_qry_dist"] = {"quantiles": _quantiles(arr)}
        if arrays.get("udf_qry_grad"):
            arr = np.concatenate(arrays["udf_qry_grad"]).reshape(-1, 3)
            norms = np.linalg.norm(arr, axis=1)
            out["fields"]["udf_qry_grad_norm"] = {"quantiles": _quantiles(norms)}
        if arrays.get("udf_qry_src_code"):
            arr = np.concatenate(arrays["udf_qry_src_code"]).reshape(-1).astype(np.int64)
            cnt = collections.Counter(arr.tolist())
            out["fields"]["udf_qry_src_code"] = {
                "counts": {str(int(k)): int(v) for k, v in sorted(cnt.items())}
            }
        return out

    if task_name == "udf_thickness_valid_qbin":
        thick = np.concatenate(arrays["udf_surf_thickness"]).reshape(-1)
        t_in = np.concatenate(arrays["udf_surf_t_in"]).reshape(-1)
        t_out = np.concatenate(arrays["udf_surf_t_out"]).reshape(-1)
        hit_out = np.concatenate(arrays["udf_surf_hit_out"]).reshape(-1)
        eps = np.float32(1e-4)
        max_t = np.float32(1.999)
        keep = (
            (hit_out > np.float32(0.5))
            & (thick > eps)
            & (t_in > eps)
            & (t_out > eps)
            & (t_in < max_t)
            & (t_out < max_t)
        )
        out["fields"]["udf_thickness_valid_support"] = {
            "support_rate": float(keep.mean()),
            "quantiles": _quantiles(thick[keep]) if bool(keep.any()) else {},
        }
        out["fields"]["udf_surf_hit_out"] = {
            "mean": float(hit_out.mean()),
            "positive_rate": float((hit_out > 0.5).mean()),
        }
        return out

    for key, chunks in arrays.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks).reshape(-1)
        if key == "udf_surf_hit_out":
            out["fields"][key] = {
                "mean": float(arr.mean()),
                "positive_rate": float((arr > 0.5).mean()),
            }
        else:
            out["fields"][key] = {"quantiles": _quantiles(arr)}
    return out


def audit_task(
    *,
    cache_root: str,
    task_name: str,
    split: str,
    max_shapes: int,
    n_ctx: int,
    n_qry: int,
    seed: int,
) -> Dict[str, Any]:
    paths = list_npz(cache_root, split)
    if int(max_shapes) > 0:
        paths = paths[: int(max_shapes)]
    if not paths:
        raise FileNotFoundError(f"no npz found: cache_root={cache_root} split={split}")
    ds = V2PrimitiveCQADataset(
        paths,
        task_name=task_name,
        context_source="surf",
        n_ctx=int(n_ctx),
        n_qry=int(n_qry),
        seed=int(seed),
        mode="eval",
    )
    return {
        "task_name": str(task_name),
        "split": str(split),
        "cache_root": str(cache_root),
        "n_paths": int(len(paths)),
        "codec": _codec_summary(ds),
        "raw": _raw_summary(task_name, list(paths)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit CQA raw targets and codec-space distributions.")
    ap.add_argument("--cache-root", required=True)
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--max-shapes", type=int, default=256)
    ap.add_argument("--n-ctx", type=int, default=2048)
    ap.add_argument("--n-qry", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    tasks = _parse_tasks(args.tasks)
    out: Dict[str, Any] = {
        "cache_root": str(args.cache_root),
        "max_shapes": int(args.max_shapes),
        "n_ctx": int(args.n_ctx),
        "n_qry": int(args.n_qry),
        "seed": int(args.seed),
        "tasks": {},
    }
    for task_name in tasks:
        split = _default_split_for_task(task_name)
        out["tasks"][task_name] = audit_task(
            cache_root=str(args.cache_root),
            task_name=str(task_name),
            split=split,
            max_shapes=int(args.max_shapes),
            n_ctx=int(args.n_ctx),
            n_qry=int(args.n_qry),
            seed=int(args.seed),
        )

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.out_json:
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
