from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .modelnet40_index import list_npz


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


def _decode_scalar(arr, default="") -> str:
    a = np.asarray(arr)
    if a.size == 0:
        return str(default)
    x = a.reshape(-1)[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _pack_vis_codes(vis: np.ndarray) -> np.ndarray:
    bits = (np.asarray(vis, dtype=np.float32) > 0.5).astype(np.int64)
    if bits.ndim == 1:
        bits = bits.reshape(-1, 1)
    return np.sum(bits * (1 << np.arange(bits.shape[1], dtype=np.int64)).reshape(1, -1), axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--splits", default="train:test")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    cache_root = str(args.cache_root)
    splits = [s for s in str(args.splits).split(":") if s]

    shape_meta: List[Dict[str, Any]] = []
    vis_code_counter: collections.Counter = collections.Counter()
    qry_src_counter: collections.Counter = collections.Counter()

    surf_fields: Dict[str, List[np.ndarray]] = collections.defaultdict(list)
    vol_fields: Dict[str, List[np.ndarray]] = collections.defaultdict(list)

    for split in splits:
        for path in list_npz(cache_root, split):
            with np.load(path, allow_pickle=False) as npz:
                shape_meta.append(
                    {
                        "split": split,
                        "synset": _decode_scalar(npz.get("synset"), default=path.split("/")[-2]),
                        "model_id": _decode_scalar(npz.get("model_id"), default=Path(path).stem),
                        "is_watertight": int(np.asarray(npz.get("is_watertight", [0])).reshape(-1)[0]),
                        "is_winding_consistent": int(
                            np.asarray(npz.get("is_winding_consistent", [0])).reshape(-1)[0]
                        ),
                        "visibility_fallback_used": int(
                            np.asarray(npz.get("visibility_fallback_used", [0])).reshape(-1)[0]
                        ),
                        "udf_surf_hit_out_rate": float(
                            np.asarray(npz.get("udf_surf_hit_out_rate", [0.0]), dtype=np.float32).reshape(-1)[0]
                        ),
                        "udf_clear_front_valid_rate": float(
                            np.asarray(npz.get("udf_clear_front_valid_rate", [0.0]), dtype=np.float32).reshape(-1)[0]
                        ),
                        "udf_clear_back_valid_rate": float(
                            np.asarray(npz.get("udf_clear_back_valid_rate", [0.0]), dtype=np.float32).reshape(-1)[0]
                        ),
                        "udf_probe_valid_rate": float(
                            np.asarray(npz.get("udf_probe_valid_rate", [0.0]), dtype=np.float32).reshape(-1)[0]
                        ),
                    }
                )

                if "mesh_surf_vis_sig" in npz:
                    packed = _pack_vis_codes(np.asarray(npz["mesh_surf_vis_sig"], dtype=np.float32))
                    vis_code_counter.update(packed.tolist())
                for key in ("mesh_surf_viscount", "mesh_surf_ao", "udf_surf_thickness", "udf_surf_clear_front", "udf_surf_clear_back"):
                    if key in npz:
                        surf_fields[key].append(np.asarray(npz[key], dtype=np.float32).reshape(-1))
                for key in ("udf_qry_dist",):
                    if key in npz:
                        vol_fields[key].append(np.asarray(npz[key], dtype=np.float32).reshape(-1))
                if "udf_qry_src_code" in npz:
                    qry_src = np.asarray(npz["udf_qry_src_code"]).reshape(-1).astype(np.int64)
                    qry_src_counter.update(qry_src.tolist())

    def _concat_dict_fields(d: Dict[str, List[np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for key, chunks in d.items():
            if not chunks:
                continue
            arr = np.concatenate(chunks, axis=0)
            out[key] = {"quantiles": _quantiles(arr)}
        return out

    vis_total = int(sum(vis_code_counter.values()))
    vis_top1_code, vis_top1_count = vis_code_counter.most_common(1)[0]
    vis_top5_count = sum(v for _k, v in vis_code_counter.most_common(5))
    shape_count = len(shape_meta)

    summary = {
        "cache_root": cache_root,
        "splits": splits,
        "shape_counts": {
            "total": shape_count,
            "by_split": dict(collections.Counter(x["split"] for x in shape_meta)),
        },
        "shape_quality": {
            "watertight_rate": float(sum(x["is_watertight"] for x in shape_meta) / max(shape_count, 1)),
            "winding_consistent_rate": float(
                sum(x["is_winding_consistent"] for x in shape_meta) / max(shape_count, 1)
            ),
            "visibility_fallback_used_rate": float(
                sum(x["visibility_fallback_used"] for x in shape_meta) / max(shape_count, 1)
            ),
        },
        "task_validity": {
            "udf_surf_hit_out_rate_mean": float(np.mean([x["udf_surf_hit_out_rate"] for x in shape_meta])),
            "udf_clear_front_valid_rate_mean": float(
                np.mean([x["udf_clear_front_valid_rate"] for x in shape_meta])
            ),
            "udf_clear_back_valid_rate_mean": float(
                np.mean([x["udf_clear_back_valid_rate"] for x in shape_meta])
            ),
            "udf_probe_valid_rate_mean": float(np.mean([x["udf_probe_valid_rate"] for x in shape_meta])),
        },
        "raw_fields": {
            "surface": _concat_dict_fields(surf_fields),
            "volume": _concat_dict_fields(vol_fields),
        },
        "mesh_visibility_codec": {
            "unique_codes": int(len(vis_code_counter)),
            "top1_code": int(vis_top1_code),
            "top1_share": float(vis_top1_count / max(vis_total, 1)),
            "top5_cumulative_share": float(vis_top5_count / max(vis_total, 1)),
            "entropy_bits": _entropy_from_counter(vis_code_counter),
        },
        "udf_qry_src_code": {
            "counts": {str(int(k)): int(v) for k, v in sorted(qry_src_counter.items())}
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
