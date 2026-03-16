import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .modelnet40_index import list_npz


def _scalar_int(arr, default=0) -> int:
    if arr is None:
        return int(default)
    a = np.asarray(arr)
    if a.size == 0:
        return int(default)
    return int(a.reshape(-1)[0])


def _scalar_float(arr, default=0.0) -> float:
    if arr is None:
        return float(default)
    a = np.asarray(arr)
    if a.size == 0:
        return float(default)
    return float(a.reshape(-1)[0])


def _decode_scalar(arr, default="") -> str:
    if arr is None:
        return str(default)
    a = np.asarray(arr)
    if a.size == 0:
        return str(default)
    x = a.reshape(-1)[0]
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _shape_key(synset: str, model_id: str) -> str:
    return f"{synset}/{model_id}"


def _visibility_allzero_rate(vis_sig: np.ndarray) -> float:
    vis = np.asarray(vis_sig)
    if vis.ndim == 1:
        vis = vis.reshape(-1, 1)
    if vis.size == 0:
        return 1.0
    return float((np.abs(vis).sum(axis=1) <= 1e-8).mean())


def _extract_record(cache_root: str, split: str, npz_path: str) -> Dict[str, object]:
    with np.load(npz_path, allow_pickle=False) as data:
        synset = _decode_scalar(data.get("synset"), default=npz_path.split(os.sep)[-2])
        model_id = _decode_scalar(
            data.get("model_id"), default=os.path.splitext(os.path.basename(npz_path))[0]
        )
        rec: Dict[str, object] = {
            "key": _shape_key(synset, model_id),
            "synset": synset,
            "model_id": model_id,
            "split": split,
            "src_relpath": os.path.relpath(npz_path, start=cache_root),
            "mesh_source_path": _decode_scalar(data.get("mesh_source_path"), default=""),
            "is_watertight": bool(_scalar_int(data.get("is_watertight"), 0)),
            "is_winding_consistent": bool(_scalar_int(data.get("is_winding_consistent"), 0)),
            "vertex_count": _scalar_int(data.get("vertex_count"), 0),
            "face_count": _scalar_int(data.get("face_count"), 0),
            "surface_area": _scalar_float(data.get("surface_area"), 0.0),
            "volume": _scalar_float(data.get("volume"), 0.0),
        }
        if "udf_surf_hit_out" in data:
            rec["udf_surf_hit_rate"] = float(np.asarray(data["udf_surf_hit_out"], dtype=np.float32).mean())
        if "mesh_surf_vis_sig" in data:
            rec["mesh_surf_vis_allzero_rate"] = _visibility_allzero_rate(data["mesh_surf_vis_sig"])
        return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train:test")
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--out_summary_json", type=str, default="")
    ap.add_argument("--out_tsv", type=str, default="")
    ap.add_argument("--require_watertight", type=int, default=1, choices=[0, 1])
    ap.add_argument("--require_winding_consistent", type=int, default=1, choices=[0, 1])
    ap.add_argument("--min_faces", type=int, default=1)
    ap.add_argument("--min_vertices", type=int, default=1)
    ap.add_argument(
        "--min_udf_hit_rate",
        type=float,
        default=-1.0,
        help="Disabled when < 0. Otherwise require udf_surf_hit_rate >= threshold.",
    )
    ap.add_argument(
        "--max_visibility_allzero_rate",
        type=float,
        default=-1.0,
        help="Disabled when < 0. Otherwise require mesh_surf_vis_allzero_rate <= threshold.",
    )
    args = ap.parse_args()

    cache_root = os.path.abspath(args.cache_root)
    splits = [s for s in args.splits.split(":") if s]
    if not splits:
        raise ValueError("no splits provided")

    records: List[Dict[str, object]] = []
    kept: List[Dict[str, object]] = []
    drop_reasons = Counter()
    examined_by_split = Counter()
    kept_by_split = Counter()
    examined_by_synset = Counter()
    kept_by_synset = Counter()

    for split in splits:
        paths = list_npz(cache_root, split)
        if not paths:
            raise RuntimeError(f"no npz under {cache_root}/{split}")
        for npz_path in paths:
            rec = _extract_record(cache_root, split, npz_path)
            reasons: List[str] = []
            examined_by_split[split] += 1
            examined_by_synset[str(rec["synset"])] += 1
            if bool(args.require_watertight) and not bool(rec["is_watertight"]):
                reasons.append("not_watertight")
            if bool(args.require_winding_consistent) and not bool(rec["is_winding_consistent"]):
                reasons.append("not_winding_consistent")
            if int(rec["face_count"]) < int(args.min_faces):
                reasons.append("too_few_faces")
            if int(rec["vertex_count"]) < int(args.min_vertices):
                reasons.append("too_few_vertices")
            if args.min_udf_hit_rate >= 0.0:
                hit_rate = rec.get("udf_surf_hit_rate")
                if hit_rate is None:
                    reasons.append("missing_udf_hit_rate")
                elif float(hit_rate) < float(args.min_udf_hit_rate):
                    reasons.append("low_udf_hit_rate")
            if args.max_visibility_allzero_rate >= 0.0:
                allzero_rate = rec.get("mesh_surf_vis_allzero_rate")
                if allzero_rate is None:
                    reasons.append("missing_visibility_allzero_rate")
                elif float(allzero_rate) > float(args.max_visibility_allzero_rate):
                    reasons.append("high_visibility_allzero_rate")

            rec["eligible"] = len(reasons) == 0
            rec["drop_reasons"] = reasons
            records.append(rec)
            if reasons:
                for reason in reasons:
                    drop_reasons[reason] += 1
            else:
                kept.append(rec)
                kept_by_split[split] += 1
                kept_by_synset[str(rec["synset"])] += 1

    meta = {
        "cache_root": cache_root,
        "splits": splits,
        "criteria": {
            "require_watertight": bool(args.require_watertight),
            "require_winding_consistent": bool(args.require_winding_consistent),
            "min_faces": int(args.min_faces),
            "min_vertices": int(args.min_vertices),
            "min_udf_hit_rate": None if args.min_udf_hit_rate < 0.0 else float(args.min_udf_hit_rate),
            "max_visibility_allzero_rate": None
            if args.max_visibility_allzero_rate < 0.0
            else float(args.max_visibility_allzero_rate),
        },
        "counts": {
            "examined": len(records),
            "kept": len(kept),
            "kept_rate": float(len(kept) / max(len(records), 1)),
        },
        "examined_by_split": dict(examined_by_split),
        "kept_by_split": dict(kept_by_split),
        "drop_reasons": dict(drop_reasons),
        "examined_by_synset": dict(sorted(examined_by_synset.items())),
        "kept_by_synset": dict(sorted(kept_by_synset.items())),
    }

    out = {"meta": meta, "records": kept}

    out_json = os.path.abspath(args.out_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = {
        "meta": meta,
        "preview": {
            "first_records": kept[:10],
        },
    }
    out_summary_json = args.out_summary_json or os.path.splitext(out_json)[0] + ".summary.json"
    with open(os.path.abspath(out_summary_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    out_tsv = args.out_tsv or os.path.splitext(out_json)[0] + ".tsv"
    with open(os.path.abspath(out_tsv), "w", encoding="utf-8") as f:
        f.write(
            "\t".join(
                [
                    "key",
                    "split",
                    "synset",
                    "model_id",
                    "src_relpath",
                    "is_watertight",
                    "is_winding_consistent",
                    "vertex_count",
                    "face_count",
                    "surface_area",
                    "volume",
                    "udf_surf_hit_rate",
                    "mesh_surf_vis_allzero_rate",
                ]
            )
            + "\n"
        )
        for rec in kept:
            f.write(
                "\t".join(
                    [
                        str(rec["key"]),
                        str(rec["split"]),
                        str(rec["synset"]),
                        str(rec["model_id"]),
                        str(rec["src_relpath"]),
                        str(int(bool(rec["is_watertight"]))),
                        str(int(bool(rec["is_winding_consistent"]))),
                        str(rec["vertex_count"]),
                        str(rec["face_count"]),
                        f"{float(rec['surface_area']):.8g}",
                        f"{float(rec['volume']):.8g}",
                        ""
                        if "udf_surf_hit_rate" not in rec
                        else f"{float(rec['udf_surf_hit_rate']):.8g}",
                        ""
                        if "mesh_surf_vis_allzero_rate" not in rec
                        else f"{float(rec['mesh_surf_vis_allzero_rate']):.8g}",
                    ]
                )
                + "\n"
            )

    print(f"[subset] wrote manifest: {out_json}")
    print(f"[subset] wrote summary: {out_summary_json}")
    print(f"[subset] wrote tsv: {out_tsv}")
    print(f"[subset] kept {len(kept)} / {len(records)} ({len(kept) / max(len(records), 1):.4f})")
    print(f"[subset] drop reasons: {dict(drop_reasons)}")


if __name__ == "__main__":
    main()
