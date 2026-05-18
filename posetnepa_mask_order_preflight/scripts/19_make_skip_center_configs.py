#!/usr/bin/env python3
"""Generate skip-k / center-position diagnostic configs.

This is the next diagnostic pass after PosetNEPA-Lite: keep the PointGPT
scaffold, but test whether the apparent full-finetune benefit comes from
immediate local continuity or positional side channels.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit("[error] PyYAML is required. Install with: pip install pyyaml") from exc


DEFAULT_VARIANTS = [
    "skip2:k=2,pos=normal,aux=0.0,mask=0.0",
    "skip4:k=4,pos=normal,aux=0.0,mask=0.0",
    "skip8:k=8,pos=normal,aux=0.0,mask=0.0",
    "poszero:k=1,pos=zero,aux=0.0,mask=0.0",
    "posshuffle:k=1,pos=shuffle,aux=0.0,mask=0.0",
    "centeraux:k=1,pos=normal,aux=0.1,mask=0.0",
    "skip4_poszero:k=4,pos=zero,aux=0.0,mask=0.0",
]


def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def rel_to_pointgpt(path: Path, pointgpt_dir: Path) -> str:
    return str(path.resolve().relative_to(pointgpt_dir.resolve()))


def safe_float(value: float | str) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def parse_variant(raw: str) -> dict:
    if ":" not in raw:
        raise SystemExit(f"[error] variant must be name:k=...,pos=...,aux=...,mask=... got: {raw}")
    name, body = raw.split(":", 1)
    fields: dict[str, str] = {}
    for item in [x.strip() for x in body.split(",") if x.strip()]:
        if "=" not in item:
            raise SystemExit(f"[error] malformed variant item `{item}` in {raw}")
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()
    return {
        "name": name.strip(),
        "skip_k": int(fields.get("k", fields.get("skip_k", "1"))),
        "pretrain_position_mode": fields.get("pos", fields.get("pretrain_position_mode", "normal")),
        "center_aux_weight": float(fields.get("aux", fields.get("center_aux_weight", "0.0"))),
        "mask_ratio": float(fields.get("mask", "0.0")),
        "order": fields.get("order", "simplified_morton"),
        "group_mode": fields.get("group", "fps_knn"),
        "readout_position_mode": fields.get("readout_pos", "normal"),
    }


def set_pretrain_common(
    cfg: dict,
    *,
    order: str,
    group: str,
    skip_k: int,
    position_mode: str,
    center_aux_weight: float,
    mask: float,
    max_epoch: int,
    total_bs: int | None,
) -> dict:
    cfg = copy.deepcopy(cfg)
    model = cfg.setdefault("model", {})
    model["order_mode"] = order
    model["group_mode"] = group
    model["nepa_skip_k"] = int(skip_k)
    model["pretrain_position_mode"] = position_mode
    model["center_aux_weight"] = float(center_aux_weight)
    tcfg = model.setdefault("transformer_config", {})
    tcfg["mask_ratio"] = float(mask)
    cfg["max_epoch"] = int(max_epoch)
    if isinstance(cfg.get("scheduler"), dict) and isinstance(cfg["scheduler"].get("kwargs"), dict):
        cfg["scheduler"]["kwargs"]["epochs"] = int(max_epoch)
    if total_bs is not None:
        cfg["total_bs"] = int(total_bs)
    return cfg


def set_ft_common(
    cfg: dict,
    *,
    order: str,
    group: str,
    readout_position_mode: str,
    max_epoch: int,
    total_bs: int | None,
) -> dict:
    cfg = copy.deepcopy(cfg)
    model = cfg.setdefault("model", {})
    model["order_mode"] = order
    model["group_mode"] = group
    if readout_position_mode != "normal":
        model["position_mode"] = readout_position_mode
    cfg["max_epoch"] = int(max_epoch)
    if isinstance(cfg.get("scheduler"), dict) and isinstance(cfg["scheduler"].get("kwargs"), dict):
        cfg["scheduler"]["kwargs"]["epochs"] = int(max_epoch)
    if total_bs is not None:
        cfg["total_bs"] = int(total_bs)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--variants", default=";".join(DEFAULT_VARIANTS))
    ap.add_argument("--base-pretrain", default="cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0.yaml")
    ap.add_argument("--base-ft-objbg", default="cfgs/PointGPT-S/finetune_scan_objbg.yaml")
    ap.add_argument("--base-ft-objonly", default="cfgs/PointGPT-S/finetune_scan_objonly.yaml")
    ap.add_argument("--base-ft-hardest", default="cfgs/PointGPT-S/finetune_scan_hardest.yaml")
    ap.add_argument("--max-epoch", type=int, default=30)
    ap.add_argument("--ft-max-epoch", type=int, default=50)
    ap.add_argument("--total-bs", type=int, default=None)
    ap.add_argument("--ft-total-bs", type=int, default=None)
    ap.add_argument("--out-dir", default="PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/skip_center")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/skip_center/manifest.json")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = (repo / args.pointgpt_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    manifest_path = repo / args.manifest

    variants = [parse_variant(x.strip()) for x in args.variants.split(";") if x.strip()]
    base_pre = load_yaml(pointgpt / args.base_pretrain)
    base_fts = {
        "objbg": load_yaml(pointgpt / args.base_ft_objbg),
        "objonly": load_yaml(pointgpt / args.base_ft_objonly),
        "hardest": load_yaml(pointgpt / args.base_ft_hardest),
    }

    manifest = {
        "purpose": "skip-k and center-position leakage diagnostics for PointGPT-style NEPA",
        "max_epoch": args.max_epoch,
        "ft_max_epoch": args.ft_max_epoch,
        "runs": [],
    }

    for variant in variants:
        vname = variant["name"].replace("/", "_")
        order = variant["order"]
        group = variant["group_mode"]
        mask = float(variant["mask_ratio"])
        skip_k = int(variant["skip_k"])
        pos = variant["pretrain_position_mode"]
        aux = float(variant["center_aux_weight"])
        readout_pos = variant["readout_position_mode"]
        run_id = (
            f"{vname}_{order}_m{safe_float(mask)}_k{skip_k}_"
            f"pos{pos}_aux{safe_float(aux)}"
        )

        pre_cfg = set_pretrain_common(
            base_pre,
            order=order,
            group=group,
            skip_k=skip_k,
            position_mode=pos,
            center_aux_weight=aux,
            mask=mask,
            max_epoch=args.max_epoch,
            total_bs=args.total_bs,
        )
        pre_path = out_dir / f"pretrain_nepa_{run_id}_e{args.max_epoch}.yaml"
        write_yaml(pre_cfg, pre_path)

        ft_entries = {}
        for split, base_ft in base_fts.items():
            ft_cfg = set_ft_common(
                base_ft,
                order=order,
                group=group,
                readout_position_mode=readout_pos,
                max_epoch=args.ft_max_epoch,
                total_bs=args.ft_total_bs,
            )
            ft_path = out_dir / f"finetune_scan_{split}_{run_id}_e{args.ft_max_epoch}.yaml"
            write_yaml(ft_cfg, ft_path)
            ft_entries[split] = rel_to_pointgpt(ft_path, pointgpt)

        manifest["runs"].append({
            "run_id": run_id,
            "variant": variant["name"],
            "order": order,
            "mask_ratio": mask,
            "group_mode": group,
            "skip_k": skip_k,
            "pretrain_position_mode": pos,
            "center_aux_weight": aux,
            "readout_position_mode": readout_pos,
            "pretrain_config": rel_to_pointgpt(pre_path, pointgpt),
            "pretrain_exp": f"posetmo_pre_{run_id}_e{args.max_epoch}",
            "finetune_configs": ft_entries,
            "finetune_exp_prefix": f"posetmo_ft_{run_id}",
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote configs to {out_dir}")
    print(f"[done] wrote manifest to {manifest_path}")
    print(f"[matrix] {len(manifest['runs'])} diagnostic variants")


if __name__ == "__main__":
    main()
