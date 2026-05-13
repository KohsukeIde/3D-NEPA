#!/usr/bin/env python3
"""Build a pretrain-order x finetune-order mismatch matrix.

This generator reuses completed Stage 1 mask-off pretrain checkpoints and writes
new fine-tune configs whose order_mode can intentionally differ from the
pretraining order. It only prepares configs/manifests; training is handled by
11_run_order_mismatch_finetune.sh.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit("[error] PyYAML is required. Install with: pip install pyyaml") from exc


SPLIT_ALIASES = {
    "obj_bg": "objbg",
    "objbg": "objbg",
    "obj_only": "objonly",
    "objonly": "objonly",
    "hardest": "hardest",
    "pb_t50_rs": "hardest",
    "pb": "hardest",
}


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_splits(raw: str) -> list[str]:
    splits: list[str] = []
    for item in parse_csv(raw):
        key = item.lower()
        if key not in SPLIT_ALIASES:
            raise SystemExit(f"[error] unsupported split: {item}")
        split = SPLIT_ALIASES[key]
        if split not in splits:
            splits.append(split)
    return splits


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def write_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def set_ft_common(cfg: dict, order: str, group: str, max_epoch: int | None, total_bs: int | None) -> dict:
    cfg = copy.deepcopy(cfg)
    model = cfg.setdefault("model", {})
    model["order_mode"] = str(order)
    model["group_mode"] = str(group)
    if max_epoch is not None:
        cfg["max_epoch"] = int(max_epoch)
        if isinstance(cfg.get("scheduler"), dict) and isinstance(cfg["scheduler"].get("kwargs"), dict):
            cfg["scheduler"]["kwargs"]["epochs"] = int(max_epoch)
    if total_bs is not None:
        cfg["total_bs"] = int(total_bs)
    return cfg


def rel_to_pointgpt(path: Path, pointgpt: Path) -> str:
    return str(path.resolve().relative_to(pointgpt.resolve()))


def safe_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def resolve_manifest(repo: Path, raw: str) -> Path:
    requested = Path(raw)
    path = requested if requested.is_absolute() else repo / requested
    if path.exists():
        return path
    generated = repo / "posetnepa_mask_order_preflight" / "generated"
    candidates = sorted(
        generated.glob("stage1_*/manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"[info] stage manifest not found at {path}; using latest Stage 1 manifest: {candidates[0]}")
        return candidates[0]
    raise SystemExit(f"[error] stage manifest not found: {path}")


def exp_root(pointgpt: Path, cfg_rel: str) -> Path:
    cfg = Path(cfg_rel)
    return pointgpt / "experiments" / cfg.stem / cfg.parent.name


def find_stage_ckpt(pointgpt: Path, entry: dict, stage_run_tag: str) -> Path | None:
    root = exp_root(pointgpt, entry["pretrain_config"])
    if not root.exists():
        return None
    candidates: list[Path] = []
    if stage_run_tag:
        candidates.append(root / f"{entry['pretrain_exp']}_{stage_run_tag}" / "ckpt-last.pth")
        candidates += list(root.glob(entry["pretrain_exp"] + f"*{stage_run_tag}*/ckpt-last.pth"))
    else:
        candidates += list(root.glob(entry["pretrain_exp"] + "*/ckpt-last.pth"))
    candidates = sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    default_tag = f"order_mismatch_{time.strftime('%Y%m%d_%H%M%S')}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--stage1-manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--stage1-run-tag", default="", help="Defaults to the config parent in the Stage 1 manifest.")
    ap.add_argument("--run-tag", default=default_tag)
    ap.add_argument("--source-mask", type=float, default=0.0)
    ap.add_argument("--pretrain-orders", default="", help="Comma subset of Stage 1 pretrain orders; default all selected mask rows.")
    ap.add_argument("--finetune-orders", default="", help="Comma finetune orders; default equals selected pretrain orders.")
    ap.add_argument("--splits", default="hardest")
    ap.add_argument("--group-mode", default="", help="Override group_mode; default uses each source row group_mode.")
    ap.add_argument("--base-ft-objbg", default="cfgs/PointGPT-S/finetune_scan_objbg.yaml")
    ap.add_argument("--base-ft-objonly", default="cfgs/PointGPT-S/finetune_scan_objonly.yaml")
    ap.add_argument("--base-ft-hardest", default="cfgs/PointGPT-S/finetune_scan_hardest.yaml")
    ap.add_argument("--ft-max-epoch", type=int, default=50)
    ap.add_argument("--ft-total-bs", type=int, default=None)
    ap.add_argument("--out-dir", default="", help="PointGPT-relative or repo-relative config dir.")
    ap.add_argument("--manifest", default="", help="Output JSON manifest path.")
    ap.add_argument("--out-md", default="", help="Optional markdown plan path.")
    ap.add_argument("--allow-missing-ckpt", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Validate and print the matrix without writing files.")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = (repo / args.pointgpt_dir).resolve()
    stage_manifest_path = resolve_manifest(repo, args.stage1_manifest)
    stage_manifest = json.loads(stage_manifest_path.read_text())
    stage_run_tag = args.stage1_run_tag or Path(stage_manifest["runs"][0]["pretrain_config"]).parent.name
    splits = parse_splits(args.splits)
    pretrain_filter = set(parse_csv(args.pretrain_orders))

    selected = [
        entry for entry in stage_manifest["runs"]
        if abs(float(entry["mask_ratio"]) - float(args.source_mask)) < 1e-9
        and (not pretrain_filter or entry["order"] in pretrain_filter)
    ]
    if not selected:
        raise SystemExit("[error] no Stage 1 source rows selected")

    finetune_orders = parse_csv(args.finetune_orders) or sorted({entry["order"] for entry in selected})
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/{args.run_tag}")
    out_dir = out_dir if out_dir.is_absolute() else repo / out_dir
    manifest_path = Path(args.manifest) if args.manifest else repo / "posetnepa_mask_order_preflight" / "generated" / args.run_tag / "order_mismatch_manifest.json"
    manifest_path = manifest_path if manifest_path.is_absolute() else repo / manifest_path
    out_md = Path(args.out_md) if args.out_md else manifest_path.with_suffix(".md")
    out_md = out_md if out_md.is_absolute() else repo / out_md

    base_fts = {
        "objbg": load_yaml(pointgpt / args.base_ft_objbg),
        "objonly": load_yaml(pointgpt / args.base_ft_objonly),
        "hardest": load_yaml(pointgpt / args.base_ft_hardest),
    }

    rows: list[dict] = []
    missing: list[str] = []
    for entry in selected:
        ckpt = find_stage_ckpt(pointgpt, entry, stage_run_tag)
        if ckpt is None:
            missing.append(entry["run_id"])
            if not args.allow_missing_ckpt:
                continue
        for ft_order in finetune_orders:
            for split in splits:
                group = args.group_mode or entry["group_mode"]
                pre_order = safe_name(entry["order"])
                ft_order_safe = safe_name(ft_order)
                mtag = f"m{safe_float(float(entry['mask_ratio']))}"
                row_id = f"pre_{pre_order}_ft_{ft_order_safe}_{split}_{mtag}"
                ft_cfg = set_ft_common(base_fts[split], ft_order, group, args.ft_max_epoch, args.ft_total_bs)
                ft_path = out_dir / f"finetune_scan_{split}_{row_id}_e{args.ft_max_epoch}.yaml"
                ckpt_run = ckpt.parent.name if ckpt is not None else f"{entry['pretrain_exp']}_{stage_run_tag}"
                rows.append({
                    "row_id": row_id,
                    "split": split,
                    "pretrain_run_id": entry["run_id"],
                    "pretrain_order": entry["order"],
                    "pretrain_mask_ratio": entry["mask_ratio"],
                    "finetune_order": ft_order,
                    "group_mode": group,
                    "stage1_run_tag": stage_run_tag,
                    "ckpt_path": str(ckpt or ""),
                    "ckpt_exists": ckpt is not None,
                    "finetune_config": rel_to_pointgpt(ft_path, pointgpt),
                    "finetune_exp": f"posetmo_mismatch_{row_id}_from_{ckpt_run}_{args.run_tag}",
                })
                if not args.dry_run:
                    write_yaml(ft_cfg, ft_path)

    if missing and not args.allow_missing_ckpt:
        raise SystemExit("[error] missing Stage 1 ckpt(s): " + ", ".join(missing))

    manifest = {
        "purpose": "pretrain-order x finetune-order mismatch matrix",
        "source_stage1_manifest": str(stage_manifest_path),
        "source_mask": args.source_mask,
        "stage1_run_tag": stage_run_tag,
        "run_tag": args.run_tag,
        "splits": splits,
        "finetune_orders": finetune_orders,
        "ft_max_epoch": args.ft_max_epoch,
        "rows": rows,
    }

    if args.dry_run:
        print(f"[dry-run] selected_sources={len(selected)} finetune_orders={len(finetune_orders)} splits={len(splits)} rows={len(rows)}")
        for row in rows:
            print(
                f"[row] {row['row_id']} ckpt_exists={row['ckpt_exists']} "
                f"config={row['finetune_config']} exp={row['finetune_exp']}"
            )
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with out_md.open("w") as f:
        f.write("# Order Mismatch Matrix\n\n")
        f.write(f"Source Stage 1 manifest: `{stage_manifest_path}`\n\n")
        f.write("| row | pretrain order | finetune order | split | ckpt exists | config |\n")
        f.write("|---|---|---|---|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['row_id']} | {row['pretrain_order']} | {row['finetune_order']} | "
                f"{row['split']} | {row['ckpt_exists']} | `{row['finetune_config']}` |\n"
            )
    print(f"[done] wrote configs to {out_dir}")
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {out_md}")
    print(f"[matrix] {len(selected)} pretrain orders x {len(finetune_orders)} finetune orders x {len(splits)} splits = {len(rows)} jobs")


if __name__ == "__main__":
    main()
