#!/usr/bin/env python3
"""Prepare scratch and early-pretrain comparison jobs.

The generated plan has two phases:
  1. optional short pretrain jobs for early checkpoints, e.g. 1/5/10 epochs;
  2. ScanObjectNN fine-tune jobs from scratch, early checkpoints, and the Stage
     1 last checkpoint.

Training is handled by 14_run_scratch_early_comparison.sh. This generator is
safe to run with --dry-run to inspect the plan without writing configs.
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


def parse_ints(raw: str) -> list[int]:
    vals: list[int] = []
    for item in parse_csv(raw):
        value = int(item)
        if value < 0:
            raise SystemExit(f"[error] epoch must be non-negative: {item}")
        if value not in vals:
            vals.append(value)
    return vals


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


def set_common(cfg: dict, order: str, group: str, max_epoch: int | None, total_bs: int | None, mask: float | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    model = cfg.setdefault("model", {})
    model["order_mode"] = str(order)
    model["group_mode"] = str(group)
    if mask is not None:
        tcfg = model.setdefault("transformer_config", {})
        tcfg["mask_ratio"] = float(mask)
        if "mask_ratio" in cfg:
            cfg["mask_ratio"] = float(mask)
    if max_epoch is not None:
        cfg["max_epoch"] = int(max_epoch)
        if isinstance(cfg.get("scheduler"), dict) and isinstance(cfg["scheduler"].get("kwargs"), dict):
            cfg["scheduler"]["kwargs"]["epochs"] = int(max_epoch)
    if total_bs is not None:
        cfg["total_bs"] = int(total_bs)
    return cfg


def safe_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def rel_to_pointgpt(path: Path, pointgpt: Path) -> str:
    return str(path.resolve().relative_to(pointgpt.resolve()))


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
    candidates = [
        root / f"{entry['pretrain_exp']}_{stage_run_tag}" / "ckpt-last.pth",
        *root.glob(entry["pretrain_exp"] + f"*{stage_run_tag}*/ckpt-last.pth"),
    ]
    candidates = sorted({p for p in candidates if p.exists()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def expected_ckpt(pointgpt: Path, cfg_rel: str, exp_name: str) -> Path:
    return exp_root(pointgpt, cfg_rel) / exp_name / "ckpt-last.pth"


def main() -> None:
    default_tag = f"scratch_early_{time.strftime('%Y%m%d_%H%M%S')}"
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--stage1-manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    ap.add_argument("--stage1-run-tag", default="")
    ap.add_argument("--run-tag", default=default_tag)
    ap.add_argument("--orders", default="", help="Comma subset; default all selected Stage 1 rows.")
    ap.add_argument("--masks", default="0.0")
    ap.add_argument("--splits", default="hardest")
    ap.add_argument("--early-epochs", default="1,5,10")
    ap.add_argument("--include-scratch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-stage1-last", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--include-stage1-frozen", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--group-mode", default="")
    ap.add_argument("--base-pretrain", default="cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0.yaml")
    ap.add_argument("--base-ft-objbg", default="cfgs/PointGPT-S/finetune_scan_objbg.yaml")
    ap.add_argument("--base-ft-objonly", default="cfgs/PointGPT-S/finetune_scan_objonly.yaml")
    ap.add_argument("--base-ft-hardest", default="cfgs/PointGPT-S/finetune_scan_hardest.yaml")
    ap.add_argument("--pretrain-total-bs", type=int, default=None)
    ap.add_argument("--ft-max-epoch", type=int, default=50)
    ap.add_argument("--ft-total-bs", type=int, default=None)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = (repo / args.pointgpt_dir).resolve()
    stage_manifest_path = resolve_manifest(repo, args.stage1_manifest)
    stage_manifest = json.loads(stage_manifest_path.read_text())
    stage_run_tag = args.stage1_run_tag or Path(stage_manifest["runs"][0]["pretrain_config"]).parent.name
    order_filter = set(parse_csv(args.orders))
    mask_filter = {float(x) for x in parse_csv(args.masks)}
    splits = parse_splits(args.splits)
    early_epochs = [e for e in parse_ints(args.early_epochs) if e > 0]

    selected = [
        entry for entry in stage_manifest["runs"]
        if float(entry["mask_ratio"]) in mask_filter
        and (not order_filter or entry["order"] in order_filter)
    ]
    if not selected:
        raise SystemExit("[error] no Stage 1 rows selected")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/{args.run_tag}")
    out_dir = out_dir if out_dir.is_absolute() else repo / out_dir
    manifest_path = Path(args.manifest) if args.manifest else repo / "posetnepa_mask_order_preflight" / "generated" / args.run_tag / "scratch_early_manifest.json"
    manifest_path = manifest_path if manifest_path.is_absolute() else repo / manifest_path
    out_md = Path(args.out_md) if args.out_md else manifest_path.with_suffix(".md")
    out_md = out_md if out_md.is_absolute() else repo / out_md

    base_pre = load_yaml(pointgpt / args.base_pretrain)
    base_fts = {
        "objbg": load_yaml(pointgpt / args.base_ft_objbg),
        "objonly": load_yaml(pointgpt / args.base_ft_objonly),
        "hardest": load_yaml(pointgpt / args.base_ft_hardest),
    }

    pretrain_jobs: list[dict] = []
    finetune_jobs: list[dict] = []
    for entry in selected:
        order = entry["order"]
        safe_order = safe_name(order)
        mask = float(entry["mask_ratio"])
        mtag = f"m{safe_float(mask)}"
        group = args.group_mode or entry["group_mode"]

        early_by_epoch: dict[int, dict] = {}
        for epoch in early_epochs:
            run_id = f"{safe_order}_{mtag}_e{epoch}"
            pre_cfg = set_common(base_pre, order, group, epoch, args.pretrain_total_bs, mask=mask)
            pre_path = out_dir / f"pretrain_nepa_early_{run_id}.yaml"
            pre_cfg_rel = rel_to_pointgpt(pre_path, pointgpt)
            exp_name = f"posetmo_early_pre_{run_id}_{args.run_tag}"
            ckpt = expected_ckpt(pointgpt, pre_cfg_rel, exp_name)
            job = {
                "job_id": f"pretrain_{run_id}",
                "order": order,
                "mask_ratio": mask,
                "group_mode": group,
                "epoch": epoch,
                "pretrain_config": pre_cfg_rel,
                "exp_name": exp_name,
                "expected_ckpt": str(ckpt),
                "ckpt_exists": ckpt.exists(),
            }
            early_by_epoch[epoch] = job
            pretrain_jobs.append(job)
            if not args.dry_run:
                write_yaml(pre_cfg, pre_path)

        stage_ckpt = find_stage_ckpt(pointgpt, entry, stage_run_tag)
        stage_epoch = int(stage_manifest.get("max_epoch", 0))

        for split in splits:
            ft_cfg = set_common(base_fts[split], order, group, args.ft_max_epoch, args.ft_total_bs, mask=None)

            if args.include_scratch:
                scratch_id = f"scratch_{safe_order}_{split}_{mtag}"
                scratch_path = out_dir / f"finetune_scan_{split}_{scratch_id}_e{args.ft_max_epoch}.yaml"
                finetune_jobs.append({
                    "job_id": scratch_id,
                    "init_kind": "scratch",
                    "freeze_backbone": False,
                    "source_epoch": 0,
                    "order": order,
                    "finetune_order": order,
                    "mask_ratio": mask,
                    "group_mode": group,
                    "split": split,
                    "finetune_config": rel_to_pointgpt(scratch_path, pointgpt),
                    "exp_name": f"posetmo_{scratch_id}_{args.run_tag}",
                    "ckpt_path": "",
                    "ckpt_exists": False,
                    "needs_pretrain_job": "",
                })
                if not args.dry_run:
                    write_yaml(ft_cfg, scratch_path)

            for epoch, pre_job in early_by_epoch.items():
                early_id = f"earlye{epoch}_{safe_order}_{split}_{mtag}"
                early_path = out_dir / f"finetune_scan_{split}_{early_id}_e{args.ft_max_epoch}.yaml"
                finetune_jobs.append({
                    "job_id": early_id,
                    "init_kind": "early_pretrain",
                    "freeze_backbone": False,
                    "source_epoch": epoch,
                    "order": order,
                    "finetune_order": order,
                    "mask_ratio": mask,
                    "group_mode": group,
                    "split": split,
                    "finetune_config": rel_to_pointgpt(early_path, pointgpt),
                    "exp_name": f"posetmo_{early_id}_{args.run_tag}",
                    "ckpt_path": pre_job["expected_ckpt"],
                    "ckpt_exists": pre_job["ckpt_exists"],
                    "needs_pretrain_job": pre_job["job_id"],
                })
                if not args.dry_run:
                    write_yaml(ft_cfg, early_path)

            if args.include_stage1_last:
                last_id = f"stage1last_e{stage_epoch}_{safe_order}_{split}_{mtag}"
                last_path = out_dir / f"finetune_scan_{split}_{last_id}_e{args.ft_max_epoch}.yaml"
                finetune_jobs.append({
                    "job_id": last_id,
                    "init_kind": "stage1_last",
                    "freeze_backbone": False,
                    "source_epoch": stage_epoch,
                    "order": order,
                    "finetune_order": order,
                    "mask_ratio": mask,
                    "group_mode": group,
                    "split": split,
                    "finetune_config": rel_to_pointgpt(last_path, pointgpt),
                    "exp_name": f"posetmo_{last_id}_{args.run_tag}",
                    "ckpt_path": str(stage_ckpt or ""),
                    "ckpt_exists": stage_ckpt is not None,
                    "needs_pretrain_job": "",
                })
                if not args.dry_run:
                    write_yaml(ft_cfg, last_path)

            if args.include_stage1_frozen:
                frozen_id = f"stage1frozen_e{stage_epoch}_{safe_order}_{split}_{mtag}"
                frozen_path = out_dir / f"finetune_scan_{split}_{frozen_id}_e{args.ft_max_epoch}.yaml"
                finetune_jobs.append({
                    "job_id": frozen_id,
                    "init_kind": "stage1_frozen",
                    "freeze_backbone": True,
                    "source_epoch": stage_epoch,
                    "order": order,
                    "finetune_order": order,
                    "mask_ratio": mask,
                    "group_mode": group,
                    "split": split,
                    "finetune_config": rel_to_pointgpt(frozen_path, pointgpt),
                    "exp_name": f"posetmo_{frozen_id}_{args.run_tag}",
                    "ckpt_path": str(stage_ckpt or ""),
                    "ckpt_exists": stage_ckpt is not None,
                    "needs_pretrain_job": "",
                })
                if not args.dry_run:
                    write_yaml(ft_cfg, frozen_path)

    manifest = {
        "purpose": "scratch versus early-pretrain comparison",
        "source_stage1_manifest": str(stage_manifest_path),
        "stage1_run_tag": stage_run_tag,
        "run_tag": args.run_tag,
        "splits": splits,
        "early_epochs": early_epochs,
        "ft_max_epoch": args.ft_max_epoch,
        "pretrain_jobs": pretrain_jobs,
        "finetune_jobs": finetune_jobs,
    }

    if args.dry_run:
        print(f"[dry-run] selected_sources={len(selected)} pretrain_jobs={len(pretrain_jobs)} finetune_jobs={len(finetune_jobs)}")
        for job in finetune_jobs:
            print(
                f"[job] {job['job_id']} init={job['init_kind']} epoch={job['source_epoch']} "
                f"ckpt_exists={job['ckpt_exists']} config={job['finetune_config']}"
            )
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with out_md.open("w") as f:
        f.write("# Scratch/Early-Epoch Comparison\n\n")
        f.write(f"Source Stage 1 manifest: `{stage_manifest_path}`\n\n")
        f.write("## Pretrain Jobs\n\n")
        f.write("| job | order | mask | epoch | ckpt exists | config |\n")
        f.write("|---|---|---:|---:|---:|---|\n")
        for job in pretrain_jobs:
            f.write(
                f"| {job['job_id']} | {job['order']} | {job['mask_ratio']} | "
                f"{job['epoch']} | {job['ckpt_exists']} | `{job['pretrain_config']}` |\n"
            )
        f.write("\n## Fine-Tune Jobs\n\n")
        f.write("| job | init | order | mask | split | source epoch | frozen | ckpt exists | config |\n")
        f.write("|---|---|---|---:|---|---:|---:|---:|---|\n")
        for job in finetune_jobs:
            f.write(
                f"| {job['job_id']} | {job['init_kind']} | {job['order']} | {job['mask_ratio']} | "
                f"{job['split']} | {job['source_epoch']} | {job.get('freeze_backbone', False)} | "
                f"{job['ckpt_exists']} | `{job['finetune_config']}` |\n"
            )
    print(f"[done] wrote configs to {out_dir}")
    print(f"[done] wrote {manifest_path}")
    print(f"[done] wrote {out_md}")
    print(f"[plan] {len(pretrain_jobs)} pretrain jobs, {len(finetune_jobs)} fine-tune jobs")


if __name__ == "__main__":
    main()
