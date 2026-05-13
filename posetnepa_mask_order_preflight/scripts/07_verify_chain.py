#!/usr/bin/env python3
"""Verify generated pre-flight configs and local launcher prerequisites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import yaml
except Exception as e:
    raise SystemExit("[error] PyYAML is required for config verification") from e


def load_yaml(path: Path):
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--pointgpt-dir", default="PointGPT")
    ap.add_argument("--manifest", default="posetnepa_mask_order_preflight/generated/manifest.json")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    pointgpt = repo / args.pointgpt_dir
    manifest_path = repo / args.manifest
    errors: list[str] = []

    for rel in [
        "pointnepa/scripts/local/pointgpt_train_local_ddp.sh",
        "pointnepa/scripts/local/pointgpt_finetune_local_ddp.sh",
        "PointGPT/models/PointGPT.py",
    ]:
        if not (repo / rel).exists():
            errors.append(f"missing required file: {rel}")
    if not manifest_path.exists():
        errors.append(f"missing manifest: {manifest_path}")

    if not errors and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest.get("runs", []):
            for key in ["pretrain_config", *[f"finetune:{s}" for s in entry.get("finetune_configs", {})]]:
                if key.startswith("finetune:"):
                    split = key.split(":", 1)[1]
                    cfg_rel = entry["finetune_configs"][split]
                    expected_mask = None
                else:
                    cfg_rel = entry["pretrain_config"]
                    expected_mask = float(entry["mask_ratio"])
                cfg_path = pointgpt / cfg_rel
                if not cfg_path.exists():
                    errors.append(f"{entry['run_id']}: missing config: {cfg_rel}")
                    continue
                cfg = load_yaml(cfg_path)
                model = cfg.get("model", {})
                if model.get("order_mode") != entry["order"]:
                    errors.append(f"{cfg_rel}: order_mode mismatch")
                if model.get("group_mode") != entry["group_mode"]:
                    errors.append(f"{cfg_rel}: group_mode mismatch")
                if expected_mask is not None:
                    mask = model.get("transformer_config", {}).get("mask_ratio")
                    if float(mask) != expected_mask:
                        errors.append(f"{cfg_rel}: mask_ratio mismatch {mask} != {expected_mask}")

    if errors:
        print("[chain-verify] FAILED")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("[chain-verify] OK")


if __name__ == "__main__":
    main()
