#!/usr/bin/env python
"""Orchestrate Point-MAE / PCP-MAE object-side diagnostics on 4 GPUs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path
from typing import Any

import yaml

from object_ssl_common import (
    PART_CONDITIONS,
    SCAN_CONDITIONS,
    ensure_dir,
    file_sha256,
    git_commit,
    markdown_table,
    read_json,
    repo_root_from_script,
    write_csv,
    write_json,
    write_text,
)


POINTMAE_URLS = {
    "pretrain": ("pretrain.pth", "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth"),
    # The official release filenames are misleading in the local artifacts:
    # scan_objonly.pth stores a 90.0172 checkpoint, while scan_objbg.pth stores
    # an 88.2960 checkpoint. Use the checkpoint metrics to map them to splits.
    "obj_bg": ("scan_objonly.pth", "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth"),
    "obj_only": ("scan_objbg.pth", "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth"),
    "pb_t50_rs": ("scan_hardest.pth", "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth"),
    "part": ("part_seg.pth", "https://github.com/Pang-Yatian/Point-MAE/releases/download/main/part_seg.pth"),
}

PCPMAE_FOLDERS = {
    "pretrain": "https://drive.google.com/drive/folders/1smQMWBBEdMOXVAzIBs3xCBrcyQDg8_GS?usp=drive_link",
    "obj_bg": "https://drive.google.com/drive/folders/1He3bUfXJ36nwAcGbQE4I9tOUnxjEmfae?usp=drive_link",
    "obj_only": "https://drive.google.com/drive/folders/1xuJlAwSYMwc0bTKvnzaoePggMrLqQw3r?usp=drive_link",
    "pb_t50_rs": "https://drive.google.com/drive/folders/1YWJrThywU6G4yoUn4-GvtnHH_bi_Uprp?usp=drive_link",
}

VARIANTS = ["obj_bg", "obj_only", "pb_t50_rs"]
CONFIG_BY_MODEL = {
    "pointmae": {
        "obj_bg": "cfgs/finetune_scan_objbg.yaml",
        "obj_only": "cfgs/finetune_scan_objonly.yaml",
        "pb_t50_rs": "cfgs/finetune_scan_hardest.yaml",
    },
    "pcpmae": {
        "obj_bg": "cfgs/finetune_scan_objbg.yaml",
        "obj_only": "cfgs/finetune_scan_objonly.yaml",
        "pb_t50_rs": "cfgs/finetune_scan_hardest.yaml",
    },
}

DATASET_NAME = {
    "obj_bg": "ScanObjectNN",
    "obj_only": "ScanObjectNN",
    "pb_t50_rs": "ScanObjectNN_hardest",
}


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    p = argparse.ArgumentParser("Run the full Point-MAE / PCP-MAE object SSL chain")
    p.add_argument("--root", default=str(root))
    p.add_argument("--result-dir", default=str(root / "results" / "object_ssl_pointmae_pcpmae"))
    p.add_argument("--log-dir", default=str(root / "logs" / "object_ssl_pointmae_pcpmae"))
    p.add_argument("--stage", default="all", choices=["all", "audit", "download", "eval", "pcp_seg_ft", "heldout", "summary"])
    p.add_argument("--gpus", default=os.environ.get("CHAIN_GPUS", "0,1,2,3"))
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--heldout-frac", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--skip-pcp-seg-ft", action="store_true")
    p.add_argument("--skip-heldout", action="store_true")
    p.add_argument("--heldout-models", default="pointmae,pcpmae", help="comma-separated subset for heldout stage")
    p.add_argument("--heldout-variants", default=",".join(VARIANTS), help="comma-separated subset for heldout stage")
    p.add_argument("--max-batches", type=int, default=0, help="debug limit; 0 means full eval")
    p.add_argument("--pointmae-python", default="")
    p.add_argument("--pcpmae-python", default="")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size-scan", type=int, default=32)
    p.add_argument("--batch-size-part", type=int, default=16)
    return p.parse_args()


def resolve_python(root: Path, requested: str, default_candidates: list[Path]) -> str:
    if requested:
        return requested
    for cand in default_candidates:
        if cand.is_file():
            return str(cand)
    return sys.executable


class Chain:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = Path(args.root).resolve()
        self.result_dir = ensure_dir(args.result_dir)
        self.log_dir = ensure_dir(args.log_dir)
        self.raw_dir = ensure_dir(self.result_dir / "raw")
        self.ckpt_dir = ensure_dir(self.result_dir / "checkpoints")
        self.config_dir = ensure_dir(self.result_dir / "configs")
        self.work_dir = ensure_dir(self.result_dir / "work")
        self.split_dir = ensure_dir(self.root / "splits")
        self.docs_dir = ensure_dir(self.root / "docs")
        self.pointmae_root = self.root / "Point-MAE"
        self.pcpmae_root = self.root / "PCP-MAE"
        self.pointmae_python = resolve_python(
            self.root,
            args.pointmae_python,
            [self.root / ".venv" / "bin" / "python", Path("/home/minesawa/anaconda3/envs/geopcp-pcpmae-cu118/bin/python")],
        )
        self.pcpmae_python = resolve_python(
            self.root,
            args.pcpmae_python,
            [Path("/home/minesawa/anaconda3/envs/geopcp-pcpmae-cu118/bin/python"), self.root / ".venv" / "bin" / "python"],
        )
        self.gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
        self.git_commit = git_commit(self.root)

    def run(self) -> None:
        stage = self.args.stage
        if self.args.dry_run:
            print("[dry-run] audit/summary files will not be written")
        if stage in {"all", "audit"}:
            if not self.args.dry_run:
                self.write_audit("initial")
        if stage in {"all", "download"} and not self.args.skip_download:
            self.download_checkpoints()
        if stage in {"all", "eval"}:
            self.run_official_eval_stage()
        if stage in {"all", "pcp_seg_ft"} and not self.args.skip_pcp_seg_ft:
            self.run_pcp_seg_ft_stage()
            self.run_pcp_seg_eval_stage()
        if stage in {"all", "heldout"} and not self.args.skip_heldout:
            self.run_heldout_stage()
        if stage in {"all", "summary"}:
            if not self.args.dry_run:
                if not list((self.result_dir / "raw_grouping").glob("*.json")):
                    self.write_grouping_blocked()
                self.aggregate()
                self.write_audit("final")

    def checkpoint_path(self, model: str, key: str) -> Path:
        if model == "pointmae":
            filename = POINTMAE_URLS[key][0]
            return self.ckpt_dir / "pointmae" / filename
        if model == "pcpmae":
            manifest = self.manifest_path()
            if manifest.is_file():
                data = read_json(manifest)
                p = data.get("pcpmae", {}).get(key, {}).get("path")
                if p:
                    return Path(p)
            return self.ckpt_dir / "pcpmae" / key / "UNRESOLVED.pth"
        raise ValueError(model)

    def manifest_path(self) -> Path:
        return self.result_dir / "checkpoint_manifest.json"

    def download_checkpoints(self) -> None:
        manifest: dict[str, Any] = {"pointmae": {}, "pcpmae": {}}
        for key, (filename, url) in POINTMAE_URLS.items():
            out = self.ckpt_dir / "pointmae" / filename
            if not out.is_file() or self.args.force:
                self.download_url(url, out)
            manifest["pointmae"][key] = {
                "path": str(out.resolve()),
                "url": url,
                "sha256": file_sha256(out),
            }
        for key, folder in PCPMAE_FOLDERS.items():
            out_dir = ensure_dir(self.ckpt_dir / "pcpmae" / key)
            if (not any(out_dir.rglob("*.pth"))) or self.args.force:
                self.download_gdrive_folder(folder, out_dir, self.pcpmae_python)
            selected = self.pick_pcp_checkpoint(out_dir, key)
            manifest["pcpmae"][key] = {
                "path": str(selected.resolve()) if selected else "",
                "url": folder,
                "sha256": file_sha256(selected) if selected else "",
            }
        write_json(self.manifest_path(), manifest)

    def download_url(self, url: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.args.dry_run:
            print(f"[dry-run] download {url} -> {out}")
            return
        print(f"[download] {url} -> {out}")
        tmp = out.with_suffix(out.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(out)

    def download_gdrive_folder(self, folder_url: str, out_dir: Path, python_bin: str) -> None:
        if self.args.dry_run:
            print(f"[dry-run] gdown folder {folder_url} -> {out_dir}")
            return
        print(f"[download] Google Drive folder {folder_url} -> {out_dir}")
        code = "import gdown"
        probe = subprocess.run([python_bin, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode != 0:
            subprocess.run([python_bin, "-m", "pip", "install", "-q", "gdown"], check=True)
        subprocess.run([python_bin, "-m", "gdown", "--folder", folder_url, "-O", str(out_dir)], check=True)

    def pick_pcp_checkpoint(self, out_dir: Path, key: str) -> Path | None:
        files = sorted(out_dir.rglob("*.pth"))
        if not files:
            return None
        if key == "pretrain":
            prefs = ["ckpt-300", "ckpt_300", "epoch-300", "ckpt-275", "ckpt"]
        else:
            prefs = ["ckpt-best", "best", "model_best", "checkpoint"]
        for pref in prefs:
            for f in files:
                if pref.lower() in f.name.lower():
                    return f
        return files[0]

    def run_official_eval_stage(self) -> None:
        tasks = []
        for model in ["pointmae", "pcpmae"]:
            for variant in VARIANTS:
                ckpt = self.checkpoint_path(model, variant)
                cfg = self.config_path(model, variant)
                out = self.raw_dir / "scanobjectnn" / f"{model}_{variant}_official.json"
                tasks.append(
                    {
                        "name": f"scan_{model}_{variant}",
                        "model": model,
                        "gpu": None,
                        "cmd": [
                            self.python_for_model(model),
                            str(self.root / "scripts" / "object_ssl" / "eval_scanobjectnn_mae.py"),
                            "--model",
                            model,
                            "--repo-root",
                            str(self.repo_for_model(model)),
                            "--config",
                            str(cfg),
                            "--checkpoint",
                            str(ckpt),
                            "--variant",
                            variant,
                            "--batch-size",
                            str(self.args.batch_size_scan),
                            "--num-workers",
                            str(self.args.num_workers),
                            "--seed",
                            str(self.args.seed),
                            "--max-batches",
                            str(self.args.max_batches),
                            "--selection-protocol",
                            "official_checkpoint",
                            "--output-json",
                            str(out),
                        ],
                        "out": out,
                    }
                )
        point_part = self.checkpoint_path("pointmae", "part")
        part_root = self.find_shapenetpart_root()
        if part_root:
            tasks.append(self.part_eval_task("pointmae", point_part, part_root, "official_checkpoint", "pointmae_part_official"))
        else:
            self.write_blocker("ShapeNetPart root unavailable; Point-MAE ShapeNetPart official eval blocked.")
        self.run_packed(tasks)

    def run_pcp_seg_ft_stage(self) -> None:
        part_ckpt = self.result_dir / "checkpoints" / "pcpmae" / "part_seg_from_pretrain" / "best_model.pth"
        if part_ckpt.is_file() and not self.args.force:
            print(f"[skip] PCP-MAE ShapeNetPart FT exists: {part_ckpt}")
            return
        pretrain = self.checkpoint_path("pcpmae", "pretrain")
        part_root = self.find_shapenetpart_root()
        if not part_root:
            self.write_blocker("ShapeNetPart root unavailable; PCP-MAE ShapeNetPart FT blocked.")
            return
        log_tag = "pcpmae_public_pretrain_shapenetpart_seed2026"
        seg_out_dir = self.result_dir / "checkpoints" / "pcpmae" / "part_seg" / log_tag
        if seg_out_dir.exists() and not self.args.force:
            log_tag = f"{log_tag}_rerun_{time.strftime('%Y%m%d_%H%M%S')}"
        seg_gpus = self.gpus[0] if self.gpus else "0"
        env = self.base_env("pcpmae", gpu=seg_gpus)
        env["PCPMAE_SEG_LOG_ROOT"] = str(self.result_dir / "checkpoints" / "pcpmae")
        env["USE_WANDB"] = env.get("USE_WANDB", "0")
        cmd = [
            self.pcpmae_python,
            str(self.pcpmae_root / "segmentation" / "main.py"),
            "--gpu",
            seg_gpus,
            "--ckpts",
            str(pretrain),
            "--log_dir",
            log_tag,
            "--learning_rate",
            "0.0002",
            "--epoch",
            "300",
            "--batch_size",
            "16",
            "--root",
            str(part_root),
            "--seed",
            str(self.args.seed),
        ]
        self.run_cmd("pcpmae_shapenetpart_ft", cmd, env=env, cwd=self.pcpmae_root / "segmentation")
        produced = self.result_dir / "checkpoints" / "pcpmae" / "part_seg" / log_tag / "checkpoints" / "best_model.pth"
        if produced.is_file():
            part_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, part_ckpt)
        else:
            self.write_blocker(f"PCP-MAE ShapeNetPart FT finished but best_model.pth was not found at {produced}")

    def run_pcp_seg_eval_stage(self) -> None:
        part_root = self.find_shapenetpart_root()
        part_ckpt = self.result_dir / "checkpoints" / "pcpmae" / "part_seg_from_pretrain" / "best_model.pth"
        if part_root and part_ckpt.is_file():
            self.run_packed([self.part_eval_task("pcpmae", part_ckpt, part_root, "finetuned_from_public_pretrain", "pcpmae_part_ft")])
        elif part_root:
            self.write_blocker("PCP-MAE ShapeNetPart checkpoint unavailable after FT; eval blocked.")

    def run_heldout_stage(self) -> None:
        split_json = self.split_dir / "scanobjectnn_trainval_seed2026.json"
        split_root = self.result_dir / "splits" / "scanobjectnn_trainval_seed2026"
        variants = [x.strip() for x in self.args.heldout_variants.split(",") if x.strip()]
        models = [x.strip() for x in self.args.heldout_models.split(",") if x.strip()]
        unknown_variants = sorted(set(variants) - set(VARIANTS))
        unknown_models = sorted(set(models) - {"pointmae", "pcpmae"})
        if unknown_variants or unknown_models:
            raise ValueError(f"unknown heldout filters: models={unknown_models}, variants={unknown_variants}")
        for variant in variants:
            cmd = [
                sys.executable,
                str(self.root / "scripts" / "object_ssl" / "make_scanobjectnn_heldout_split.py"),
                "--variant",
                variant,
                "--out-root",
                str(split_root),
                "--split-json",
                str(split_json),
                "--seed",
                str(self.args.seed),
                "--heldout-frac",
                str(self.args.heldout_frac),
            ]
            self.run_cmd(f"make_split_{variant}", cmd)
        for model in models:
            for variant in variants:
                self.run_one_heldout_ft(model, variant, split_root / variant)

    def run_one_heldout_ft(self, model: str, variant: str, split_root: Path) -> None:
        cfg = self.make_heldout_config(model, variant, split_root)
        exp_name = f"{model}_{variant}_heldout_seed{self.args.seed}"
        ckpt = self.checkpoint_path(model, "pretrain")
        work = ensure_dir(self.work_dir / f"{model}_heldout")
        python_bin = self.python_for_model(model)
        repo = self.repo_for_model(model)
        env = self.base_env(model, gpu=",".join(self.gpus))
        env["PYTHONPATH"] = f"{repo}:{self.root}{os.pathsep}{env.get('PYTHONPATH', '')}"
        if model == "pcpmae":
            env["PCPMAE_EXPERIMENTS_ROOT"] = str(self.result_dir / "heldout_experiments" / "pcpmae")
        cmd = [
            python_bin,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(len(self.gpus)),
            "--master_port",
            str(self.heldout_master_port(model, variant)),
            str(repo / "main.py"),
            "--launcher",
            "pytorch",
            "--config",
            str(cfg),
            "--exp_name",
            exp_name,
            "--ckpts",
            str(ckpt),
            "--finetune_model",
            "--seed",
            str(self.args.seed),
            "--val_freq",
            "1",
            "--num_workers",
            "8",
        ]
        marker = self.result_dir / ".markers" / f"heldout_{model}_{variant}.done"
        if marker.is_file() and not self.args.force:
            print(f"[skip] heldout FT {model} {variant}")
        else:
            self.run_cmd(f"heldout_{model}_{variant}", cmd, env=env, cwd=work, python_path=python_bin)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        ckpt_best = self.find_heldout_best(model, cfg, exp_name, work)
        if ckpt_best:
            out = self.raw_dir / "scanobjectnn" / f"{model}_{variant}_heldout.json"
            eval_cmd = [
                python_bin,
                str(self.root / "scripts" / "object_ssl" / "eval_scanobjectnn_mae.py"),
                "--model",
                model,
                "--repo-root",
                str(repo),
                "--config",
                str(cfg),
                "--checkpoint",
                str(ckpt_best),
                "--variant",
                variant,
                "--batch-size",
                str(self.args.batch_size_scan),
                "--seed",
                str(self.args.seed),
                "--max-batches",
                str(self.args.max_batches),
                "--selection-protocol",
                "heldout_selection",
                "--output-json",
                str(out),
            ]
            self.run_cmd(f"eval_heldout_{model}_{variant}", eval_cmd, env=self.base_env(model, gpu=self.gpus[0]))
        else:
            self.write_blocker(f"Held-out best checkpoint not found for {model} {variant}.")

    def heldout_master_port(self, model: str, variant: str) -> int:
        model_offset = {"pointmae": 0, "pcpmae": 100}[model]
        variant_offset = {"obj_bg": 1, "obj_only": 2, "pb_t50_rs": 3}[variant]
        return 29600 + model_offset + variant_offset

    def make_heldout_config(self, model: str, variant: str, split_root: Path) -> Path:
        source = self.config_path(model, variant)
        cfg = yaml.safe_load(source.read_text())
        base = {
            "NAME": DATASET_NAME[variant],
            "ROOT": str(split_root.resolve()),
        }
        base_path = self.config_dir / model / f"heldout_dataset_{variant}.yaml"
        cfg_path = self.config_dir / model / f"heldout_{variant}.yaml"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(yaml.safe_dump(base, sort_keys=False))
        for section in ["train", "val", "test"]:
            cfg["dataset"][section]["_base_"] = str(base_path.resolve())
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return cfg_path

    def find_heldout_best(self, model: str, cfg: Path, exp_name: str, work: Path) -> Path | None:
        if model == "pcpmae":
            root = self.result_dir / "heldout_experiments" / "pcpmae"
        else:
            root = work / "experiments"
        candidates = sorted(root.rglob(f"*/{exp_name}/ckpt-best.pth"))
        return candidates[0] if candidates else None

    def part_eval_task(self, model: str, ckpt: Path, part_root: Path, selection: str, tag: str) -> dict[str, Any]:
        out = self.raw_dir / "shapenetpart" / f"{tag}.json"
        return {
            "name": f"part_{tag}",
            "model": model,
            "gpu": None,
            "cmd": [
                self.python_for_model(model),
                str(self.root / "scripts" / "object_ssl" / "eval_shapenetpart_mae.py"),
                "--model",
                model,
                "--repo-root",
                str(self.repo_for_model(model)),
                "--checkpoint",
                str(ckpt),
                "--data-root",
                str(part_root),
                "--batch-size",
                str(self.args.batch_size_part),
                "--seed",
                str(self.args.seed),
                "--max-batches",
                str(self.args.max_batches),
                "--selection-protocol",
                selection,
                "--output-json",
                str(out),
            ],
            "out": out,
        }

    def run_packed(self, tasks: list[dict[str, Any]]) -> None:
        if not tasks:
            return
        pending = list(tasks)
        running: dict[Future, tuple[dict[str, Any], str]] = {}
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.args.max_workers, max(1, len(self.gpus)))) as pool:
            while pending or running:
                while pending and len(running) < min(self.args.max_workers, len(self.gpus)):
                    task = pending.pop(0)
                    out = Path(task["out"])
                    if out.is_file() and not self.args.force:
                        print(f"[skip] {task['name']} -> {out}")
                        continue
                    gpu = self.gpus[len(running) % len(self.gpus)] if self.gpus else "0"
                    env = self.base_env(task.get("model", "pointmae"), gpu=gpu)
                    fut = pool.submit(self.run_cmd, task["name"], task["cmd"], env, None)
                    running[fut] = (task, gpu)
                if not running:
                    continue
                done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    task, _gpu = running.pop(fut)
                    fut.result()

    def run_cmd(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        python_path: str | None = None,
    ) -> None:
        log = self.log_dir / f"{name}.log"
        if self.args.dry_run:
            print("[dry-run]", " ".join(cmd))
            return
        env_use = os.environ.copy()
        if env:
            env_use.update(env)
        if python_path:
            env_use["PYTHON_BIN"] = python_path
        print(f"[run] {name}: {' '.join(cmd)}")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            proc = subprocess.run(cmd, cwd=str(cwd) if cwd else str(self.root), env=env_use, stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"{name} failed with code {proc.returncode}; see {log}")

    def base_env(self, model: str, gpu: str) -> dict[str, str]:
        repo = self.repo_for_model("pcpmae" if model == "pcpmae" else "pointmae")
        return {
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{repo}:{self.root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0"),
            "USE_WANDB": os.environ.get("USE_WANDB", "0"),
            "POINTNET2_ALLOW_FALLBACK": os.environ.get("POINTNET2_ALLOW_FALLBACK", "1"),
            "CHAMFER_ALLOW_FALLBACK": os.environ.get("CHAMFER_ALLOW_FALLBACK", "1"),
        }

    def repo_for_model(self, model: str) -> Path:
        return self.pointmae_root if model == "pointmae" else self.pcpmae_root

    def python_for_model(self, model: str) -> str:
        return self.pointmae_python if model == "pointmae" else self.pcpmae_python

    def config_path(self, model: str, variant: str) -> Path:
        return self.repo_for_model(model) / CONFIG_BY_MODEL[model][variant]

    def find_shapenetpart_root(self) -> Path | None:
        candidates = [
            self.root / "data" / "shapenetcore_partanno_segmentation_benchmark_v0_normal",
            Path("/mnt/urashima/users/minesawa/3D-NEPA-data/shapenetcore_partanno_segmentation_benchmark_v0_normal"),
        ]
        for cand in candidates:
            if (cand / "synsetoffset2category.txt").is_file() and (cand / "train_test_split").is_dir():
                return cand
        return None

    def write_blocker(self, text: str) -> None:
        path = self.result_dir / "blockers.md"
        with path.open("a") as f:
            f.write(f"- {text}\n")
        print(f"[blocker] {text}")

    def write_grouping_blocked(self) -> None:
        rows = []
        for model in ["pointmae", "pcpmae"]:
            rows.append(
                {
                    "model": model,
                    "task": "grouping",
                    "status": f"BLOCKED: grouping swap incompatible with {model} without architecture hooks or retraining",
                    "git_commit": self.git_commit,
                }
            )
        write_json(self.result_dir / "object_ssl_pointmae_pcpmae_grouping.json", rows)
        write_csv(self.result_dir / "object_ssl_pointmae_pcpmae_grouping.csv", rows, ["model", "task", "status", "git_commit"])
        write_text(
            self.result_dir / "object_ssl_pointmae_pcpmae_grouping.md",
            "# Point-MAE / PCP-MAE Grouping Diagnostics\n\n" + markdown_table(rows, ["model", "task", "status", "git_commit"]),
        )

    def aggregate(self) -> None:
        raw_files = sorted(self.raw_dir.rglob("*.json"))
        clean_rows: list[dict[str, Any]] = []
        support_rows: list[dict[str, Any]] = []
        topk_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        raw_payloads = []
        for path in raw_files:
            payload = read_json(path)
            raw_payloads.append(payload)
            meta = payload["metadata"]
            conditions = payload["conditions"]
            clean = next((x for x in conditions if x["condition"] == "clean"), None)
            if not clean:
                continue
            if "n_samples" not in meta and "n_shapes" in clean:
                meta = {**meta, "n_samples": clean["n_shapes"]}
            if meta["task"] == "scanobjectnn":
                clean_metrics = ["top1", "top2_hit", "top5_hit"]
                for metric in clean_metrics:
                    clean_rows.append(self.metric_row(meta, metric, clean.get(metric)))
                for row in conditions:
                    support_rows.append(
                        {
                            **self.base_row(meta),
                            "condition": row["condition"],
                            "metric_name": "Top-1 (%)",
                            "score": row.get("top1"),
                            "damage_pp": row.get("damage_pp"),
                        }
                    )
                topk = {
                    **self.base_row(meta),
                    "top1": clean.get("top1"),
                    "top2_hit": clean.get("top2_hit"),
                    "top5_hit": clean.get("top5_hit"),
                    "oracle2_score": clean.get("oracle2_score"),
                    "oracle5_score": clean.get("oracle5_score"),
                    "hardest_pair": clean.get("hardest_pair"),
                    "n_samples": clean.get("n_samples"),
                }
                topk_rows.append(topk)
                if meta.get("selection_protocol") in {"heldout_selection", "selection=reporting"}:
                    selection_rows.append(topk)
            elif meta["task"] == "shapenetpart":
                for metric in ["class_avg_miou", "instance_avg_miou", "point_top1", "point_top2_hit", "point_top5_hit"]:
                    clean_rows.append(self.metric_row(meta, metric, clean.get(metric)))
                for row in conditions:
                    support_rows.append(
                        {
                            **self.base_row(meta),
                            "condition": row["condition"],
                            "metric_name": "Instance mIoU (%)",
                            "score": row.get("instance_avg_miou"),
                            "damage_pp": row.get("damage_pp"),
                        }
                    )
                topk_rows.append(
                    {
                        **self.base_row(meta),
                        "top1": clean.get("point_top1"),
                        "top2_hit": clean.get("point_top2_hit"),
                        "top5_hit": clean.get("point_top5_hit"),
                        "oracle2_score": clean.get("oracle2_instance_avg_miou"),
                        "oracle5_score": clean.get("oracle5_instance_avg_miou"),
                        "hardest_pair": "",
                        "n_samples": clean.get("n_shapes"),
                    }
                )

        write_json(self.result_dir / "object_ssl_pointmae_pcpmae_clean.json", clean_rows)
        write_json(self.result_dir / "object_ssl_pointmae_pcpmae_support_stress.json", support_rows)
        write_json(self.result_dir / "object_ssl_pointmae_pcpmae_topk.json", topk_rows)
        write_json(self.result_dir / "object_ssl_pointmae_pcpmae_selection.json", selection_rows)
        clean_fields = [
            "model",
            "checkpoint_path",
            "task",
            "split",
            "selection_protocol",
            "metric_name",
            "score",
            "n_samples",
            "seed",
            "script",
            "git_commit",
            "notes",
        ]
        write_csv(self.result_dir / "object_ssl_pointmae_pcpmae_clean.csv", clean_rows, clean_fields)
        write_csv(
            self.result_dir / "object_ssl_pointmae_pcpmae_support_stress.csv",
            support_rows,
            clean_fields[:5] + ["condition", "metric_name", "score", "damage_pp", "n_samples", "seed", "script", "git_commit", "notes"],
        )
        topk_fields = [
            "model",
            "task",
            "split",
            "selection_protocol",
            "top1",
            "top2_hit",
            "top5_hit",
            "oracle2_score",
            "oracle5_score",
            "hardest_pair",
            "n_samples",
            "checkpoint_path",
            "notes",
        ]
        write_csv(self.result_dir / "object_ssl_pointmae_pcpmae_topk.csv", topk_rows, topk_fields)
        write_csv(self.result_dir / "object_ssl_pointmae_pcpmae_selection.csv", selection_rows, topk_fields)
        write_text(self.result_dir / "object_ssl_pointmae_pcpmae_clean.md", markdown_table(clean_rows, clean_fields))
        write_text(
            self.result_dir / "object_ssl_pointmae_pcpmae_support_stress.md",
            markdown_table(support_rows, ["model", "task", "split", "condition", "metric_name", "score", "damage_pp", "selection_protocol"]),
        )
        write_text(self.result_dir / "object_ssl_pointmae_pcpmae_topk.md", markdown_table(topk_rows, topk_fields))
        write_text(self.result_dir / "object_ssl_pointmae_pcpmae_selection.md", markdown_table(selection_rows, topk_fields))
        self.write_summary(clean_rows, support_rows, topk_rows, selection_rows)
        write_json(self.result_dir / "raw_payloads.json", raw_payloads)

    def base_row(self, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": meta.get("model"),
            "checkpoint_path": meta.get("checkpoint_path"),
            "task": meta.get("task"),
            "split": meta.get("split"),
            "selection_protocol": meta.get("selection_protocol"),
            "n_samples": meta.get("n_samples", ""),
            "seed": meta.get("seed"),
            "script": meta.get("script"),
            "git_commit": meta.get("git_commit"),
            "notes": meta.get("notes", ""),
        }

    def metric_row(self, meta: dict[str, Any], metric: str, score: Any) -> dict[str, Any]:
        return {
            **self.base_row(meta),
            "metric_name": metric,
            "score": score,
        }

    def write_summary(
        self,
        clean_rows: list[dict[str, Any]],
        support_rows: list[dict[str, Any]],
        topk_rows: list[dict[str, Any]],
        selection_rows: list[dict[str, Any]],
    ) -> None:
        blockers = (self.result_dir / "blockers.md").read_text() if (self.result_dir / "blockers.md").is_file() else "_None recorded._\n"
        manifest = read_json(self.manifest_path()) if self.manifest_path().is_file() else {}
        lines = [
            "# Point-MAE / PCP-MAE Object SSL Diagnostics Summary",
            "",
            "These diagnostics test whether the object-side support and readout ambiguities persist beyond the PointGPT scaffold.",
            "They do not by themselves prove a universal object-level 3D SSL failure.",
            "",
            f"- git commit: `{self.git_commit}`",
            f"- result dir: `{self.result_dir}`",
            f"- log dir: `{self.log_dir}`",
            "",
            "## Checkpoints",
            "",
            "```json",
            yaml.safe_dump(manifest, sort_keys=True),
            "```",
            "",
            "## Clean Reproduction",
            "",
            markdown_table(clean_rows, ["model", "task", "split", "selection_protocol", "metric_name", "score"]),
            "## Q3 Support Perturbations",
            "",
            markdown_table(support_rows, ["model", "task", "split", "condition", "metric_name", "score", "damage_pp"]),
            "## Q4 Candidate Sets",
            "",
            markdown_table(topk_rows, ["model", "task", "split", "selection_protocol", "top1", "top2_hit", "top5_hit", "oracle2_score", "oracle5_score"]),
            "## Q4 Selection Protocol",
            "",
            markdown_table(selection_rows, ["model", "task", "split", "selection_protocol", "top1", "top2_hit", "top5_hit"]),
            "## Grouping Diagnostics",
            "",
            self.grouping_summary_text(),
            "",
            "## Data Paths",
            "",
            f"- ScanObjectNN: `{self.root / 'data' / 'ScanObjectNN' / 'h5_files'}`",
            f"- ShapeNetPart: `{self.find_shapenetpart_root() if self.find_shapenetpart_root() else 'BLOCKED: not found'}`",
            "",
            "## Scripts And Commands",
            "",
            "- Main orchestrator: `scripts/object_ssl/pointmae_pcpmae_chain.py`",
            "- ScanObjectNN adapter: `scripts/object_ssl/eval_scanobjectnn_mae.py`",
            "- ShapeNetPart adapter: `scripts/object_ssl/eval_shapenetpart_mae.py`",
            "- Held-out split maker: `scripts/object_ssl/make_scanobjectnn_heldout_split.py`",
            f"- Logs with exact commands: `{self.log_dir}`",
            "",
            "## Blockers",
            "",
            blockers,
            "",
        ]
        write_text(self.docs_dir / "object_ssl_pointmae_pcpmae_diagnostics_summary.md", "\n".join(lines))

    def grouping_summary_text(self) -> str:
        grouping_md = self.result_dir / "object_ssl_pointmae_pcpmae_grouping.md"
        if grouping_md.is_file():
            text = grouping_md.read_text().strip()
            marker = "- raw dir:"
            if marker in text:
                return text[text.index(marker) :].strip()
            return text
        return (
            "BLOCKED: grouping swaps are not run because Point-MAE/PCP-MAE grouping is embedded "
            "in model internals and swapping modes would require architecture hooks or retraining."
        )

    def write_audit(self, label: str) -> None:
        part_root = self.find_shapenetpart_root()
        manifest = read_json(self.manifest_path()) if self.manifest_path().is_file() else {}
        lines = [
            "# Point-MAE / PCP-MAE Object SSL Audit",
            "",
            f"- phase: `{label}`",
            f"- git commit: `{self.git_commit}`",
            f"- root: `{self.root}`",
            f"- Point-MAE root: `{self.pointmae_root}`",
            f"- PCP-MAE root: `{self.pcpmae_root}`",
            f"- Point-MAE python: `{self.pointmae_python}`",
            f"- PCP-MAE python: `{self.pcpmae_python}`",
            f"- GPUs: `{','.join(self.gpus)}`",
            f"- ScanObjectNN root: `{self.root / 'data' / 'ScanObjectNN' / 'h5_files'}`",
            f"- ShapeNetPart root: `{part_root if part_root else 'BLOCKED: not found'}`",
            "",
            "## Checkpoint Manifest",
            "",
            "```json",
            yaml.safe_dump(manifest, sort_keys=True),
            "```",
            "",
            "## Checkpoint Notes",
            "",
            "- Point-MAE official ScanObjectNN release filenames were mapped by checkpoint-internal metrics: `scan_objonly.pth` is used for `obj_bg` and `scan_objbg.pth` is used for `obj_only` in these diagnostics.",
            "",
            "## Existing support/top-k utilities",
            "",
            "- Reused protocol: `PointGPT/tools/eval_scanobjectnn_support_stress.py` structured/random/xyz-zero semantics.",
            "- New adapters: `scripts/object_ssl/eval_scanobjectnn_mae.py`, `scripts/object_ssl/eval_shapenetpart_mae.py`.",
            "",
        ]
        write_text(self.docs_dir / "object_ssl_pointmae_pcpmae_audit.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    Chain(args).run()


if __name__ == "__main__":
    main()
