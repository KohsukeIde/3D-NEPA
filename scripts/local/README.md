# Local Ops

This directory is the maintained local-only operations area for PatchNEPA and
external baseline work on this workstation.

It exists because local GPU execution is the current default execution surface
for this machine, while ABCI-facing entrypoints are kept separately under
`scripts/abci/`.

## Boundary

- this folder is the source of truth for workstation execution
- local assumptions are allowed here:
  - fixed small-GPU layouts such as 1-2 GPU DDP
  - local log roots under `logs/local/`
  - local W&B naming / resumption conventions
  - machine-specific queue manifests
- if a launcher is meant for ABCI users, it does not belong here
- `scripts/sanity/pointgpt_*_local_*.sh` are compatibility shims only
- science and benchmark interpretation do not belong here; keep those in
  `nepa3d/docs/...`

## Files

- `patchnepa_local_queue.tsv`
  - machine-readable queue manifest
- `patchnepa_local_queue_runner.sh`
  - local queue runner
- `patchnepa_local_status.sh`
  - status view over the manifest + runtime state
- `pointgpt_train_local_ddp.sh`
  - PointGPT local 2-GPU DDP pretrain wrapper
- `pointgpt_finetune_local_ddp.sh`
  - PointGPT local 2-GPU DDP ScanObjectNN fine-tune wrapper
- `pointgpt_nepa_vs_cdl12_pipeline.sh`
  - local sequential pipeline:
    `nepa_cosine pretrain -> cdl12 pretrain -> obj_bg fine-tune compare`
- `pointgpt_ft_recipe_matrix_2x2.sh`
  - local sequential matrix:
    `pretrain objective {nepa_cosine, cdl12} x FT recipe {cls-only, PointGPT FT}`

## Source of Truth

- scientific conclusions still live in:
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- execution order, gating, and local budget live in:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- local-vs-ABCI operational boundary lives in:
  - `nepa3d/docs/operations/README.md`

## Usage

Run the queue:

```bash
GPU_IDS=0,1 bash scripts/local/patchnepa_local_queue_runner.sh
```

Check status:

```bash
bash scripts/local/patchnepa_local_status.sh
```

Run the current visibility-first exploratory branch directly:

```bash
bash scripts/local/patchnepa_visocc_branch.sh
```

## Operational Rules

- runner state is written under `logs/local_queue/<queue_name>/`
- the runner does not update docs automatically
- canonicalization must be done manually from structured outputs
- keep launch commands relative to the repo root
- prefer existing maintained wrappers under `scripts/pretrain/`,
  `scripts/analysis/`, and `scripts/finetune/`
- use direct `python -m ...` only when the maintained wrapper does not expose
  the needed knobs
- if a branch script already manages both GPUs internally (for example
  `patchnepa_visocc_branch.sh`), prefer launching it directly or make it the
  only enabled queue row
- `scripts/sanity/pointgpt_*_local_*.sh` are compatibility shims only;
  maintained local entrypoints live here under `scripts/local/`
