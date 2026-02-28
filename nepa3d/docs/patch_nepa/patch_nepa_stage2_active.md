# Patch-NEPA Stage 2 Active Plan

Last updated: 2026-02-28

## 1. Scope

This document is the active source for the new pipeline:

- pretrain: `patch_nepa` (`nepa3d/models/patch_nepa.py`)
- finetune/eval: `patchcls` (`nepa3d/train/finetune_patch_cls.py`)
- target protocol: ScanObjectNN variant split (`obj_bg`, `obj_only`, `pb_t50_rs`)

`query_nepa` (`nepa3d/models/query_nepa.py`) is treated as a separate legacy line for historical comparison and is not used for new Stage-2 runs.

## 2. Model/Script Roles

- Patch pretrain model:
  - `nepa3d/models/patch_nepa.py`
  - task: next-embedding prediction on patch tokens
  - optional Option-A ray bind: ray -> nearest point patch center
- Patch pretrain entry:
  - `nepa3d/train/pretrain_patch_nepa.py`
- Patch pretrain launchers:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
- Patch finetune/eval entry:
  - `nepa3d/train/finetune_patch_cls.py`

## 2.1 Query-NEPA vs Patch-NEPA Settings Mapping

The following Query-NEPA controls are not active in current Patch-NEPA point-only runs:

- `qa_layout` (`interleave` / `split` / `split_sep`)
- `sequence_mode` (`block` / `event`)
- `event_order_mode`
- `ray_order_mode`

Current Stage-2 point-only run is:

- patch sequence (`serial_order=morton`, `group_size=32`, `num_groups=32`)
- `N_RAY=0`, `USE_RAY_PATCH=0`
- NEPA next-embedding prediction on patch tokens

## 3. Runtime Policy

- Stage-2 baseline starts from point-only pretrain:
  - `N_RAY=0`
  - `USE_RAY_PATCH=0`
  - ShapeNet pointcloud-only mix first
- Load policy for pretrain -> finetune:
  - `strict=False`
  - encoder-compatible config must be matched (`qk_norm*`, `layerscale`, `patch_embed`, grouping)
- Split policy:
  - follow Point-MAE-style split settings for fair scratch/finetune comparison
  - avoid mixing with old `main_split_v2` headline reporting

## 4. Current Active Run Set

- point-only pretrain submission:
  - job: `99591.qjcm`
  - run_set: `patchnepa_pointonly_20260228_143750`
  - save: `runs/patchnepa_pointonly/patchnepa_pointonly_20260228_143750`
  - log: `logs/patch_nepa_pretrain/patchnepa_pointonly_20260228_143750/run_pointonly_patchnepa_pointonly_20260228_143750.log`

## 5. Reporting Rule

- High-level benchmark tables stay in:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- Job-by-job execution history for Stage-2 goes to:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Old ledgers remain historical and are not primary sources for new Stage-2 claims.

## 6. Porting Checklist from Query-NEPA Line

Required to declare parity at the objective/recipe level:

1. Keep core NEPA cosine next-step loss (already active).
2. Add optional masking/target-selection policy parity where needed.
3. Re-introduce Query-line aux losses selectively (B2/B3/C/D/E equivalents) on patch tokens.
4. Re-run protocol-correct variant split evaluation after each parity milestone.
