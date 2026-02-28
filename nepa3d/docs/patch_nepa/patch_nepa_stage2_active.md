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

- patch sequence (`patch_embed=fps_knn`, `group_size=32`, `num_groups=64`)
- `N_RAY=0`, `USE_RAY_PATCH=0`
- NEPA next-embedding prediction on patch tokens

## 2.2 Pretrain Baseline Parity (excluding split/dual-mask/QA)

Current baseline policy for Patch-NEPA pretrain is:

- keep Query-NEPA-equivalent training recipe where applicable:
  - `LR=3e-4`
  - `BATCH(per-proc)=16` and explicit global-batch logging
  - `weight_decay=0.05`
  - `lr_scheduler=none` (default)
  - `auto_resume=1`
  - `pt_sample_mode=rfps` (default, with `pt_rfps_m=4096`)
- keep PatchCLS-compatible backbone defaults:
  - `patch_embed=fps_knn`
  - `group_size=32`, `num_groups=64`
  - `qk_norm=1`, `qk_norm_affine=0`, `qk_norm_bias=0`
  - `layerscale_value=1e-5`
  - `rope_theta=100`
- augmentation knobs are exposed with Query-NEPA semantics (`aug_*`) and default OFF.

Reference audit/checklist:

- `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`

## 3. Runtime Policy

- Stage-2 baseline starts from point-only pretrain:
  - `N_RAY=0`
  - `USE_RAY_PATCH=0`
  - ShapeNet pointcloud-only mix first
- Load policy for pretrain -> finetune:
  - `strict=False`
  - encoder-compatible config must be matched (`qk_norm*`, `layerscale`, `patch_embed`, grouping)
- Split policy:
  - `val_split_mode=file` only (strict mainline policy)
  - `group_*` and `pointmae(test-as-val)` are historical/reference-only and must not be used for new mainline runs
  - avoid mixing with old `main_split_v2` headline reporting

## 4. Current Active Run Set

- point-only pretrain submission:
  - `99591.qjcm`: early point-only baseline (single-process launch path, legacy defaults)
  - `99602.qjcm`: intermediate point-only run (4-GPU allocation, non-DDP launch; stopped)
  - `99613.qjcm`: first DDP migration run (2 nodes x 4 GPUs = 8 GPUs total; stopped for parity fixes)
    - run_set: `patchnepa_pointonly_ddp8_20260228_152058`
    - save: `runs/patchnepa_pointonly/patchnepa_pointonly_ddp8_20260228_152058`
    - pbs log: `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_20260228_152058.pbs.log`
    - node logs:
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99613.qjcm_run_pointonly_patchnepa_pointonly_ddp8_20260228_152058/logs/qh454.patchnepa.log`
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99613.qjcm_run_pointonly_patchnepa_pointonly_ddp8_20260228_152058/logs/qh455.patchnepa.log`
  - `99634.qjcm`: active DDP run (2 nodes x 4 GPUs = 8 GPUs total; parity-fixed launcher defaults)
    - run_set: `patchnepa_pointonly_ddp8_fix_20260228_153151`
    - save: `runs/patchnepa_pointonly/patchnepa_pointonly_ddp8_fix_20260228_153151`
    - pbs log: `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_fix_20260228_153151.pbs.log`
    - node logs:
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99634.qjcm_run_pointonly_patchnepa_pointonly_ddp8_fix_20260228_153151/logs/qh138.patchnepa.log`
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99634.qjcm_run_pointonly_patchnepa_pointonly_ddp8_fix_20260228_153151/logs/qh143.patchnepa.log`
  - `99643.qjcm`: active DDP run (2 nodes x 4 GPUs = 8 GPUs total; `rfps` default enforced)
    - run_set: `patchnepa_pointonly_ddp8_rfpsfix_20260228_154649`
    - save: `runs/patchnepa_pointonly/patchnepa_pointonly_ddp8_rfpsfix_20260228_154649`
    - pbs log: `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_rfpsfix_20260228_154649.pbs.log`
    - node logs:
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh138.patchnepa.log`
      - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh142.patchnepa.log`

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
