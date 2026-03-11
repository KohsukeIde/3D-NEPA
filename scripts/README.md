# Scripts Layout

This directory is split into primary execution surfaces, supporting workers,
and archived legacy scripts.

## Execution Boundary

- `scripts/local/`
  - source of truth for workstation / local GPU execution
  - use this for runs that assume the local filesystem, local logs, local W&B
    state, or fixed 1-2 GPU DDP layouts
- `scripts/abci/`
  - source of truth for collaborator-facing ABCI entrypoints
  - use this when the question is "which script should I run on ABCI?"
- `scripts/sanity/`
  - supporting sanity / screening / compatibility area only
  - do not treat this as a primary launch surface

If a launcher is maintained and local-only, it belongs in `scripts/local/`.
If it is maintained and ABCI-facing, it belongs in `scripts/abci/`.

## Active entrypoints

- `scripts/pretrain/`
  - maintained pretrain PBS wrappers and submitters
  - mainline examples: `nepa3d_pretrain.sh`, `submit_pretrain_abcd_qf.sh`,
    `submit_pretrain_patch_nepa_pointonly_qf.sh`,
    `submit_pretrain_patch_nepa_tokens_qf.sh`
- `scripts/eval/`
  - maintained classification / CPAC eval workers and submitters
  - mainline examples: `nepa3d_eval_cls_cpac_qf.sh`,
    `submit_abcd_cls_cpac_qf.sh`,
    `submit_sotafair_variants_llrd_droppath_ablation_qf.sh`
- `scripts/preprocess/`
  - cache/data preparation workers and submitters
  - includes specialized migration/backfill helpers
- `scripts/analysis/`
  - maintained UCPR / CPAC / k-plane evaluation wrappers
- `scripts/finetune/`
  - maintained fine-tune entrypoints (`patchcls_*`, `patchnepa_*`,
    `nepa3d_finetune*.sh`)
- `scripts/sanity/`
  - external baseline / environment sanity jobs (Point-MAE, PointGPT, patch
    ablation submitters) plus compatibility shims
  - see `scripts/sanity/README.md`
- `scripts/logs/`
  - maintained workspace cleanup helpers
- `scripts/local/`
  - maintained local-only queue manifests, runners, and workstation pipelines
  - this is the primary local execution surface
  - see `scripts/local/README.md`
- `scripts/abci/`
  - curated collaborator-facing ABCI submit wrappers for the current PatchNEPA
    line
  - this is the primary ABCI execution surface
  - see `scripts/abci/README.md`

## Archived scripts

- archived scripts are moved under `scripts/legacy/`
- current archive batches:
  - `scripts/legacy/orphaned_20260226/`
    - unreferenced/orphaned scripts from the first audit
  - `scripts/legacy/local_chains_20260226/`
    - all historical `*_local.sh` orchestration scripts
  - `scripts/legacy/deprecated_20260306/`
    - outdated chain helpers that only remained in historical docs and whose
      default active-tree targets had already disappeared

## Audit note

- 2026-03-06 audit result:
  - active `scripts/finetune/` no longer contains the broken M1/chain helpers
    that depended on removed active local-chain launchers
  - docs that still refer to archived local chains should point to
    `scripts/legacy/local_chains_20260226/` or
    `scripts/legacy/deprecated_20260306/` explicitly

## Maintenance rule

When adding new automation:

1. prefer `submit_*_qf.sh` style for PBS entrypoints
2. keep maintained local-only launchers under `scripts/local/`; do not put
   new workstation entrypoints under `scripts/sanity/`, `scripts/pretrain/`,
   or `scripts/finetune/`
3. keep collaborator-facing ABCI shortcuts under `scripts/abci/` and point
   them at maintained workers/submitters rather than duplicating logic
4. use `scripts/sanity/` only for sanity checks, screening jobs, environment
   setup helpers, or backward-compatible shims
5. if a script becomes historical, broken-by-default, or unused, move it to
   `scripts/legacy/<date_tag>/`
