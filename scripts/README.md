# Scripts Layout

This directory is split into active scheduler/runner entrypoints and archived
legacy scripts.

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
    ablation submitters)
- `scripts/logs/`
  - maintained workspace cleanup helpers
- `scripts/local/`
  - maintained local-only queue manifests and runners for post-PBS operation
- `scripts/abci/`
  - curated collaborator-facing ABCI submit wrappers for the current PatchNEPA
    line
  - start here when someone asks "which script should I run on ABCI?"
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
2. keep ad-hoc local chains out of active folders; maintained local-only ops
   belong under `scripts/local/`
3. keep collaborator-facing ABCI shortcuts under `scripts/abci/` and point
   them at maintained workers/submitters rather than duplicating logic
4. if a script becomes historical, broken-by-default, or unused, move it to
   `scripts/legacy/<date_tag>/`
