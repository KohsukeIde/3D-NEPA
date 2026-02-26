# Scripts Layout

This directory is split into active entrypoints and archived legacy scripts.

## Active entrypoints

- `scripts/pretrain/submit_*.sh`
  - pretrain job submission (A/B/C/D, rfps, mesh-udf-only, point-only).
- `scripts/eval/submit_*.sh`
  - eval job submission (classification / CPAC / variant split / query-rethink).
- `scripts/pipeline/submit_*.sh`
  - chained pretrain->eval pipelines.
- `scripts/preprocess/submit_*.sh`
  - dataset/cache preprocessing submissions.
- `scripts/sanity/submit_*.sh`
  - external sanity baselines (e.g. Point-MAE ScanObjectNN).

## Archived scripts

- archived scripts are moved under `scripts/legacy/`.
- current archive batch:
  - `scripts/legacy/orphaned_20260226/`
  - `scripts/legacy/local_chains_20260226/` (`*_local.sh` full move)
  - criterion: no in-repo references and not part of active docs/pipeline.

## Maintenance rule

When adding new automation:

1. prefer `submit_*_qf.sh` style for PBS entrypoints.
2. keep ad-hoc local chains out of active folders.
3. if a script becomes unused, move it to `scripts/legacy/<date_tag>/`.
