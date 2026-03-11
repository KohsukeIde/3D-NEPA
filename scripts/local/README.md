# Local Ops

This directory is the maintained local-only operations area for PatchNEPA.

It exists because local GPU execution is now the primary launch surface until
NeurIPS, while PBS/QF runners are no longer the default operational path.

## Files

- `patchnepa_local_queue.tsv`
  - machine-readable queue manifest
- `patchnepa_local_queue_runner.sh`
  - local queue runner
- `patchnepa_local_status.sh`
  - status view over the manifest + runtime state

## Source of Truth

- scientific conclusions still live in:
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- execution order, gating, and local budget live in:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`

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
