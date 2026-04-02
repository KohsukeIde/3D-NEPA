# Itachi Local Geo-Teacher Post-Train Ops

This note defines the local-only execution boundary for downstream jobs that
start from the current geo-teacher pretrain checkpoint on `itachi`.

## Boundary

- this machine is **not** ABCI
- local downstream launches should live under
  `scripts/local/patchnepa_geo_teacher/`
- local ScanObjectNN cache preparation should live under
  `scripts/local/patchnepa_data/`
- collaborator-facing ABCI submit wrappers remain under `scripts/abci/`
- paper-facing protocol and interpretation remain in the parent `patch_nepa/`
  docs

## Current checkpoint boundary

The maintained local post-train chain assumes a ready pretrain checkpoint from:

- save root:
  - `runs/cqa_itachi/`
- default run tag:
  - `geo_teacher_distnorm_unsigned_100ep_itachi_main`
- default checkpoint:
  - `ckpt_final.pt`

The chain can wait for the pretrain job to stop, but it does not retrigger
pretraining itself.

## Current runnable downstream set

The maintained local chain currently covers the runnable subset of the Route
A/B matrix.

Route A subset:

- `ScanObjectNN` direct FT on:
  - `obj_bg`
  - `obj_only`
  - `pb_t50_rs`
- budget:
  - `300` epochs each

Route B subset:

- multitype token suite:
  - `same_context`
  - `degraded_context`
  - control deltas
- `udf_distance` completion:
  - same context
  - degraded context

Not yet wired into the maintained local chain:

- `ShapeNetPart`

## Current local entrypoints

- post-train launcher:
  - `scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh`
- post-train status:
  - `scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh`
- ScanObjectNN variant-cache launcher:
  - `scripts/local/patchnepa_data/run_scanobjectnn_variants_local.sh`
- ScanObjectNN variant-cache status:
  - `scripts/local/patchnepa_data/status_scanobjectnn_variants_local.sh`

## Launch policy

Start detached:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```

Check status:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh
```

Foreground debug:

```bash
FOREGROUND=1 bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```

## Local resource policy

The current downstream chain assumes:

- `4 x A100 80GB`
- three independent `ScanObjectNN` FT jobs on `GPU 0,1,2`
- Route-B evals on `GPU 3`

The current local chain uses one GPU per FT job instead of local multi-GPU FT.
This keeps the three ScanObjectNN variants moving at once while preserving one
GPU for the geometry readouts.

## W&B note

Local FT jobs can use W&B, but the maintained post-train chain defaults to
offline mode unless explicitly overridden at launch time.

Reason:

- local W&B auth on this machine is not guaranteed to be active at every login
- downstream automation should not fail just because online logging is missing

Override when desired:

```bash
FT_WANDB_MODE=online bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```
