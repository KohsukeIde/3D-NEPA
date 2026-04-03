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
- execution:
  - run one variant at a time
  - `4 GPU` local DDP per FT run
  - direct `scan_h5` input from raw `ScanObjectNN` H5 roots
  - H5 direct eval uses on-the-fly deterministic FPS when `pt_sample_mode_eval=fps`
    so the eval crop policy is aligned with Point-MAE / PCP-MAE style FPS eval
- `ShapeNetPart` direct part-seg FT:
  - budget `300` epochs
  - `4 GPU` local DDP
  - raw txt root `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`
  - fetched locally via `scripts/local/patchnepa_data/prepare_shapenetpart_local.sh`

Route B subset:

- multitype token suite:
  - `same_context`
  - `degraded_context`
  - control deltas
- `udf_distance` completion:
  - same context
  - degraded context
- analysis-only extension:
  - frozen `curvature` probe from `answer_hidden`
- launch order:
  - after all Route-A jobs finish

## Current local entrypoints

- post-train launcher:
  - `scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh`
- post-train status:
  - `scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh`
- full `300 epoch` chain launcher:
  - `scripts/local/patchnepa_geo_teacher/run_geo_teacher_full300_chain_local.sh`
- full `300 epoch` chain status:
  - `scripts/local/patchnepa_geo_teacher/status_geo_teacher_full300_chain_local.sh`
- standalone curvature probe launcher:
  - `scripts/local/patchnepa_geo_teacher/run_geo_teacher_curvature_probe_local.sh`
- standalone curvature probe status:
  - `scripts/local/patchnepa_geo_teacher/status_geo_teacher_curvature_probe_local.sh`
- ScanObjectNN variant-cache launcher:
  - `scripts/local/patchnepa_data/run_scanobjectnn_variants_local.sh`
- ScanObjectNN variant-cache status:
  - `scripts/local/patchnepa_data/status_scanobjectnn_variants_local.sh`
- ShapeNetPart data prep launcher:
  - `scripts/local/patchnepa_data/prepare_shapenetpart_local.sh`
- local result ledger:
  - `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`

## Launch policy

Start detached:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```

Check status:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh
```

Run only the curvature probe against the current checkpoint:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_curvature_probe_local.sh
```

Foreground debug:

```bash
FOREGROUND=1 bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```

Queue the full `300 epoch` local chain:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_full300_chain_local.sh
```

## Local resource policy

The current downstream chain assumes:

- `4 x A100 80GB`
- one `ScanObjectNN` FT at a time with `4 GPU` local DDP
- one `ShapeNetPart` FT with `4 GPU` local DDP
- `scan_h5` direct input for FT; no variant-cache prep required
- ShapeNetPart raw txt input under `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`
- Route-B evals after FT completes

## W&B note

Local FT jobs on `itachi` should use W&B online logging by default.

Maintained default:

- `FT_WANDB_MODE=online`
- `FT_WANDB_PROJECT=patchnepa-geo-teacher-scanobjectnn`
- `PARTSEG_WANDB_MODE=online`
- `PARTSEG_WANDB_PROJECT=patchnepa-geo-teacher-shapenetpart`

Maintained ShapeNetPart fetch path:

- primary:
  `https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip`
- fallback:
  `https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip`

Override only when needed:

```bash
FT_WANDB_MODE=offline bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```
