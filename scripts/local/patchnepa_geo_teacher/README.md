# Local Geo-Teacher Ops

This folder is the maintained `itachi`-local execution area for the
geo-teacher / CQA pretrain line.

Use this folder when:

- the machine is **not** ABCI
- the run should execute directly on local GPUs
- the launcher needs `tmux`, local log roots, or local DDP assumptions

Do not put collaborator-facing ABCI submit wrappers here. Those remain under
`scripts/abci/`.

## Files

- `run_geo_teacher_pretrain_local.sh`
  - detached local launcher for the matched geo-teacher pretrain
- `status_geo_teacher_pretrain_local.sh`
  - lightweight status and log-tail view
- `_run_geo_teacher_pretrain_inner.sh`
  - worker-side local DDP entrypoint used by the detached launcher
- `run_geo_teacher_posttrain_local.sh`
  - detached local launcher for downstream jobs after the current geo-teacher
    pretrain checkpoint is ready
- `status_geo_teacher_posttrain_local.sh`
  - lightweight status and artifact-count view for the downstream chain
- `_run_geo_teacher_posttrain_inner.sh`
  - local orchestration entrypoint for ScanObjectNN FT, ShapeNetPart FT, and
    Route-B evals
- `run_geo_teacher_curvature_probe_local.sh`
  - detached local launcher for the standalone frozen curvature probe
- `status_geo_teacher_curvature_probe_local.sh`
  - lightweight status view for the standalone curvature probe
- `_run_geo_teacher_curvature_probe_inner.sh`
  - worker-side entrypoint for the standalone curvature probe
- `run_geo_teacher_full300_chain_local.sh`
  - detached local launcher for the `300 epoch` pretrain -> downstream chain
- `status_geo_teacher_full300_chain_local.sh`
  - lightweight status and log-tail view for the `300 epoch` chain
- `_run_geo_teacher_full300_chain_inner.sh`
  - worker-side orchestration entrypoint for the `300 epoch` chain
- `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`
  - running result ledger for local pilot/canonical outputs

## Default run

The default local run is the first matched teacher-target arm:

- config:
  `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml`
- protocol:
  `packed + multihead + per_task + no_q + independent`
- GPUs:
  `4 x A100`
- epochs:
  `100`

The default launcher uses `python -m torch.distributed.run` because the
maintained `scripts/pretrain/*primitive_answering*` wrapper is PBS/Qsub-facing.

## Usage

Start detached:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```

Check status:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_pretrain_local.sh
```

Foreground smoke:

```bash
FOREGROUND=1 MAX_STEPS=2 bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```

Launch downstream chain after pretrain:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh
```

Check downstream chain status:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh
```

Launch standalone curvature probe for the current checkpoint:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_curvature_probe_local.sh
```

Check standalone curvature probe status:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_curvature_probe_local.sh
```

Launch the `300 epoch` full chain:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_full300_chain_local.sh
```

Check the `300 epoch` full chain:

```bash
bash scripts/local/patchnepa_geo_teacher/status_geo_teacher_full300_chain_local.sh
```

The current local downstream chain covers the runnable subset of the Route A/B
matrix:

- Route A:
  - `ScanObjectNN` direct FT for `obj_bg`, `obj_only`, `pb_t50_rs`
  - one variant at a time
  - `4 GPU` local DDP for each FT run
  - direct `scan_h5` input from raw `ScanObjectNN/h5_files/*`
  - H5 route eval now computes on-the-fly deterministic FPS, so `pt_sample_mode_eval=fps`
    matches the Point-MAE / PCP-MAE eval protocol more closely
  - `ShapeNetPart` direct part-seg FT
  - `4 GPU` local DDP
  - raw txt input from `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`
- Route B:
  - runs after the three FT jobs finish
  - multitype same/offdiag/control suite
  - `udf_distance` completion under same and degraded context
  - frozen `curvature` probe from `answer_hidden`

The maintained `300 epoch` chain is:

- `ShapeNet` geo-teacher pretrain
  - `distance + normal_unsigned`
  - `300 epoch`
  - `4 GPU` local DDP
  - global batch `128` via `BATCH=32/GPU`
- Route A:
  - `ScanObjectNN` `obj_bg -> obj_only -> pb_t50_rs`
  - `ShapeNetPart` part segmentation
- Route B:
  - same/offdiag/control suite
  - `udf_distance` completion
  - frozen `curvature` probe

## W&B

Local geo-teacher runs use W&B by default through the launcher environment.

- package environment:
  `/home/minesawa/anaconda3/bin/python`
- default online logging:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```

Local ScanObjectNN FT launched through the post-train chain defaults to:

- `WANDB_MODE=online`
- `WANDB_PROJECT=patchnepa-geo-teacher-scanobjectnn`
- `FT_VISIBLE_GPUS=0,1,2,3`
- `FT_NPROC_PER_NODE=4`
- `FT_DATA_FORMAT=scan_h5`

Local ShapeNetPart FT launched through the post-train chain defaults to:

- `WANDB_MODE=online`
- `WANDB_PROJECT=patchnepa-geo-teacher-shapenetpart`
- `PARTSEG_VISIBLE_GPUS=0,1,2,3`
- `PARTSEG_NPROC_PER_NODE=4`
- raw root `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

Default ShapeNetPart fetch path:

- primary:
  `https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip`
- fallback:
  `https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip`

Notes:

- a run that already started with `USE_WANDB=0` cannot be attached to W&B later
- keep the current run as-is if it is already mid-flight; switch on W&B from the next launch
- override only when needed:

```bash
USE_WANDB=0 bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```
