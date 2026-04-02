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
  - local orchestration entrypoint for ScanObjectNN FT and Route-B evals

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

The current local downstream chain covers the runnable subset of the Route A/B
matrix:

- Route A:
  - `ScanObjectNN` direct FT for `obj_bg`, `obj_only`, `pb_t50_rs`
- Route B:
  - multitype same/offdiag/control suite
  - `udf_distance` completion under same and degraded context

`ShapeNetPart` is still part of the scientific decision rule, but it is not
yet wired into the maintained local chain on `itachi`.

## W&B

Local geo-teacher runs use W&B by default through the launcher environment.

- package environment:
  `/home/minesawa/anaconda3/bin/python`
- default online logging:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```

Notes:

- a run that already started with `USE_WANDB=0` cannot be attached to W&B later
- keep the current run as-is if it is already mid-flight; switch on W&B from the next launch
- override only when needed:

```bash
USE_WANDB=0 bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```
