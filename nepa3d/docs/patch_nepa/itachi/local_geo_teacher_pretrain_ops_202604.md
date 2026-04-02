# Itachi Local Geo-Teacher Pretrain Ops

This note defines the local-only execution boundary for the matched
geo-teacher pretrain on `itachi`.

## Boundary

- this machine is **not** ABCI
- local long-running pretrains should be launched from
  `scripts/local/patchnepa_geo_teacher/`
- collaborator-facing ABCI submit wrappers remain under `scripts/abci/`
- the scientific protocol remains in the parent `patch_nepa/` docs

## Why a local wrapper exists

The maintained primitive-answering pretrain worker is general-purpose, but the
existing wrapper under `scripts/pretrain/` is PBS/Qsub-oriented.

On `itachi` we need extra local assumptions:

- direct multi-GPU launch via `python -m torch.distributed.run`
- detached `tmux` execution so the job does not depend on the SSH session
- local log roots under `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/`
- a clear handoff boundary so the same protocol can later move to ABCI

These are operational concerns, not paper claims.

## Current local roots

- launcher folder:
  - `scripts/local/patchnepa_geo_teacher/`
- local log root:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher/`
- local save root:
  - `runs/cqa_itachi/`

## Default current run

The current local default is the first strong teacher-target arm:

- mix config:
  - `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml`
- tasks:
  - `udf_distance`
  - `mesh_normal_unsigned`
- protocol:
  - `packed`
  - `multihead`
  - `loss_balance=per_task`
  - `answer_factorization=independent`
  - `query_interface_mode=no_q`
- budget:
  - `100` epochs
- hardware:
  - `4 x A100 80GB`

## Launch policy

Start long local pretrains detached:

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

## W&B policy on itachi

The local wrapper now defaults to online W&B logging, but W&B is still decided at launch time.

- current mid-run jobs are not restarted just to add logging
- if a run started with `USE_WANDB=0`, keep it unchanged
- default next launch:

```bash
bash scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh
```

Environment note:

- the maintained local Python is `/home/minesawa/anaconda3/bin/python`
- `wandb` is installed there for future local runs

## Handoff to ABCI

The local wrapper is for workstation execution only.

When the same protocol is moved to ABCI, the intended submit-side entrypoint is:

- `scripts/abci/submit_patchnepa_geo_teacher_compare_pretrain.sh`

Keep protocol changes in the parent docs/configs. Keep local machine
assumptions only in this note and in `scripts/local/patchnepa_geo_teacher/`.
