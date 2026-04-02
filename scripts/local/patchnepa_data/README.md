# PatchNEPA Local Data Ops

This directory is the maintained local-only execution area for data/cache
preparation on this workstation.

Use this folder when the launch assumptions are specific to this machine, for
example:

- `/mnt/urashima/users/minesawa/3D-NEPA-data` as the physical data root
- `/home/minesawa/anaconda3/bin/python` as the Python runtime
- local timeout / retry policy for long-tail ShapeNet meshes
- local log placement under `/mnt/.../logs/`

Do not put ABCI submit wrappers here. Those still belong under `scripts/abci/`.
Do not duplicate the core worker logic here. The maintained workers remain in
`scripts/preprocess/`; this folder only contains workstation entrypoints.

## Current entrypoints

- `run_shapenet_worldvis_local.sh`
  - local wrapper for canonical `worldvis/v3`-style ShapeNet preprocessing
  - keeps the original worldvis settings
  - adds local timeout / resume defaults
  - starts detached by default in a dedicated `tmux` session
  - uses shared-server defaults: `WORKERS=16`, `TASK_TIMEOUT_SEC=900`, `nice=10`, `ionice best-effort/7`
- `status_shapenet_worldvis_local.sh`
  - prints tmux/pid state, cache count, and recent log lines

## Runtime notes

- physical data root:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data`
- canonical local log root:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data/`
- pid file:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data/shapenet_worldvis_local.pid`

## Launch mode

- default:
  - detached `tmux` launch for long-running server-side preprocessing
- interactive debugging:
  - `FOREGROUND=1 bash scripts/local/patchnepa_data/run_shapenet_worldvis_local.sh`
- fallback:
  - `LAUNCH_MODE=nohup bash scripts/local/patchnepa_data/run_shapenet_worldvis_local.sh`

## Source of truth

- local-vs-ABCI boundary:
  - `scripts/local/README.md`
  - `scripts/abci/README.md`
- workstation-specific PatchNEPA notes:
  - `nepa3d/docs/patch_nepa/itachi/README.md`
  - `nepa3d/docs/patch_nepa/itachi/local_data_ops_202604.md`
- active scientific storyline:
  - `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
