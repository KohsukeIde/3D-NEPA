# Itachi Local Data Ops

This note defines where local-only data-preparation launchers should live and
how they differ from ABCI-facing submit wrappers on `itachi`.

## Boundary

- local workstation entrypoints belong under `scripts/local/patchnepa_data/`
- core preprocessing workers remain under `scripts/preprocess/`
- collaborator-facing ABCI wrappers remain under `scripts/abci/`
- science and protocol claims remain in the parent `patch_nepa/` docs

## Current local roots

- launcher folder:
  - `scripts/local/patchnepa_data/`
- physical data root:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data`
- local log root:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data/`
- local pid file:
  - `/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data/shapenet_worldvis_local.pid`

## Why this split exists

The maintained ShapeNet/ScanObjectNN preprocessing workers are general-purpose
workers. On `itachi` we need extra local assumptions:

- `/mnt`-backed data placement instead of repo-local `data/`
- local Python/runtime selection
- local timeout/defer policy for pathological meshes
- resume-first execution over partially built caches

Those are operational concerns, not scientific protocol changes, so they
should live in local wrappers and local notes rather than in ABCI submit docs.

## Current launch policy

Long-running local ShapeNet preprocessing should not be started as a foreground
SSH-bound job. The maintained launcher under `scripts/local/patchnepa_data/`
starts detached in `tmux` by default and records:

- a stable log file under `/mnt/.../logs/patchnepa_data/`
- a pid file for liveness checks
- resume-first execution against the existing cache root
- reduced shared-server pressure via moderate worker count, shorter defer timeout,
  and low-priority scheduling

Use `FOREGROUND=1` only when debugging a short run interactively. Use
`LAUNCH_MODE=nohup` only as a fallback when `tmux` is unavailable.
