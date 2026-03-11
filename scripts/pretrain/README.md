# Pretrain Workers

This folder holds maintained pretrain workers, scheduler-facing wrappers, and
submit helpers.

It is not the primary direct-run surface for this workstation.

## Boundary

- if you are directly launching a job on this PC, start from `scripts/local/`
- if you are handing someone an ABCI entrypoint, start from `scripts/abci/`
- scripts here are the worker / wrapper layer used by those primary surfaces
- direct invocation from this folder is for debugging, internal development,
  or when a maintained local wrapper does not yet expose the needed knob

## Contents

- `nepa3d_pretrain*.sh`, `nepa3d_kplane_pretrain.sh`,
  `nepa3d_pretrain_patch_nepa*.sh`
  - maintained pretrain wrappers / workers
- `submit_pretrain*_qf.sh`
  - scheduler-facing submit helpers
- `nepa_b.sh`, `nepa_l.sh`
  - 2D NEPA pretrain helpers retained as maintained worker scripts
- `prune_pretrain_checkpoints.sh`
  - pretrain checkpoint maintenance helper

## Rule

Do not put workstation-specific orchestration, local queue policy, or
machine-local W&B conventions here. Those belong in `scripts/local/`.
