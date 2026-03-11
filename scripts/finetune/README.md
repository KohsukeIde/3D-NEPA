# Finetune Workers

This folder holds maintained fine-tune workers and scheduler-facing wrappers.

It is not the primary direct-run surface for this workstation.

## Boundary

- if you are directly launching a job on this PC, start from `scripts/local/`
- if you are handing someone an ABCI entrypoint, start from `scripts/abci/`
- scripts here are the worker / wrapper layer used by those primary surfaces
- direct invocation from this folder is for debugging, internal development,
  or when a maintained local wrapper does not yet expose the needed knob

## Contents

- `nepa3d_finetune*.sh`, `patchnepa_scanobjectnn_finetune.sh`
  - maintained 3D fine-tune workers / wrappers
- `patchcls_*`
  - classification fine-tune helpers
- `nepa_b_sft.sh`, `nepa_l_sft.sh`
  - 2D NEPA supervised fine-tune helpers retained as maintained worker scripts

## Rule

Do not put workstation-specific orchestration, local queue policy, or
machine-local W&B conventions here. Those belong in `scripts/local/`.
