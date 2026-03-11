# Sanity And Compatibility

This folder is not a primary execution surface.

Use it only for:

- environment checks
- one-off sanity / smoke jobs
- QF screening submitters
- external baseline probes
- compatibility shims kept to avoid breaking old references

## Boundary

- local workstation pipelines belong in `scripts/local/`
- collaborator-facing ABCI entrypoints belong in `scripts/abci/`
- maintained scheduler workers still belong in `scripts/pretrain/`,
  `scripts/finetune/`, `scripts/eval/`, and `scripts/analysis/`

In particular:

- `scripts/sanity/pointgpt_train_local_ddp.sh`
- `scripts/sanity/pointgpt_finetune_local_ddp.sh`
- `scripts/sanity/pointgpt_nepa_vs_cdl12_pipeline.sh`

are compatibility shims only. Their maintained implementations live under
`scripts/local/`.

## Rule

Do not add a new long-lived launcher here if it is meant to be the default
local or ABCI path. Put it in the correct primary surface and, if needed, keep
only a thin shim here.
