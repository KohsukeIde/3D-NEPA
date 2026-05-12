# pointNEPA Sanity Scripts

This directory contains PointGPT / pointNEPA QF, smoke, one-off, and
compatibility launchers.

Use these for environment checks, short ABCI screening jobs, and historical
compatibility wrappers. Maintained local execution should start from
`pointnepa/scripts/local/`.

## Compatibility Shims

- `pointgpt_train_local_ddp.sh`
- `pointgpt_finetune_local_ddp.sh`
- `pointgpt_nepa_vs_cdl12_pipeline.sh`

These forward to the maintained local implementations in
`pointnepa/scripts/local/`.

## Boundary

PointGPT / pointNEPA QF jobs belong here, not under the repo-global
`scripts/sanity/` tree. PatchNEPA and Point-MAE sanity jobs remain outside this
sidecar.
