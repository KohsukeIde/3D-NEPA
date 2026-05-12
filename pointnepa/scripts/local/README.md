# pointNEPA Local Scripts

This directory is the maintained local execution surface for PointGPT /
pointNEPA wrapper runs.

## Primary Local Entrypoints

- `pointgpt_train_local_ddp.sh`
  - local DDP pretrain wrapper.
- `pointgpt_finetune_local_ddp.sh`
  - local DDP ScanObjectNN fine-tune wrapper.
- `pointgpt_nepa_vs_cdl12_pipeline.sh`
  - sequential PointGPT comparison pipeline.
- `pointgpt_s_vitshift_then_resume_cdl12_watchdog.sh`
  - watchdog wrapper for the PointGPT-S local chain.

## Helpers

- `bootstrap_pointgpt_assets.sh`
  - local asset/bootstrap helper.
- `prepare_pointgpt_smoke_data.sh`
  - smoke-data preparation helper.
- `pointgpt_*`
  - focused recipe, diagnostic, protocol, and status launchers.

## Boundary

These scripts may read or write under `PointGPT/`, `logs/`, `wandb/`, and
`pointnepa/results/`. They do not define PatchNEPA headline benchmark results.
