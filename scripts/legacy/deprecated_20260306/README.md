# Deprecated Scripts Archived On 2026-03-06

These scripts were moved out of active `scripts/finetune/` after an audit
against the current docs/code state.

Why these were archived:

- they were only referenced from historical docs, not current active pipelines
- their default launch targets under active `scripts/` no longer existed
- the historical local-chain targets already live under
  `scripts/legacy/local_chains_20260226/`

Moved files:

- `finetune/launch_chain_shapenet_to_main.sh`
- `finetune/launch_m1_pipeline_after_shapenet_table.sh`
- `finetune/launch_scan_main_after_shapenet_table.sh`
- `finetune/launch_scanobjectnn_m1_after_pretrain.sh`
- `finetune/run_m1_pipeline_after_shapenet_table.sh`
- `finetune/run_scanobjectnn_m1_after_pretrain.sh`

Post-archive handling:

- internal cross-calls now stay inside this archive folder
- archived M1 helpers default to the existing
  `scripts/legacy/local_chains_20260226/` launchers
- active docs should reference either current scheduler entrypoints or these
  archived paths explicitly
