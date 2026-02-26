# NEPA3D Docs Hub

Last updated: 2026-02-26

## Start Here

- Protocol-correct ScanObjectNN benchmark (canonical): `nepa3d/docs/active/benchmark_scanobjectnn_variant.md`
- Current execution log (job-level): `nepa3d/docs/active/runlog_202602.md`

## Legacy Ledgers

- Historical 1024 multi-node ledger: `nepa3d/docs/pretrain_abcd_1024_multinode_active.md`
- Historical variant re-eval ledger: `nepa3d/docs/pretrain_abcd_1024_variant_reval_active.md`

## Update Policy

- Put headline benchmark numbers only in `active/benchmark_scanobjectnn_variant.md`.
- Put job IDs, retry history, and failure details only in `active/runlog_202602.md`.
- Keep `test_acc` as headline metric for ScanObjectNN benchmark tables.
- Keep `best_val` and `best_ep` as diagnostics (not headline).
- Keep protocol metadata for every table row:
  - checkpoint path
  - `SCAN_CACHE_ROOT`
  - pretrain family (`fps`, `rfps`, etc.)
  - job ID

## Notes

- `scanobjectnn_main_split_v2` mixed-cache reporting is not used for fair benchmark headline.
- Variant-split caches (`obj_bg`, `obj_only`, `pb_t50_rs`) are the default benchmark protocol.
- Raw log pruning helper:
  - `scripts/logs/prune_superseded_logs.sh` (`--dry-run` / `--apply`)
