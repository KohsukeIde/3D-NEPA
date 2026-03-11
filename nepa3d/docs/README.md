# NEPA3D Docs Hub

Last updated: 2026-03-06

## Start Here (Current)

- Collaborator reading guide:
  - `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
- Operations boundary (local vs ABCI):
  - `nepa3d/docs/operations/README.md`
- LLM retrieval index:
  - `nepa3d/docs/llm_retrieval_index.md`
- Track split index (Query vs Patch):
  - `nepa3d/docs/patch_nepa/nepa_tracks_index.md`
- Patch-NEPA folder guide:
  - `nepa3d/docs/patch_nepa/README.md`
- QueryNEPA -> PatchNEPA -> external baseline storyline:
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- Patch-NEPA hypothesis matrix:
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- Patch-NEPA Stage-2 active plan:
  - `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- Patch-NEPA Stage-2 runlog:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Patch-NEPA Query->Patch gap audit:
  - `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- Query chronology audit for Patch porting:
  - `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

- Protocol-correct ScanObjectNN benchmark (canonical): `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- Query-NEPA historical runlog (job-level): `nepa3d/docs/query_nepa/runlog_202602.md`
- Non-retrieval results master index: `nepa3d/docs/active/results_master_nonretrieval.md`

## Default Retrieval Policy

- For most Patch-NEPA questions, read only:
  - `nepa3d/docs/llm_retrieval_index.md`
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
  - `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- Read `restart_plan_patchnepa_data_v2_20260303.md` only when current-branch
  detail is required.
- Read raw runlogs only when provenance or exact launch history is required.
- Do not retrieve `nepa3d/docs/archive/` by default.

## Legacy Ledgers

- Historical 1024 multi-node ledger: `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
- Historical variant re-eval ledger: `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`

## Folder Map

- `nepa3d/docs/operations/`: execution-surface boundary and operational entrypoint guide
- `nepa3d/docs/query_nepa/`: Query-NEPA historical ledgers, runlog, and folder guide
- `nepa3d/docs/patch_nepa/`: Patch-NEPA active plan/runlog plus synthesis docs
- `nepa3d/docs/classification/`: ScanObjectNN / ModelNet result ledgers
- `nepa3d/docs/completion/`: CPAC/UCPR/completion ledgers and planning
- `nepa3d/docs/history/`: cross-era timeline/memos/legacy consolidated history
- `nepa3d/docs/active/`: cross-track master index only
- `nepa3d/docs/archive/`: frozen low-priority docs excluded from default retrieval

## Update Policy

- Put headline benchmark numbers only in `patch_nepa/benchmark_scanobjectnn_variant.md`.
- Put Patch-NEPA Stage-2 job history in `patch_nepa/runlog_patch_nepa_202602.md`.
- Put cross-line conclusions in `patch_nepa/storyline_query_to_patch_v2_active.md`.
- Put active hypothesis status in `patch_nepa/hypothesis_matrix_active.md`.
- Keep Query-NEPA historical job history in `query_nepa/runlog_202602.md`.
- Put local-vs-ABCI operational boundary notes in `operations/README.md`.
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
- Local derived-cache cleanup (2026-03-06):
  - removed from this workspace to prevent accidental reuse:
    - `data/scanobjectnn_cache_v2`
    - `data/shapenet_cache_v0`
    - `data/shapenet_unpaired_cache_v1`
    - `data/shapenet_unpaired_splits_v1.json`
  - source dataset roots were kept
  - current mainline requires regenerating canonical caches before running:
    - ScanObjectNN: `data/scanobjectnn_obj_bg_v3_nonorm`, `data/scanobjectnn_obj_only_v3_nonorm`, `data/scanobjectnn_pb_t50_rs_v3_nonorm`
    - ShapeNet: `data/shapenet_cache_v2_20260303`, `data/shapenet_unpaired_cache_v2_20260303` and any active v2 variants
- Raw log pruning helper:
  - `scripts/logs/prune_superseded_logs.sh` (`--dry-run` / `--apply`)
