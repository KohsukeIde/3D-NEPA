# Results Index

Last updated: 2026-03-14

## Current Snapshot

- current PatchNEPA mainline:
  - PatchNEPA v2 reconstruction `recong2` full300
- current canonical ScanObjectNN headline:
  - `obj_bg=pending`
  - `obj_only=pending`
  - `pb_t50_rs=pending`
- machine-readable state source:
  - `nepa3d/docs/current_state.json`
- first collaborator handoff doc:
  - `patch_nepa/collaborator_reading_guide_active.md`
- benchmark note:
  - previous `file`-split FT headlines are now historical/internal only;
    official `test-as-val` reruns are pending

Use this index to avoid mixing canonical active benchmark docs with legacy ledgers.

## Canonical Active

- Docs hub: `README.md`
- Insight register: `insight_register_active.md`
- Docs cleanup plan: `docs_cleanup_plan_active.md`
- Docs inventory: `docs_inventory_active.md`
- Collaborator reading guide: `patch_nepa/collaborator_reading_guide_active.md`
- LLM retrieval index: `llm_retrieval_index.md`
- Operations boundary: `operations/README.md`
- Query/Patch track split index: `patch_nepa/nepa_tracks_index.md`
- Patch-NEPA folder guide: `patch_nepa/README.md`
- QueryNEPA -> PatchNEPA storyline: `patch_nepa/storyline_query_to_patch_v2_active.md`
- ScanObjectNN FT policy audit: `patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- Patch-NEPA hypothesis matrix: `patch_nepa/hypothesis_matrix_active.md`
- Patch-NEPA local execution backlog: `patch_nepa/execution_backlog_active.md`
- Patch-NEPA Stage-2 active plan: `patch_nepa/patch_nepa_stage2_active.md`
- Patch-NEPA Stage-2 runlog: `patch_nepa/runlog_patch_nepa_202602.md`
- ScanObjectNN canonical benchmark: `patch_nepa/benchmark_scanobjectnn_variant.md`
- Query-NEPA folder guide: `query_nepa/README.md`
- Query-NEPA historical runlog (job history): `query_nepa/runlog_202602.md`
- Non-retrieval results master index: `active/results_master_nonretrieval.md`

## Active (Domain/Task Specific)

- ScanObjectNN core3 tables: `classification/results_scanobjectnn_core3_active.md`
- ScanObjectNN review tables: `classification/results_scanobjectnn_review_active.md`
- ScanObjectNN PointGPT / pointNEPA sidecar ledger: `classification/results_scanobjectnn_pointgpt_pointnepa_active.md`
- ModelNet40 PointGPT-style protocol: `classification/results_modelnet40_pointgpt_active.md`
- UCPR/CPAC active results: `completion/results_ucpr_cpac_active.md`
- UCPR/CPAC table planning: `completion/eccv_ucpr_cpac_tables.md`
- Transfer -> DDA -> ScanObjectNN storyline: `history/ablation_transfer_dda_active.md`

## Legacy / Historical Ledgers

- 1024 multi-node A/B/C/D ledger: `query_nepa/pretrain_abcd_1024_multinode_active.md`
- 1024 variant re-eval ledger: `query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- ScanObjectNN review legacy snapshot: `classification/results_scanobjectnn_review_legacy.md`
- ScanObjectNN M1 legacy snapshot: `classification/results_scanobjectnn_m1_legacy.md`
- ModelNet40 legacy summary: `classification/results_modelnet40_legacy.md`
- Raw archival backup: `history/legacy_full_history.md`

## Policy

- Top-level docs (`README.md`, `llm_retrieval_index.md`, `results_index.md`)
  must stay synchronized with the current mainline and current benchmark
  headline.
- `nepa3d/docs/current_state.json` is the source of truth for the top-level
  current snapshot.
- If a recent update changes local execution policy, reflect that in:
  - `operations/README.md`
  - `patch_nepa/execution_backlog_active.md`
  - and the top-level indexes on this page.
- Start with `llm_retrieval_index.md` when the goal is efficient retrieval.
- Headline benchmark numbers are maintained only in `patch_nepa/benchmark_scanobjectnn_variant.md`.
- Patch-NEPA Stage-2 job updates are maintained in `patch_nepa/runlog_patch_nepa_202602.md`.
- Cross-line interpretation is maintained in `patch_nepa/storyline_query_to_patch_v2_active.md`.
- Hypothesis status is maintained in `patch_nepa/hypothesis_matrix_active.md`.
- Query-NEPA historical job updates are maintained in `query_nepa/runlog_202602.md`.
- Legacy ledgers are kept for traceability and should not be used as primary benchmark source.
