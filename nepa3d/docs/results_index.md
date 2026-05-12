# Results Index

Last updated: 2026-05-12

## Current Snapshot

- paper-facing direction:
  - derived geometric teacher pretraining for point-context encoders
- historical mainline kept for provenance:
  - PatchNEPA v2 reconstruction `recong2` full300
- current benchmark status:
  - ScanObjectNN PatchNEPA headline remains pending under official test-as-val; historical file-split FT numbers are internal only.
  - `obj_bg=pending`
  - `obj_only=pending`
  - `pb_t50_rs=pending`
- machine-readable state source:
  - `nepa3d/docs/current_state.json`

Use this index to avoid mixing canonical benchmark docs, Itachi-local evidence,
PointGPT sidecar results, and legacy ledgers.

## Scope

This page is the canonical non-retrieval result index.

- headline metrics:
  - classification `test_acc`
  - completion `mae`, `rmse`, `iou@tau`
  - mesh `chamfer`, `fscore`
- retrieval metrics are provenance/supporting context, not headline result rows
- read source docs for full protocol and caveats before quoting a number

## Canonical Active

- Docs hub: `README.md`
- LLM retrieval index: `llm_retrieval_index.md`
- Current LLM brief: `patch_nepa/current_llm_brief_active.md`
- Paper direction: `patch_nepa/paper_direction_geo_teacher_202604.md`
- Dataset / protocol semantics: `patch_nepa/dataset_geo_teacher_v1_spec.md`
- Route A/B matrix: `patch_nepa/experiment_route_ab_matrix_202604.md`
- Collaborator reading guide: `patch_nepa/collaborator_reading_guide_active.md`
- Operations boundary: `operations/README.md`
- Local execution backlog: `patch_nepa/execution_backlog_active.md`
- Insight register: `_meta/insight_register_active.md`
- Docs cleanup plan: `_meta/docs_cleanup_plan_active.md`
- Docs inventory: `_meta/docs_inventory_active.md`
- Code inventory: `_meta/code_inventory_active.md`
- Config inventory: `_meta/config_inventory_active.md`

## Benchmark And Result Surfaces

- ScanObjectNN canonical PatchNEPA benchmark:
  - `patch_nepa/benchmark_scanobjectnn_variant.md`
- ScanObjectNN FT policy audit:
  - `patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- Itachi-local geo-teacher results:
  - `patch_nepa/itachi/results_geo_teacher_itachi_active.md`
  - local evidence only unless copied into canonical benchmark docs.
- PointGPT / pointNEPA sidecar results:
  - `classification/results_scanobjectnn_pointgpt_pointnepa_active.md`
  - comparison context, not a PatchNEPA headline surface.
- Historical PatchNEPA storyline:
  - `patch_nepa/storyline_query_to_patch_v2_active.md`
- Geo-teacher hypothesis surface:
  - `patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
- Older mixed PatchNEPA hypothesis surface:
  - `patch_nepa/hypothesis_matrix_active.md`

## Domain-Specific Ledgers

- ScanObjectNN review tables:
  - `classification/results_scanobjectnn_review_active.md`
- ScanObjectNN core3 historical active table:
  - `classification/results_scanobjectnn_core3_active.md`
- ModelNet40 PointGPT-style protocol:
  - `classification/results_modelnet40_pointgpt_active.md`
- UCPR/CPAC active results:
  - `completion/results_ucpr_cpac_active.md`
- UCPR/CPAC plane baselines:
  - `completion/results_ucpr_cpac_plane_baselines_active.md`
- Completion AE6 active ledger:
  - `completion/results_completion_ae6_active.md`
- Completion mixed archive:
  - `archive/completion_results_ucpr_cpac_mixed_archive.md`
  - provenance backup; use CPAC/mesh sections only

## Legacy / Historical Ledgers

- Query-NEPA historical runlog:
  - `query_nepa/runlog_202602.md`
- Query-NEPA 1024 multi-node A/B/C/D ledger:
  - `query_nepa/pretrain_abcd_1024_multinode_active.md`
- Query-NEPA 1024 variant re-eval ledger:
  - `query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- Archived non-retrieval master index:
  - `archive/results_master_nonretrieval_20260226.md`
  - superseded by this file; kept only for traceback
- Legacy ScanObjectNN / ModelNet pages:
  - `classification/results_scanobjectnn_review_legacy.md`
  - `classification/results_scanobjectnn_m1_legacy.md`
  - `classification/results_modelnet40_legacy.md`
- Raw archival backup:
  - `history/legacy_full_history.md`

## Narrative Docs With Result Snippets

These are useful for context, but they are not primary result tables:

- Historical transition summary:
  - `history/ablation_transfer_dda_active.md`
- Completion-track planning memo:
  - `completion/plan_completion_ae6.md`
- Cross-doc timeline:
  - `history/ablation_all_docs_timeline_active.md`

## Policy

- Start from `llm_retrieval_index.md` when the goal is efficient retrieval.
- Headline ScanObjectNN PatchNEPA numbers live only in
  `patch_nepa/benchmark_scanobjectnn_variant.md`.
- Historical file-split FT rows stay historical/internal until official
  test-as-val reruns are canonized.
- Itachi-local results stay local evidence unless copied into benchmark-facing
  tables.
- PointGPT / pointNEPA sidecar results should not be quoted as PatchNEPA
  benchmark headlines.
- Raw runlogs are provenance, not default result surfaces.
