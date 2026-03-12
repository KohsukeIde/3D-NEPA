# Docs Inventory

Last updated: 2026-03-12

## Purpose

This file is the phase-2 inventory for `nepa3d/docs/`.

It records, for every doc currently under `docs/`:

- stable role
- retrieval tier
- cleanup status
- canonical replacement or next action

## Status Legend

- `keep`: canonical or still-needed doc; no structural move now
- `merge`: content should be absorbed into another canonical doc
- `archive`: should live in `docs/archive/` or already does
- `review`: keep for now, but revisit after nearby canonical docs stabilize

## Retrieval Legend

- `tier0`: default first-read doc
- `tier1`: active detail doc
- `tier2`: historical/provenance doc
- `archive`: excluded from default retrieval
- `n/a`: system or maintenance file

## Phase-2 Cleanup Completed In This Pass

- archived `patch_nepa/comparison_scratch_to_patchnepa.md`
- new archived location:
  - `archive/patch_nepa_scratch_to_patch_comparison_reference.md`
- active replacement for scratch-vs-PatchNEPA headline comparison:
  - `patch_nepa/benchmark_scanobjectnn_variant.md`
  - `patch_nepa/storyline_query_to_patch_v2_active.md`

## Root Docs

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `README.md` | docs hub / top-level entrypoint | `tier0` | `keep` | self | must stay synced with `current_state.json` |
| `llm_retrieval_index.md` | default retrieval contract | `tier0` | `keep` | self | main LLM routing surface |
| `results_index.md` | top-level result navigation | `tier0` | `keep` | self | human-facing index |
| `current_state.json` | machine-readable current snapshot | `n/a` | `keep` | self | source of truth for top-level sync checks |
| `insight_register_active.md` | experiment-family insight register | `tier1` | `keep` | self | should accumulate "what became newly known" |
| `docs_cleanup_plan_active.md` | cleanup policy / governance | `tier1` | `keep` | self | keep until cleanup converges |
| `docs_inventory_active.md` | full docs inventory | `n/a` | `keep` | self | inventory source for future merge/archive steps |

## `active/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `active/README.md` | folder note | `tier2` | `keep` | self | low-cost pointer page |
| `active/results_master_nonretrieval.md` | older cross-track result index | `tier2` | `merge` | `results_index.md` | overlaps with top-level result navigation; keep until merged cleanly |

## `archive/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `archive/README.md` | archive policy | `archive` | `keep` | self | explains archive usage boundary |
| `archive/completion_results_ucpr_cpac_mixed_archive.md` | frozen completion provenance | `archive` | `archive` | `completion/results_ucpr_cpac_active.md` | already archived |
| `archive/history_answer_token_expansion_crazy_ideas.md` | abandoned idea memo | `archive` | `archive` | `history/README.md` | already archived |
| `archive/patch_nepa_baseline_patchcls_scratch_reference.md` | frozen PatchCls scratch reference | `archive` | `archive` | `patch_nepa/benchmark_scanobjectnn_variant.md` | already archived |
| `archive/patch_nepa_cls_raypatch_migration_plan_legacy.md` | legacy migration plan | `archive` | `archive` | `patch_nepa/patch_nepa_stage2_active.md` | already archived |
| `archive/patch_nepa_scratch_to_patch_comparison_reference.md` | frozen scratch-vs-PatchNEPA comparison matrix | `archive` | `archive` | `patch_nepa/benchmark_scanobjectnn_variant.md`, `patch_nepa/storyline_query_to_patch_v2_active.md` | archived in this pass |

## `classification/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `classification/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `classification/results_scanobjectnn_core3_active.md` | active historical ScanObjectNN table | `tier2` | `review` | maybe merge selected rows into `patch_nepa/benchmark_scanobjectnn_variant.md` | still useful, but not headline surface |
| `classification/results_scanobjectnn_review_active.md` | active review ledger | `tier2` | `keep` | self | still used for review-era traceback |
| `classification/results_scanobjectnn_review_legacy.md` | legacy review snapshot | `archive` | `archive` | `classification/results_scanobjectnn_review_active.md` | likely next archive candidate |
| `classification/results_scanobjectnn_m1_legacy.md` | legacy M1 snapshot | `archive` | `archive` | `classification/results_scanobjectnn_review_active.md` | likely next archive candidate |
| `classification/results_modelnet40_pointgpt_active.md` | active ModelNet40 PointGPT-style baseline | `tier2` | `keep` | self | still useful external control page |
| `classification/results_modelnet40_legacy.md` | legacy ModelNet40 summary | `archive` | `archive` | `classification/results_modelnet40_pointgpt_active.md` | likely next archive candidate |

## `completion/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `completion/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `completion/eccv_ucpr_cpac_tables.md` | table-planning scaffold | `tier2` | `keep` | self | task-domain table rather than narrative memo |
| `completion/plan_completion_ae6.md` | active completion planning memo | `tier2` | `review` | maybe merge selected policy notes into `completion/results_completion_ae6_active.md` | keep until completion side is cleaned |
| `completion/results_completion_ae6_active.md` | active completion ledger | `tier2` | `keep` | self | canonical AE6 result page |
| `completion/results_ucpr_cpac_active.md` | active CPAC/UCPR ledger | `tier2` | `keep` | self | canonical active completion ledger |
| `completion/results_ucpr_cpac_plane_baselines_active.md` | active plane-baseline ledger | `tier2` | `keep` | self | separate baseline surface still useful |

## `history/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `history/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `history/ablation_all_docs_timeline_active.md` | cross-doc chronology | `tier2` | `keep` | self | useful for provenance and doc evolution |
| `history/ablation_transfer_dda_active.md` | active historical narrative | `tier2` | `keep` | self | still referenced as a cross-era narrative |
| `history/legacy_full_history.md` | broad legacy archive | `archive` | `archive` | `history/README.md` | should remain provenance-only |

## `operations/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `operations/README.md` | execution-surface boundary | `tier1` | `keep` | self | canonical local vs ABCI ops boundary |

## `patch_nepa/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `patch_nepa/README.md` | folder guide | `tier1` | `keep` | self | main folder entrypoint |
| `patch_nepa/benchmark_scanobjectnn_variant.md` | canonical benchmark table | `tier0` | `keep` | self | headline benchmark source |
| `patch_nepa/collaborator_reading_guide_active.md` | collaborator quickstart | `tier1` | `keep` | self | human handoff page |
| `patch_nepa/diagcopy_probe_100755_gap_copywin.tsv` | one-off diagnostic table | `tier2` | `review` | maybe absorb summary into `hypothesis_matrix_active.md`, then archive | not needed for default retrieval |
| `patch_nepa/execution_backlog_active.md` | local execution source of truth | `tier1` | `keep` | self | current next-run/gating page |
| `patch_nepa/gap_audit_query_to_patch_active.md` | Query->Patch porting audit | `tier1` | `keep` | self | still used for parity traceback |
| `patch_nepa/hypothesis_matrix_active.md` | canonical hypothesis surface | `tier0` | `keep` | self | core synthesis doc |
| `patch_nepa/latent_diag_snapshot_20260304.tsv` | one-off latent diagnostic table | `tier2` | `review` | maybe absorb summary into `hypothesis_matrix_active.md`, then archive | not needed for default retrieval |
| `patch_nepa/nepa_tracks_index.md` | Query/Patch split index | `tier1` | `keep` | self | important boundary page |
| `patch_nepa/patch_nepa_stage2_active.md` | older active plan / policy memo | `tier1` | `review` | compare against `execution_backlog_active.md` and `operations/README.md` for possible consolidation | still referenced widely |
| `patch_nepa/patchcls_completed_results.tsv` | historical PatchCls result table | `tier2` | `review` | keep as raw table; maybe archive later | still useful for scratch/reference lookup |
| `patch_nepa/patchcls_exhaustive_audit.tsv` | historical PatchCls audit table | `tier2` | `review` | keep as raw table; maybe archive later | provenance-heavy |
| `patch_nepa/patchnepa_ft_completed_results.tsv` | active FT result table | `tier2` | `keep` | self | current raw table backing some summaries |
| `patch_nepa/patchnepa_ft_exhaustive_audit.tsv` | FT coverage audit table | `tier2` | `keep` | self | still useful for exhaustive lookup |
| `patch_nepa/query_nepa_chronology_audit_202602_active.md` | Query-era chronology audit for Patch line | `tier1` | `keep` | self | still used to qualify inherited claims |
| `patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` | heavy active branch memo | `tier1` | `keep` | self | current-branch detail only |
| `patch_nepa/runlog_patch_nepa_202602.md` | raw PatchNEPA ledger | `tier1` | `keep` | self | canonical raw execution record |
| `patch_nepa/storyline_query_to_patch_v2_active.md` | cross-line storyline | `tier0` | `keep` | self | default synthesis doc |

## `query_nepa/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `query_nepa/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `query_nepa/pretrain_abcd_1024_multinode_active.md` | large historical ledger | `tier2` | `review` | maybe archive later; currently keep in place for traceability | already treated as archive-style in `archive/README.md` |
| `query_nepa/pretrain_abcd_1024_variant_reval_active.md` | historical policy/re-eval ledger | `tier2` | `review` | maybe archive later; currently keep in place for traceability | already treated as archive-style in `archive/README.md` |
| `query_nepa/runlog_202602.md` | raw QueryNEPA ledger | `tier2` | `keep` | self | canonical historical job record |

## Immediate Next Cleanup Targets

These are the next conservative targets after this pass:

1. `active/results_master_nonretrieval.md`
   - likely merge into `results_index.md`
2. `patch_nepa/patch_nepa_stage2_active.md`
   - compare against `patch_nepa/execution_backlog_active.md` and
     `operations/README.md`
3. `patch_nepa/diagcopy_probe_100755_gap_copywin.tsv`
4. `patch_nepa/latent_diag_snapshot_20260304.tsv`
5. legacy `classification/*_legacy.md`

## Operating Rule

When a doc changes role, update all three:

1. `docs_inventory_active.md`
2. `docs_cleanup_plan_active.md`
3. the nearest folder `README.md`
