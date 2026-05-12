# Docs Inventory

Last updated: 2026-05-12

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
- added code-organization inventory:
  - `_meta/code_inventory_active.md`
- added config ownership inventory:
  - `_meta/config_inventory_active.md`

## LLM-First Cleanup Completed In This Pass

- added current LLM brief:
  - `patch_nepa/current_llm_brief_active.md`
- promoted April 2026 geo-teacher docs to default current retrieval:
  - `patch_nepa/paper_direction_geo_teacher_202604.md`
  - `patch_nepa/dataset_geo_teacher_v1_spec.md`
  - `patch_nepa/experiment_route_ab_matrix_202604.md`
- demoted March reconstruction-era plan surfaces to historical/provenance
  unless explicitly requested:
  - `patch_nepa/patch_nepa_stage2_active.md`
  - `patch_nepa/nepa_tracks_index.md`
  - `patch_nepa/hypothesis_matrix_active.md`
- refreshed top-level routing:
  - `README.md`
  - `llm_retrieval_index.md`
  - `results_index.md`
  - `current_state.json`

## Root Control Plane

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `README.md` | docs hub / top-level entrypoint | `tier0` | `keep` | self | must stay synced with `current_state.json` |
| `llm_retrieval_index.md` | default retrieval contract | `tier0` | `keep` | self | main LLM routing surface |
| `results_index.md` | top-level result navigation | `tier0` | `keep` | self | human-facing index |
| `current_state.json` | machine-readable current snapshot | `n/a` | `keep` | self | source of truth for top-level sync checks |

No other loose root files should be added unless they become part of this
control plane.

## `_meta/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `_meta/README.md` | meta folder guide | `tier1` | `keep` | self | explains why meta docs are not loose root files |
| `_meta/code_inventory_active.md` | canonical code-organization boundary | `tier1` | `keep` | self | track/shared/compat/script boundary doc |
| `_meta/config_inventory_active.md` | canonical top-level config ownership inventory | `tier1` | `keep` | self | owner / compat / migration-risk boundary for `nepa3d/configs/` |
| `_meta/insight_register_active.md` | experiment-family insight register | `tier1` | `keep` | self | refreshed around geo-teacher / CQA / sidecar insights |
| `_meta/docs_cleanup_plan_active.md` | cleanup policy / governance | `tier1` | `keep` | self | keep until cleanup converges |
| `_meta/docs_inventory_active.md` | full docs inventory | `n/a` | `keep` | self | inventory source for future merge/archive steps |

## `archive/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `archive/README.md` | archive policy | `archive` | `keep` | self | explains archive usage boundary |
| `archive/completion_results_ucpr_cpac_mixed_archive.md` | frozen completion provenance | `archive` | `archive` | `completion/results_ucpr_cpac_active.md` | already archived |
| `archive/history_answer_token_expansion_crazy_ideas.md` | abandoned idea memo | `archive` | `archive` | `history/README.md` | already archived |
| `archive/patch_nepa_baseline_patchcls_scratch_reference.md` | frozen PatchCls scratch reference | `archive` | `archive` | `patch_nepa/benchmark_scanobjectnn_variant.md` | already archived |
| `archive/patch_nepa_cls_raypatch_migration_plan_legacy.md` | legacy migration plan | `archive` | `archive` | `patch_nepa/patch_nepa_stage2_active.md` | already archived |
| `archive/patch_nepa_scratch_to_patch_comparison_reference.md` | frozen scratch-vs-PatchNEPA comparison matrix | `archive` | `archive` | `patch_nepa/benchmark_scanobjectnn_variant.md`, `patch_nepa/storyline_query_to_patch_v2_active.md` | archived in this pass |
| `archive/results_master_nonretrieval_20260226.md` | superseded non-retrieval result index | `archive` | `archive` | `results_index.md` | `active/` absorbed and removed in this pass |

## `classification/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `classification/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `classification/results_scanobjectnn_core3_active.md` | active historical ScanObjectNN table | `tier2` | `review` | maybe merge selected rows into `patch_nepa/benchmark_scanobjectnn_variant.md` | still useful, but not headline surface |
| `classification/results_scanobjectnn_review_active.md` | active review ledger | `tier2` | `keep` | self | still used for review-era traceback |
| `classification/results_scanobjectnn_review_legacy.md` | legacy review snapshot | `archive` | `archive` | `classification/results_scanobjectnn_review_active.md` | likely next archive candidate |
| `classification/results_scanobjectnn_m1_legacy.md` | legacy M1 snapshot | `archive` | `archive` | `classification/results_scanobjectnn_review_active.md` | likely next archive candidate |
| `classification/results_modelnet40_legacy.md` | legacy ModelNet40 summary | `archive` | `archive` | `pointnepa/docs/results_modelnet40_active.md` | likely next archive candidate |

## External Sidecars

These docs are intentionally outside `nepa3d/docs/` because they belong to a
wrapper sidecar, not the PatchNEPA / CQA / geo-teacher docs tree.

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `pointnepa/README.md` | pointNEPA / PointGPT sidecar entrypoint | `tier1` | `keep` | self | wrapper workspace; depends on sibling `PointGPT/` |
| `pointnepa/docs/README.md` | sidecar docs guide | `tier1` | `keep` | self | routes sidecar result ledgers |
| `pointnepa/docs/results_scanobjectnn_active.md` | PointGPT / pointNEPA sidecar ledger | `tier1` | `keep` | self | active comparison context, not PatchNEPA headline |
| `pointnepa/docs/results_modelnet40_active.md` | active ModelNet40 PointGPT-style baseline | `tier2` | `keep` | self | external control page |
| `pointnepa/docs/pointgpt_unique_retained_diagnostics_summary.md` | unique-retained diagnostic summary | `tier2` | `keep` | `pointnepa/docs/results_scanobjectnn_active.md` | focused sidecar diagnostic page |

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
| `patch_nepa/current_llm_brief_active.md` | current LLM brief | `tier0` | `keep` | self | shortest current truth / do-not-use / routing surface |
| `patch_nepa/claude_context_active.md` | compact Claude/LLM handoff | `tier0` | `keep` | self | excludes QueryNEPA by default and summarizes dataset/protocol findings |
| `patch_nepa/paper_direction_geo_teacher_202604.md` | paper-facing direction | `tier0` | `keep` | self | current geo-teacher story source |
| `patch_nepa/dataset_geo_teacher_v1_spec.md` | paper-facing dataset/protocol spec | `tier0` | `keep` | self | current split/task/protocol semantics |
| `patch_nepa/experiment_route_ab_matrix_202604.md` | Route A/B decision matrix | `tier0` | `keep` | self | current matched-compare decision rule |
| `patch_nepa/storyline_query_to_patch_v2_active.md` | cross-line storyline | `tier0` | `keep` | self | historical trajectory plus current interpretation |
| `patch_nepa/benchmark_scanobjectnn_variant.md` | canonical benchmark table | `tier1` | `keep` | self | headline benchmark source; current PatchNEPA headline pending |
| `patch_nepa/collaborator_reading_guide_active.md` | collaborator quickstart | `tier1` | `keep` | self | human handoff page |
| `patch_nepa/hypothesis_matrix_geo_teacher_v1.md` | geo-teacher hypothesis matrix | `tier1` | `keep` | self | current paper-facing hypotheses |
| `patch_nepa/migration_cross_primitive_to_geo_teacher_202604.md` | terminology migration memo | `tier1` | `keep` | self | old-claim to new-claim mapping |
| `patch_nepa/spec_geo_teacher_vocab_v1.md` | paper-facing vocabulary spec | `tier1` | `keep` | self | maps paper tasks to runtime CQA tasks |
| `patch_nepa/spec_world_v3_schema.md` | raw cache schema spec | `tier1` | `keep` | self | raw-layer contract below geo-teacher protocol |
| `patch_nepa/diagcopy_probe_100755_gap_copywin.tsv` | one-off diagnostic table | `tier2` | `review` | maybe absorb summary into `hypothesis_matrix_active.md`, then archive | not needed for default retrieval |
| `patch_nepa/execution_backlog_active.md` | local execution source of truth | `tier1` | `keep` | self | current next-run/gating page |
| `patch_nepa/gap_audit_query_to_patch_active.md` | Query->Patch porting audit | `tier2` | `keep` | self | parity traceback, not default current truth |
| `patch_nepa/hypothesis_matrix_active.md` | older reconstruction-era hypothesis surface | `tier1` | `keep` | `patch_nepa/hypothesis_matrix_geo_teacher_v1.md` for paper-facing hypotheses | historical/mixed banner added |
| `patch_nepa/latent_diag_snapshot_20260304.tsv` | one-off latent diagnostic table | `tier2` | `review` | maybe absorb summary into `hypothesis_matrix_active.md`, then archive | not needed for default retrieval |
| `patch_nepa/nepa_tracks_index.md` | Query/Patch split index | `tier2` | `keep` | `_meta/code_inventory_active.md` for current code ownership | historical/mixed banner added |
| `patch_nepa/patch_nepa_stage2_active.md` | March Stage-2 plan / policy memo | `tier2` | `review` | `current_llm_brief_active.md`, `paper_direction_geo_teacher_202604.md`, `experiment_route_ab_matrix_202604.md` | historical/stale banner added |
| `patch_nepa/patchcls_completed_results.tsv` | historical PatchCls result table | `tier2` | `review` | keep as raw table; maybe archive later | still useful for scratch/reference lookup |
| `patch_nepa/patchcls_exhaustive_audit.tsv` | historical PatchCls audit table | `tier2` | `review` | keep as raw table; maybe archive later | provenance-heavy |
| `patch_nepa/patchnepa_ft_completed_results.tsv` | active FT result table | `tier2` | `keep` | self | current raw table backing some summaries |
| `patch_nepa/patchnepa_ft_exhaustive_audit.tsv` | FT coverage audit table | `tier2` | `keep` | self | still useful for exhaustive lookup |
| `patch_nepa/query_nepa_chronology_audit_202602_active.md` | Query-era chronology audit for Patch line | `tier2` | `keep` | self | inherited-claim traceback |
| `patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` | heavy branch memo | `tier2` | `keep` | self | detailed reconstruction/CQA provenance only |
| `patch_nepa/runlog_patch_nepa_202602.md` | raw PatchNEPA ledger | `tier2` | `keep` | self | canonical raw execution record |
| `patch_nepa/scanobjectnn_ft_policy_audit_active.md` | ScanObjectNN FT policy boundary audit | `tier1` | `keep` | self | authoritative list of file-split historical rows |
| `patch_nepa/cqa_external_baseline_plan.md` | external CQA baseline plan | `tier2` | `keep` | self | useful for external-control provenance |
| `patch_nepa/p4_udfdist_kplane_worldv3_note.md` | k-plane CQA bridge note | `tier2` | `keep` | self | side baseline note |
| `patch_nepa/itachi/README.md` | Itachi local notes guide | `tier1` | `keep` | self | local-only boundary |
| `patch_nepa/itachi/local_data_ops_202604.md` | Itachi local data ops | `tier1` | `keep` | self | local-only operations |
| `patch_nepa/itachi/local_geo_teacher_pretrain_ops_202604.md` | Itachi pretrain ops | `tier1` | `keep` | self | local-only operations |
| `patch_nepa/itachi/local_geo_teacher_posttrain_ops_202604.md` | Itachi post-train ops | `tier1` | `keep` | self | local-only operations |
| `patch_nepa/itachi/local_geopcp_pcpmae_ops_202604.md` | Itachi Geo-PCP / PCP-MAE ops | `tier1` | `keep` | self | local-only Route-A engine note |
| `patch_nepa/itachi/results_geo_teacher_itachi_active.md` | Itachi-local geo-teacher results | `tier1` | `keep` | benchmark docs if promoted | local evidence, not paper headline by default |

## `query_nepa/`

| path | role | retrieval | status | canonical target / action | note |
|---|---|---|---|---|---|
| `query_nepa/README.md` | folder guide | `tier2` | `keep` | self | lightweight entrypoint |
| `query_nepa/pretrain_abcd_1024_multinode_active.md` | large historical ledger | `tier2` | `review` | maybe archive later; currently keep in place for traceability | already treated as archive-style in `archive/README.md` |
| `query_nepa/pretrain_abcd_1024_variant_reval_active.md` | historical policy/re-eval ledger | `tier2` | `review` | maybe archive later; currently keep in place for traceability | already treated as archive-style in `archive/README.md` |
| `query_nepa/runlog_202602.md` | raw QueryNEPA ledger | `tier2` | `keep` | self | canonical historical job record |

## Immediate Next Cleanup Targets

These are the next conservative targets after this pass:

1. `patch_nepa/patch_nepa_stage2_active.md`
   - compare against `patch_nepa/execution_backlog_active.md` and
     `operations/README.md`
2. `patch_nepa/diagcopy_probe_100755_gap_copywin.tsv`
3. `patch_nepa/latent_diag_snapshot_20260304.tsv`
4. legacy `classification/*_legacy.md`

## Operating Rule

When a doc changes role, update all three:

1. `_meta/docs_inventory_active.md`
2. `_meta/docs_cleanup_plan_active.md`
3. the nearest folder `README.md`
