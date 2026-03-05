# Results Master Index (Non-Retrieval)

Last updated: 2026-02-26

## Scope

This page is a single entry point for all docs that contain quantitative results, excluding retrieval metrics as headline.

- Target metrics: classification (`test_acc`), completion (`mae/rmse/iou@tau`), mesh (`chamfer/fscore`)
- Excluded from headline: retrieval metrics (UCPR, retrieval recall/rank metrics)
- Rule: use this page as index only; read each source `.md` for full context and caveats.

## Canonical First Read

| role | doc |
|---|---|
| ScanObjectNN benchmark (canonical) | `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` |
| job-level run history (active) | `nepa3d/docs/query_nepa/runlog_202602.md` |

## Classification Result Docs

| task | status | doc | notes |
|---|---|---|---|
| ScanObjectNN variant benchmark | canonical | `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` | headline `test_acc`, variant-split protocol |
| ScanObjectNN review reruns | active (historical chain) | `nepa3d/docs/classification/results_scanobjectnn_review_active.md` | `v0->v3`, fair/provisional diagnostics |
| ScanObjectNN review baseline | legacy | `nepa3d/docs/classification/results_scanobjectnn_review_legacy.md` | causal-era snapshot |
| ScanObjectNN core3 table | legacy-active snapshot | `nepa3d/docs/classification/results_scanobjectnn_core3_active.md` | pre-bidir reference |
| ScanObjectNN M1 table | legacy | `nepa3d/docs/classification/results_scanobjectnn_m1_legacy.md` | old cache/protocol naming |
| ModelNet40 PointGPT protocol | active baseline page | `nepa3d/docs/classification/results_modelnet40_pointgpt_active.md` | full FT + few-shot LP |
| ModelNet40 legacy summary | legacy | `nepa3d/docs/classification/results_modelnet40_legacy.md` | v0/v1 transfer-era summary |

## Completion / CPAC Result Docs

| task | status | doc | notes |
|---|---|---|---|
| completion A-E/6 track | active | `nepa3d/docs/completion/results_completion_ae6_active.md` | completion-focused ledger |
| UCPR/CPAC main log | active | `nepa3d/docs/completion/results_ucpr_cpac_active.md` | use CPAC/mesh sections only (ignore retrieval rows) |
| UCPR/CPAC plane baselines | active | `nepa3d/docs/completion/results_ucpr_cpac_plane_baselines_active.md` | use CPAC sections only |
| UCPR/CPAC mixed archive | archive | `nepa3d/docs/archive/completion_results_ucpr_cpac_mixed_archive.md` | provenance backup; CPAC sections only |

## Integrated Ledgers (Classification + CPAC mixed history)

| status | doc | notes |
|---|---|---|
| active policy/protocol ledger | `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md` | compact policy + selected result snapshots |
| historical execution ledger | `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md` | includes superseded/failed runs; use with validity boundary |
| broad historical archive | `nepa3d/docs/history/legacy_full_history.md` | mixed-era results and operations |
| active operations runlog | `nepa3d/docs/query_nepa/runlog_202602.md` | latest job outcomes incl. classification/CPAC sections |

## Narrative Docs With Result Snippets

| status | doc | notes |
|---|---|---|
| active storyline memo | `nepa3d/docs/history/ablation_transfer_dda_active.md` | transition summary with selected quantitative checkpoints |
| active completion memo | `nepa3d/docs/completion/plan_completion_ae6.md` | completion-track policy with metric-oriented checkpoints |
| archived idea memo | `nepa3d/docs/archive/history_answer_token_expansion_crazy_ideas.md` | mixed notes including completion-side observed numbers |
| meta timeline | `nepa3d/docs/history/ablation_all_docs_timeline_active.md` | cross-doc timeline; references many result pages |

## Non-Result / Planning Docs (Not part of this index)

These docs are useful but are not primary result tables:

- `nepa3d/docs/completion/eccv_ucpr_cpac_tables.md` (run matrix template)
- `nepa3d/docs/results_index.md` (navigation hub)
- `nepa3d/docs/README.md` (hub page)

## Reading Order (practical)

1. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
2. `nepa3d/docs/query_nepa/runlog_202602.md`
3. task-specific detail page from the tables above
4. legacy pages only when traceback/provenance is needed
