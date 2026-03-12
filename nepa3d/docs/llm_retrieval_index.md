# LLM Retrieval Index

Last updated: 2026-03-12

## Purpose

This file defines the default retrieval order for `nepa3d/docs/` so an LLM can
answer questions without loading raw ledgers unnecessarily.

## Current Mainline Snapshot

- active PatchNEPA line:
  - PatchNEPA v2 reconstruction `recong2` full300
- current canonical ScanObjectNN headline:
  - `obj_bg=0.8485`
  - `obj_only=0.8589`
  - `pb_t50_rs=0.8140`
- machine-readable state source:
  - `nepa3d/docs/current_state.json`
- docs-maintenance inventory:
  - `nepa3d/docs/docs_inventory_active.md`
- current local execution source of truth:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- human-facing quickstart:
  - `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
  - note: this is useful for human handoff, but is not required in default LLM
    Tier 0 retrieval

## Tier 0: Default First Read

Read these first for almost any Patch-NEPA discussion:

1. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
2. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
3. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`

## Tier 1: Active Branch Detail

Read only when the question needs current branch specifics:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`
- `nepa3d/docs/operations/README.md`

## Tier 2: Historical Traceback

Read only when provenance or legacy comparison is explicitly needed:

- `nepa3d/docs/query_nepa/runlog_202602.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- `nepa3d/docs/active/results_master_nonretrieval.md`
- `nepa3d/docs/history/ablation_all_docs_timeline_active.md`

## Tier 3: Archive / Skip By Default

Do not retrieve these unless the question is explicitly about legacy planning,
provenance backup, or abandoned ideas:

- everything under `nepa3d/docs/archive/`
- docs named `*_legacy.md`
- old scratch/reference docs that are linked only from archive or storyline

## Task Routing

| question type | minimum docs |
|---|---|
| "What is currently true for Patch-NEPA?" | `storyline_query_to_patch_v2_active.md`, `hypothesis_matrix_active.md` |
| "What are the exact latest v2 branch findings?" | Tier 0 + `restart_plan_patchnepa_data_v2_20260303.md` |
| "What should run next locally?" | `patch_nepa/execution_backlog_active.md`, `operations/README.md`, then `patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` if needed |
| "Where did this claim come from?" | add `runlog_patch_nepa_202602.md` or `query_nepa/runlog_202602.md` |
| "What is the benchmark headline?" | `benchmark_scanobjectnn_variant.md` |
| "What should I send a collaborator first?" | `patch_nepa/collaborator_reading_guide_active.md`, `patch_nepa/storyline_query_to_patch_v2_active.md`, `patch_nepa/benchmark_scanobjectnn_variant.md` |
| "How is `docs/` organized or what can be archived?" | `docs_inventory_active.md`, `docs_cleanup_plan_active.md`, then folder `README.md` if needed |
| "What old experiments existed?" | Tier 2, then archive only if still needed |

## Retrieval Rules

- Prefer active synthesis docs over raw ledgers.
- Prefer benchmark tables over scattered result mentions.
- Use archive docs only for traceback, never as default evidence.
- Keep this file synchronized with the top-level docs contract in
  `nepa3d/docs/README.md`.
- Keep this file synchronized with `nepa3d/docs/current_state.json`.
- If a question is about Patch-NEPA vs PointGPT, start from
  `storyline_query_to_patch_v2_active.md` before opening raw logs.
