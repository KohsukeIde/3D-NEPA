# LLM Retrieval Index

Last updated: 2026-03-06

## Purpose

This file defines the default retrieval order for `nepa3d/docs/` so an LLM can
answer questions without loading raw ledgers unnecessarily.

## Tier 0: Default First Read

Read these first for almost any Patch-NEPA discussion:

1. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
2. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
3. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`

## Tier 1: Active Branch Detail

Read only when the question needs current branch specifics:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

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
| "Where did this claim come from?" | add `runlog_patch_nepa_202602.md` or `query_nepa/runlog_202602.md` |
| "What is the benchmark headline?" | `benchmark_scanobjectnn_variant.md` |
| "What old experiments existed?" | Tier 2, then archive only if still needed |

## Retrieval Rules

- Prefer active synthesis docs over raw ledgers.
- Prefer benchmark tables over scattered result mentions.
- Use archive docs only for traceback, never as default evidence.
- If a question is about Patch-NEPA vs PointGPT, start from
  `storyline_query_to_patch_v2_active.md` before opening raw logs.
