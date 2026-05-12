# LLM Retrieval Index

Last updated: 2026-05-12

## Purpose

This file defines the default retrieval order for `nepa3d/docs/` so an LLM can
answer current questions without loading raw ledgers or stale March-era routing
by accident.

## Current Mainline Snapshot

- paper-facing direction:
  - derived geometric teacher pretraining for point-context encoders
- historical mainline kept for provenance:
  - PatchNEPA v2 reconstruction `recong2` full300
- current route decision:
  - Route A if matched 100-epoch geo-teacher pretraining is favorable on ScanObjectNN and ShapeNetPart transfer; Route B if the strongest signal remains direct geometry readouts, controls, and completion.
- benchmark status:
  - ScanObjectNN PatchNEPA headline remains pending under official test-as-val; historical file-split FT numbers are internal only.
  - `obj_bg=pending`
  - `obj_only=pending`
  - `pb_t50_rs=pending`
- machine-readable state source:
  - `nepa3d/docs/current_state.json`
- current local execution source:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- human-facing quickstart:
  - `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`

## Tier 0: Default First Read

Read these first for almost any current PatchNEPA / geo-teacher discussion:

1. `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
2. `nepa3d/docs/patch_nepa/claude_context_active.md`
3. `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
4. `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
5. `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
6. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`

## Tier 1: Current Evidence And Operations

Read when the question needs current numbers, route evidence, or execution
boundaries:

- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
- `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`
- `pointnepa/docs/results_scanobjectnn_active.md`
- `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- `nepa3d/docs/operations/README.md`
- `nepa3d/docs/_meta/code_inventory_active.md`
- `nepa3d/docs/_meta/config_inventory_active.md`

## Tier 2: Active Branch Detail / Provenance

Read only when exact branch detail, old reconstruction diagnostics, or job
traceback is required:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`
- `nepa3d/docs/query_nepa/runlog_202602.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`

## Tier 3: Archive / Skip By Default

Do not retrieve these unless the question explicitly asks for legacy planning,
provenance backup, or abandoned ideas:

- everything under `nepa3d/docs/archive/`
- docs named `*_legacy.md`
- raw runlogs when exact job history is not needed

## Task Routing

| question type | minimum docs |
|---|---|
| "What should Claude read first?" | `current_llm_brief_active.md`, `claude_context_active.md` |
| "What is the current paper-facing direction?" | `current_llm_brief_active.md`, `paper_direction_geo_teacher_202604.md` |
| "What changed in the dataset/protocol?" | `claude_context_active.md`, `dataset_geo_teacher_v1_spec.md`, `spec_geo_teacher_vocab_v1.md` |
| "Route A or Route B?" | `experiment_route_ab_matrix_202604.md`, then `itachi/results_geo_teacher_itachi_active.md` if local evidence is needed |
| "Which ScanObjectNN numbers are headline-safe?" | `benchmark_scanobjectnn_variant.md`, `scanobjectnn_ft_policy_audit_active.md` |
| "What should a collaborator run on ABCI?" | `collaborator_reading_guide_active.md`, `scripts/abci/README.md`, `operations/README.md` |
| "What is Itachi-local vs paper-facing?" | `operations/README.md`, `patch_nepa/itachi/README.md`, `itachi/results_geo_teacher_itachi_active.md` |
| "Is PointGPT / pointNEPA a PatchNEPA headline result?" | `pointnepa/docs/results_scanobjectnn_active.md`, then `benchmark_scanobjectnn_variant.md` |
| "Where did this exact claim come from?" | add `runlog_patch_nepa_202602.md` or `query_nepa/runlog_202602.md` |
| "How is code/config ownership organized?" | `_meta/code_inventory_active.md`, `_meta/config_inventory_active.md` |
| "How is docs cleanup organized?" | `_meta/docs_inventory_active.md`, `_meta/docs_cleanup_plan_active.md` |

## Retrieval Rules

- Prefer the current LLM brief and paper-facing geo-teacher docs over March
  reconstruction-era docs.
- Prefer benchmark tables over scattered result mentions.
- Treat Itachi docs as local evidence unless canonical benchmark docs copy the
  result.
- Treat PointGPT / pointNEPA sidecar docs as comparison context, not PatchNEPA
  headline evidence.
- Use archive docs only for traceback, never as default evidence.
- Keep this file synchronized with `nepa3d/docs/current_state.json`.
