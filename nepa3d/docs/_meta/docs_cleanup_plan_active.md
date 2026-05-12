# Docs Cleanup Plan

Last updated: 2026-05-12

## Purpose

This file defines how to simplify `nepa3d/docs/` without losing scientific
traceability.

The goal is not to delete history. The goal is to make retrieval cheaper and
human reading faster.

## Core Principle

Do not organize docs by when they were written. Organize them by role.

Every doc should clearly fall into one of these buckets:

1. `entrypoint`
2. `canonical synthesis`
3. `active branch memo`
4. `raw ledger`
5. `task/domain table`
6. `archive`

If a file does not have a stable role, it should be merged, demoted, or
archived.

## Keep / Merge / Archive Rules

### Keep

Keep a doc as-is only if it is the canonical home of unique information.

Examples:

- `nepa3d/docs/README.md`
- `nepa3d/docs/llm_retrieval_index.md`
- `nepa3d/docs/results_index.md`
- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/execution_backlog_active.md`

### Merge

Merge when two docs have the same scope but different granularity and the
reader should not need both for the same question.

Typical merge candidates:

- small comparison memos whose conclusions are already absorbed by
  `storyline_query_to_patch_v2_active.md`
- small result summaries whose headline numbers already live in
  `benchmark_scanobjectnn_variant.md`
- operational notes duplicated between folder READMEs and task-specific docs

### Archive

Archive when a doc is still useful for traceback, but should never be part of
default retrieval.

Archive criteria:

- superseded planning memo
- old reference baseline note
- abandoned idea memo
- legacy snapshot duplicated by a newer synthesis or benchmark page

## Current High-Level Decision

The current cleanup pass is LLM-first and conservative:

- keep history in place
- make current paper-facing truth unambiguous
- demote stale March-era plan surfaces through banners and retrieval tiers
- group docs-governance and inventory files under `_meta/`
- avoid scientific archive moves until the retrieval contract is stable

### Physical Tree Rule

Root files are reserved for the docs control plane:

- `README.md`
- `current_state.json`
- `llm_retrieval_index.md`
- `results_index.md`

All other docs should live in a role folder:

- docs about docs, inventories, and cross-cutting insights: `_meta/`
- current PatchNEPA / geo-teacher science: `patch_nepa/`
- task-domain ledgers: `classification/` or `completion/`
- execution boundaries: `operations/`
- historical predecessor ledgers: `query_nepa/`
- broad chronology: `history/`
- frozen or superseded material: `archive/`

### Stable Canonical Set

These should remain easy to find and light to retrieve:

- `nepa3d/docs/README.md`
- `nepa3d/docs/llm_retrieval_index.md`
- `nepa3d/docs/results_index.md`
- `nepa3d/docs/_meta/code_inventory_active.md`
- `nepa3d/docs/_meta/config_inventory_active.md`
- `nepa3d/docs/_meta/insight_register_active.md`
- `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
- `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
- `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
- `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
- `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
- `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- `nepa3d/docs/operations/README.md`

### Active But Heavy

These stay, but should not be the first read:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/query_nepa/runlog_202602.md`
- `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- `nepa3d/docs/patch_nepa/nepa_tracks_index.md`
- `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

### Likely Archive / Merge Candidates

These need case-by-case review:

- small one-off comparison docs whose conclusions are already reflected in
  `storyline_query_to_patch_v2_active.md`
- audit TSVs that are no longer used directly in current discussion

Important:

- do not archive anything until the unique content has a canonical landing
  place.

## Recommended Procedure

### Phase 1: Build the insight layer

Done when:

- `_meta/insight_register_active.md` can answer "what did we learn from each major
  experiment family?"

This reduces pressure to read every branch memo or runlog.

### Phase 2: Inventory all docs

For each doc, record:

- path
- role
- current status (`keep`, `merge`, `archive`, `review`)
- canonical replacement if superseded
- retrieval tier

This can be done folder by folder.

### Phase 3: Merge obvious overlaps

Start with docs whose conclusions are already fully represented in:

- `storyline_query_to_patch_v2_active.md`
- `benchmark_scanobjectnn_variant.md`
- `hypothesis_matrix_active.md`

### Phase 4: Archive low-value docs

Move only after:

- backlinks are updated
- the canonical replacement is explicit
- retrieval index does not depend on the old file

## Minimal Operating Rule

When a new experiment completes:

1. update the canonical destination
2. update `_meta/insight_register_active.md` if a new insight appeared
3. update top-level docs if current mainline or headline changed
4. do not create a new standalone memo unless the existing canonical docs
   cannot absorb the information cleanly

Current canonical destinations:

- paper direction / route decision:
  - `patch_nepa/paper_direction_geo_teacher_202604.md`
  - `patch_nepa/dataset_geo_teacher_v1_spec.md`
  - `patch_nepa/experiment_route_ab_matrix_202604.md`
- benchmark headline:
  - `patch_nepa/benchmark_scanobjectnn_variant.md`
- local-only Itachi evidence:
  - `patch_nepa/itachi/results_geo_teacher_itachi_active.md`
- PointGPT / pointNEPA sidecar:
  - `classification/results_scanobjectnn_pointgpt_pointnepa_active.md`

## Immediate Next Step

Do not try to clean the whole tree in one pass.

Start with:

1. PatchNEPA active docs
2. QueryNEPA historical docs
3. classification / completion side docs
4. archive tail last

## Completed In This Pass

- added `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
- added `nepa3d/docs/patch_nepa/claude_context_active.md`
- grouped docs governance files under `nepa3d/docs/_meta/`
- added `nepa3d/docs/_meta/README.md`
- absorbed the former cross-track non-retrieval result index into
  `nepa3d/docs/results_index.md`
- archived the old cross-track result index at
  `nepa3d/docs/archive/results_master_nonretrieval_20260226.md`
- removed the now-empty `nepa3d/docs/active/` folder
- refreshed:
  - `nepa3d/docs/current_state.json`
  - `nepa3d/docs/README.md`
  - `nepa3d/docs/llm_retrieval_index.md`
  - `nepa3d/docs/results_index.md`
  - `nepa3d/docs/_meta/insight_register_active.md`
- promoted April 2026 geo-teacher docs into default retrieval:
  - `paper_direction_geo_teacher_202604.md`
  - `dataset_geo_teacher_v1_spec.md`
  - `experiment_route_ab_matrix_202604.md`
- added historical/stale banners to:
  - `patch_nepa_stage2_active.md`
  - `nepa_tracks_index.md`
  - `hypothesis_matrix_active.md`
- extended `scripts/analysis/check_top_level_docs_sync.py` to check Tier 0/Tier
  1 file existence and local references

- created `nepa3d/docs/_meta/docs_inventory_active.md`
- created `nepa3d/docs/_meta/code_inventory_active.md`
- created `nepa3d/docs/_meta/config_inventory_active.md`
- archived:
  - `nepa3d/docs/archive/patch_nepa_scratch_to_patch_comparison_reference.md`
- active references now point to:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- added canonical policy-boundary audit:
  - `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- added canonical code-boundary inventory so active guide docs can point to
  current ownership while runlogs keep historical paths
- added canonical config-boundary inventory so `nepa3d/configs/` can be
  migrated family-by-family without confusing ownership
