# PatchNEPA Current LLM Brief

Last updated: 2026-05-12

## Current Truth

- Paper-facing direction:
  - derived geometric teacher pretraining for point-context encoders
  - short name: geo-teacher
- Historical mainline kept for provenance:
  - PatchNEPA v2 reconstruction `recong2` full300
  - `recon_chamfer`, `composite`, generator depth `2`
- Current route decision:
  - Route A if matched 100-epoch geo-teacher pretraining is favorable on
    ScanObjectNN and ShapeNetPart transfer; Route B if the strongest signal
    remains direct geometry readouts, controls, and completion.
- Current benchmark status:
  - ScanObjectNN PatchNEPA headline remains pending under official
    test-as-val; historical file-split FT numbers are internal only.
  - `obj_bg=pending`
  - `obj_only=pending`
  - `pb_t50_rs=pending`

## Read First

For almost any current PatchNEPA / geo-teacher question, read in this order:

1. `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
2. `nepa3d/docs/patch_nepa/claude_context_active.md`
3. `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
4. `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
5. `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
6. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`

For current results and boundaries, then add:

- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`
- `nepa3d/docs/classification/results_scanobjectnn_pointgpt_pointnepa_active.md`

## Do Not Use As Current Truth

- Do not describe the active paper direction as symmetric cross-primitive
  input learning.
- Do not use historical file-split FT rows as current ScanObjectNN benchmark
  headline numbers.
- Do not treat Itachi-local results as paper-facing benchmark rows unless they
  have been copied into the canonical benchmark page.
- Do not treat the PointGPT / pointNEPA sidecar page as a PatchNEPA headline
  result; it is active comparison context.
- Do not start from raw runlogs unless the question asks for exact job
  provenance.

## Where Specific Answers Live

| question | read |
|---|---|
| current paper story | `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md` |
| data / split semantics | `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md` |
| Route A/B decision rule | `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md` |
| CQA / geo-teacher hypotheses | `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md` |
| historical PatchNEPA trajectory | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md` |
| ScanObjectNN headline status | `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` |
| file-split demotion rule | `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md` |
| local execution / gating | `nepa3d/docs/patch_nepa/execution_backlog_active.md` |
| local vs ABCI boundary | `nepa3d/docs/operations/README.md` |
| Itachi-local evidence | `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md` |
| PointGPT / pointNEPA sidecar | `nepa3d/docs/classification/results_scanobjectnn_pointgpt_pointnepa_active.md` |

## Source Of Truth Contract

- `nepa3d/docs/current_state.json` is the machine-readable state source.
- `nepa3d/docs/llm_retrieval_index.md` is the retrieval contract.
- `nepa3d/docs/results_index.md` is the result navigation surface.
- Raw ledgers remain append-only provenance.
