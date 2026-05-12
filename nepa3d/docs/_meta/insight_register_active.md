# Insight Register

Last updated: 2026-05-12

## Purpose

This file records what each decision-relevant experiment family taught us.

Use it before opening raw ledgers. One row should represent one insight, not
one job.

## Status Labels

- `current`: part of the current paper / route decision surface
- `supporting`: useful current context, but not the headline route by itself
- `historical`: provenance or older motivation
- `negative-result`: useful because it closed a branch
- `sidecar`: comparison context outside the PatchNEPA headline

## Current Highest-Value Insights

1. The paper-facing line is now geo-teacher: derived geometric teacher
   pretraining for point-context encoders, not symmetric cross-primitive input
   learning.
2. The safest geometry anchor remains CQA `DISTANCE + NORMAL_UNSIGNED`, with
   packed common-split training giving a small positive over mixture.
3. Itachi 100/300-epoch results are useful local evidence, but benchmark-facing
   claims still require canonization in the canonical benchmark docs.
4. ScanObjectNN PatchNEPA headline values remain pending under official
   test-as-val; historical file-split FT numbers are internal only.
5. PointGPT / pointNEPA results are active sidecar comparison context, not
   PatchNEPA headline results.

## Insight Table

| ID | period / line | experiment family | what was newly learned | status | canonical evidence |
|---|---|---|---|---|---|
| I001 | QueryNEPA -> PatchNEPA | historical QueryNEPA audit | QueryNEPA provides protocol lessons, but mixed historical runs should not be used directly as current benchmark evidence. | historical | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md` |
| I002 | PatchNEPA v2 cosine | centered-cosine / `skip_k` / mask controls | The cosine path repeatedly enters `cos_tgt ~= cos_prev`; small target tweaks and dual-mask parity do not explain the collapse. | negative-result | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md` |
| I003 | PatchNEPA v2 reconstruction | `recon_mse` / `recon_chamfer` / `g0` / `g2` | Reconstruction objectives use context, and `g2` was the strongest historical transfer candidate, but its ScanObjectNN FT numbers are file-split historical until official rerun. | historical | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` |
| I004 | ScanObjectNN policy | public downstream split audit | Benchmark-facing ScanObjectNN parity follows official test-as-val; earlier file-split FT rows must be treated as internal ranking/provenance. | current | `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`, `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` |
| I005 | Paper framing | cross-primitive -> geo-teacher migration | The current paper story is derived geometric teacher supervision for point/surface context encoders; `surf -> pc_bank` is degraded-context evaluation, not the central cross-primitive claim. | current | `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`, `nepa3d/docs/patch_nepa/migration_cross_primitive_to_geo_teacher_202604.md` |
| I006 | Dataset protocol | `world_v3` raw layer + geo-teacher semantic layer | Raw `world_v3` stays frozen; the paper-facing shift happens through manifests, splits, task selection, and packed same-shape protocol. | current | `nepa3d/docs/patch_nepa/spec_world_v3_schema.md`, `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md` |
| I007 | Route decision | first matched geo-teacher compare | The immediate decision is a matched 100-epoch `recon / distance / distance+normal` comparison; adding more task types is phase 2. | current | `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`, `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md` |
| I008 | CQA single-task | frozen `world_v3` `udf_distance` | `udf_distance` is a stable method branch across same-context, degraded-context, controls, seed pack, and dense completion. | supporting | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md` |
| I009 | CQA main anchor | C034 `DISTANCE + NORMAL_UNSIGNED` | The strict 2-type line remains the safest discrete geometry anchor: strong distance/normal reads and better core completion than richer task packs. | current | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md` |
| I010 | Multi-probe tradeoff | C035 / C042 / C043 | Richer task packs can improve downstream utility slightly, but they do not beat C034 on the core distance+normal answering/completion anchor; AO-HQ is supportive, not headline-safe. | supporting | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md` |
| I011 | Protocol isolation | common-split mixture vs packed | Removing the split confound does not collapse the 2-type row; packed all-type-per-shape is a small positive over mixture, but not large enough to explain every tradeoff. | current | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md` |
| I012 | External CQA control | frozen Point-MAE CQA readout | Frozen Point-MAE plus lightweight CQA head is a real non-internal `udf_distance` control, but it is a single-task sidecar harness rather than a replacement for the CQA mainline. | sidecar | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/cqa_external_baseline_plan.md` |
| I013 | Itachi Route A/B | local 100/300 epoch geo-teacher runs | Itachi full-budget 300 epochs improves same-context geometry more than Route-A utility; the interim matched compare budget remains 100 epochs. | supporting | `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`, `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md` |
| I014 | PointGPT sidecar | PointGPT / pointNEPA ScanObjectNN matrix | PointGPT/pointNEPA sidecar results are active comparison context and protocol sanity, not PatchNEPA benchmark headline values. | sidecar | `nepa3d/docs/classification/results_scanobjectnn_pointgpt_pointnepa_active.md`, `nepa3d/docs/results_index.md` |
| I015 | Operations split | local vs ABCI boundaries | Local Itachi workflows, ABCI collaborator wrappers, and internal workers have separate documentation surfaces; local results need explicit canonization before entering benchmark docs. | current | `nepa3d/docs/operations/README.md`, `scripts/local/README.md`, `scripts/abci/README.md` |

## Maintenance Rule

- Add a row only when an experiment family changes the decision surface.
- If a finding is superseded, keep the row and update `status` plus evidence.
- Keep raw job history in:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
  - `nepa3d/docs/query_nepa/runlog_202602.md`
