# Patch-NEPA Docs

This folder is the active home for the Patch-NEPA line and its direct
comparisons against the legacy Query-NEPA line and external controls such as
Point-MAE and PointGPT.

## Reading Order

0. `../llm_retrieval_index.md`
   - Default retrieval policy for the whole `docs/` tree.
1. `collaborator_reading_guide_active.md`
   - One-page collaborator-facing entrypoint.
   - Start here when sharing the current line with a coauthor.
2. `paper_direction_geo_teacher_202604.md`
   - Paper-facing direction layer added above the historical mainline docs.
3. `dataset_geo_teacher_v1_spec.md`
   - Paper-facing dataset / split / protocol source of truth.
4. `experiment_route_ab_matrix_202604.md`
   - First matched `100`-epoch decision matrix for Route A vs Route B.
5. `storyline_query_to_patch_v2_active.md`
   - Cross-line storyline and interpretation boundary.
   - Start here when asking "what happened?" or "which result is valid?"
6. `hypothesis_matrix_geo_teacher_v1.md`
   - Paper-facing geo-teacher hypothesis sheet.
7. `hypothesis_matrix_active.md`
   - Hypotheses, supporting evidence, current status, and next minimal tests.
8. `scanobjectnn_ft_policy_audit_active.md`
   - Exact list of old ScanObjectNN FT result families that are now
     historical/internal because they came from the earlier `file`-split
     policy.
9. `migration_cross_primitive_to_geo_teacher_202604.md`
   - Old-claim to new-claim migration note.
10. `spec_geo_teacher_vocab_v1.md`
   - Paper-facing canonical task vocabulary.
11. `execution_backlog_active.md`
   - Canonical local-only execution backlog and gating rules.
   - Start here when asking "what should run next locally?"
12. `restart_plan_patchnepa_data_v2_20260303.md`
   - Active investigation memo for the current v2 token-path branch.
13. `patch_nepa_stage2_active.md`
   - Active mainline policy and stage-2 execution rules.
14. `runlog_patch_nepa_202602.md`
   - Raw execution ledger for Patch-NEPA jobs.
15. `benchmark_scanobjectnn_variant.md`
   - Canonical headline ScanObjectNN benchmark table.
16. `../code_inventory_active.md`
   - Canonical code-organization boundary for current path ownership.

## Machine-Specific Notes

Workstation-specific operational notes for `itachi` live under:

- `itachi/README.md`
- `itachi/local_data_ops_202604.md`
- `itachi/local_geo_teacher_pretrain_ops_202604.md`
- `itachi/local_geo_teacher_posttrain_ops_202604.md`

These are local execution notes only. Do not treat them as paper-facing or
ABCI-facing source-of-truth docs.

## Collaborator Quick Path

If a collaborator asks:

- "which ABCI script should I run?"
- "which docs should I read first?"
- "where do I quickly confirm the current result?"

start from:

1. `collaborator_reading_guide_active.md`
   - one-page reading order and file-role memo
2. `scripts/abci/README.md`
   - curated ABCI entrypoints for the current PatchNEPA line
   - includes the thin submit wrappers for pretrain / finetune / mini-CPAC
3. `storyline_query_to_patch_v2_active.md`
   - shortest answer to "what is the current valid result?"
4. `benchmark_scanobjectnn_variant.md`
   - canonical ScanObjectNN headline table
5. `scanobjectnn_ft_policy_audit_active.md`
   - exact boundary between historical file-split FT rows and maintained
     official benchmark rows
6. `restart_plan_patchnepa_data_v2_20260303.md`
   - current branch memo and the most detailed active result backfill

Current short answer:

- current main reconstruction line: PatchNEPA v2 `recong2` full300
- current paper-facing direction: geometric-teacher pretraining on the CQA path
- current ScanObjectNN benchmark headline: pending revalidation
- historical file-split FT headline: `0.8485 / 0.8589 / 0.8140`
- current thin ABCI entrypoints:
  - `scripts/abci/submit_patchnepa_current_pretrain.sh`
  - `scripts/abci/submit_patchnepa_current_ft.sh`
  - `scripts/abci/submit_patchnepa_current_cpac.sh`
  - `scripts/abci/submit_patchnepa_current_cqa_pretrain.sh` (experimental)
  - `scripts/abci/submit_patchnepa_geo_teacher_compare_pretrain.sh`

## Code Boundary

Current code ownership is:

- track-specific implementation: `nepa3d/tracks/patch_nepa/*`
- shared active infra: `nepa3d/data/`, `nepa3d/backends/`, `nepa3d/token/`,
  `nepa3d/utils/`, `nepa3d/core/models/`
- compatibility entrypoints: `nepa3d/models/`, `nepa3d/train/`,
  `nepa3d/analysis/`

Important:

- active guide docs should point to the canonical ownership paths above
- historical ledgers may still mention the shim paths that were executed at the
  time
- the canonical boundary doc is `../code_inventory_active.md`

## Experimental CQA Branch

An additive explicit-query CQA branch now exists alongside the current
`recong2/composite` mainline.

- canonical model home:
  - `nepa3d/tracks/patch_nepa/cqa/models/primitive_answering.py`
- canonical dataset home:
  - `nepa3d/tracks/patch_nepa/cqa/data/dataset_cqa.py`
- fixed vocab spec: `spec_cqa_vocab.md`
- canonical config home:
  - `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_unpaired_mix_v2_cqa.yaml`
- current matched-compare configs:
  - `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distance_only_v1.yaml`
  - `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml`
- maintained wrapper default:
  - `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_unpaired_mix_v2_cqa.yaml`
- matched-compare wrapper:
  - `scripts/abci/submit_patchnepa_geo_teacher_compare_pretrain.sh`
- historical top-level compatibility copy:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml`
- canonical audit script:
  - `nepa3d/tracks/patch_nepa/cqa/analysis/audit_cqa_targets.py`
- classification utility wrapper: `scripts/eval/nepa3d_cqa_cls_qg.sh`
- translation/completion wrapper: `scripts/analysis/nepa3d_cqa_udfdist_translation_qg.sh`
- external baseline plan: `nepa3d/docs/patch_nepa/cqa_external_baseline_plan.md`
- next launcher-compatible smoke configs:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfsurf.yaml`
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`

Current merge rules:

- existing `recong2/composite` remains the mainline baseline
- CQA is evaluated as an additive branch
- the paper-facing geo-teacher layer is added above the historical mainline
  docs rather than replacing them
- surface-aligned tasks use `surf_xyz` as the query carrier
- only `ASK_DISTANCE` uses `udf_qry_xyz`
- the first `mesh_visibility + udf_thickness` smoke is a wiring check only, not
  headline evidence for promptable answering

## Data Freeze

The current raw world-package contract is frozen as:

- `spec_world_v3_schema.md`

Current Phase-1 data-freeze utilities:

- `scripts/preprocess/augment_shapenet_world_v3.sh`
  - add `world_v3` summary/provenance fields to the existing source cache
- `scripts/preprocess/audit_world_v3.sh`
  - build the canonical global audit summary for the frozen cache
- `scripts/preprocess/build_shapenet_subset_manifest.sh`
  - build the clean `subset_watertight` manifest from the current source cache

The data-freeze policy is:

- do **not** full-rebuild ShapeNet unless the raw contract itself is proven wrong
- first freeze the raw cache contract, validity summaries, and subset manifests
- then rerun only the smallest decisive experiments on top of that fixed cache

## Document Roles

- `runlog_patch_nepa_202602.md`
  - Raw job-by-job record. Keep details here, not conclusions.
- `restart_plan_patchnepa_data_v2_20260303.md`
  - Current experiment branch notebook. Use for in-flight diagnostics and local
    decision notes.
- `execution_backlog_active.md`
  - Canonical local-only execution order, gating, and canonization targets.
- `storyline_query_to_patch_v2_active.md`
  - Cross-era summary across QueryNEPA, PatchNEPA v1, PatchNEPA v2, and
    external baselines.
- `hypothesis_matrix_active.md`
  - Stable decision surface for cone, reconstruction, generator, masking, and
    fairness hypotheses.
- `gap_audit_query_to_patch_active.md`
  - Porting parity checklist from Query-NEPA to Patch-NEPA.
- `query_nepa_chronology_audit_202602_active.md`
  - Historical validity audit for findings inherited from Query-NEPA.
- `scanobjectnn_ft_policy_audit_active.md`
  - Explicit audit of which ScanObjectNN FT result families are historical
    because they were produced under the old `file`-split policy.

## Supporting Tables

- `patchcls_completed_results.tsv`
- `patchcls_exhaustive_audit.tsv`
- `patchnepa_ft_completed_results.tsv`
- `patchnepa_ft_exhaustive_audit.tsv`
- `diagcopy_probe_100755_gap_copywin.tsv`
- `latent_diag_snapshot_20260304.tsv`

## Archived Reference

- `../archive/patch_nepa_scratch_to_patch_comparison_reference.md`
  - archived comparison matrix after its headline role moved to:
    - `benchmark_scanobjectnn_variant.md`
    - `storyline_query_to_patch_v2_active.md`

## Policy

- Use `storyline_query_to_patch_v2_active.md` as the default retrieval target.
- Put raw run additions into `runlog_patch_nepa_202602.md`.
- Put current-branch reasoning updates into
  `restart_plan_patchnepa_data_v2_20260303.md`.
- Put local-only next-run decisions into `execution_backlog_active.md`.
- Put cross-line conclusions only into
  `storyline_query_to_patch_v2_active.md` and
  `hypothesis_matrix_active.md`.
- For collaborator-facing ABCI usage, prefer the curated wrappers under
  `scripts/abci/` over searching the broader `scripts/` tree directly.
- Do not treat historical Point-MAE-style split rows as mainline evidence unless
  explicitly labeled as historical/reference.
- Do not retrieve archive docs for Patch-NEPA by default.
