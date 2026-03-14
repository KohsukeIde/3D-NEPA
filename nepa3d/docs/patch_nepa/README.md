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
2. `storyline_query_to_patch_v2_active.md`
   - Cross-line storyline and interpretation boundary.
   - Start here when asking "what happened?" or "which result is valid?"
3. `hypothesis_matrix_active.md`
   - Hypotheses, supporting evidence, current status, and next minimal tests.
4. `execution_backlog_active.md`
   - Canonical local-only execution backlog and gating rules.
   - Start here when asking "what should run next locally?"
5. `restart_plan_patchnepa_data_v2_20260303.md`
   - Active investigation memo for the current v2 token-path branch.
6. `patch_nepa_stage2_active.md`
   - Active mainline policy and stage-2 execution rules.
7. `runlog_patch_nepa_202602.md`
   - Raw execution ledger for Patch-NEPA jobs.
8. `benchmark_scanobjectnn_variant.md`
   - Canonical headline ScanObjectNN benchmark table.

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
5. `restart_plan_patchnepa_data_v2_20260303.md`
   - current branch memo and the most detailed active result backfill

Current short answer:

- current main reconstruction line: PatchNEPA v2 `recong2` full300
- current headline FT: `0.8485 / 0.8589 / 0.8140`
- current thin ABCI entrypoints:
  - `scripts/abci/submit_patchnepa_current_pretrain.sh`
  - `scripts/abci/submit_patchnepa_current_ft.sh`
  - `scripts/abci/submit_patchnepa_current_cpac.sh`
  - `scripts/abci/submit_patchnepa_current_cqa_pretrain.sh` (experimental)

## Experimental CQA Branch

An additive explicit-query CQA branch now exists alongside the current
`recong2/composite` mainline.

- core model: `nepa3d/models/primitive_answering.py`
- dataset: `nepa3d/data/dataset_cqa.py`
- fixed vocab spec: `spec_cqa_vocab.md`
- minimal config: `nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml`

Current merge rules:

- existing `recong2/composite` remains the mainline baseline
- CQA is evaluated as an additive branch
- surface-aligned tasks use `surf_xyz` as the query carrier
- only `ASK_DISTANCE` uses `udf_qry_xyz`

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
