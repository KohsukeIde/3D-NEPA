# Patch-NEPA Docs

This folder is the active home for the Patch-NEPA line and its direct
comparisons against the legacy Query-NEPA line and external controls such as
Point-MAE and PointGPT.

## Reading Order

0. `../llm_retrieval_index.md`
   - Default retrieval policy for the whole `docs/` tree.
1. `storyline_query_to_patch_v2_active.md`
   - Cross-line storyline and interpretation boundary.
   - Start here when asking "what happened?" or "which result is valid?"
2. `hypothesis_matrix_active.md`
   - Hypotheses, supporting evidence, current status, and next minimal tests.
3. `execution_backlog_active.md`
   - Canonical local-only execution backlog and gating rules.
   - Start here when asking "what should run next locally?"
4. `restart_plan_patchnepa_data_v2_20260303.md`
   - Active investigation memo for the current v2 token-path branch.
5. `patch_nepa_stage2_active.md`
   - Active mainline policy and stage-2 execution rules.
6. `runlog_patch_nepa_202602.md`
   - Raw execution ledger for Patch-NEPA jobs.
7. `benchmark_scanobjectnn_variant.md`
   - Canonical headline ScanObjectNN benchmark table.

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

## Policy

- Use `storyline_query_to_patch_v2_active.md` as the default retrieval target.
- Put raw run additions into `runlog_patch_nepa_202602.md`.
- Put current-branch reasoning updates into
  `restart_plan_patchnepa_data_v2_20260303.md`.
- Put local-only next-run decisions into `execution_backlog_active.md`.
- Put cross-line conclusions only into
  `storyline_query_to_patch_v2_active.md` and
  `hypothesis_matrix_active.md`.
- Do not treat historical Point-MAE-style split rows as mainline evidence unless
  explicitly labeled as historical/reference.
- Do not retrieve archive docs for Patch-NEPA by default.
