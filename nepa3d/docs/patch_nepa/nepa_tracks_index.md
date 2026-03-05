# NEPA Tracks Index

Last updated: 2026-03-06

## 1. Purpose

This file separates documentation by model line to avoid mixing conclusions:

- Query-NEPA line: token-level sequence (`1 point = 1 token`)
- Patch-NEPA line: patch-level sequence (`K points = 1 token`)

## 2. Active Line (Patch-NEPA)

- Folder guide: `nepa3d/docs/patch_nepa/README.md`
- Cross-line storyline: `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- Hypothesis matrix: `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
- Plan: `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- Runlog: `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Gap audit: `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- Query chronology audit for porting: `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`
- Benchmark table (shared output surface): `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- Scratch->PatchNEPA comparison matrix: `nepa3d/docs/patch_nepa/comparison_scratch_to_patchnepa.md`

## 3. Legacy Line (Query-NEPA)

- Folder guide: `nepa3d/docs/query_nepa/README.md`
- Historical ledger:
  - `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
  - `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- Historical job runlog:
  - `nepa3d/docs/query_nepa/runlog_202602.md`

These are retained for traceability and ablation reference, not as active execution source for Stage-2.

## 4. Mapping Note: Query Settings vs Patch Settings

Query-NEPA settings that do not directly apply to current Patch-NEPA run:

- `qa_layout` (`interleave` / `split` / `split_sep`)
- `sequence_mode` (`block` / `event`)
- `event_order_mode`
- `ray_order_mode`

Current Patch-NEPA mainline recipe uses:

- patch sequence (`fps_knn`, `group_size=32`, `num_groups=64`)
- ray stream enabled (`N_RAY=1024`, `USE_RAY_PATCH=1`)
- sample mode fixed (`pt_sample_mode=rfps_cached`, `pt_rfps_key=auto|<bank_key>`, `pt_rfps_m=4096`)
- Stage-2 mainline disallows on-the-fly RFPS/FPS fallback (RFPS bank required)
- NEPA next-embedding objective on patch tokens
- transfer path fixed to direct PatchNEPA finetune (`--model_source patchnepa`)

## 5. Migration Policy

- Do not merge Query-NEPA and Patch-NEPA run records into one active document.
- Add new Stage-2 runs only to `runlog_patch_nepa_202602.md`.
- Add cross-line conclusions only to `storyline_query_to_patch_v2_active.md`.
- Add hypothesis-status updates only to `hypothesis_matrix_active.md`.
- Keep comparisons explicit as cross-line ablations in benchmark tables.
- Stage-2 pretrain mainline must use 16 GPUs (`4 nodes x 4 GPU/node`) via
  `nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`.

## 6. Implementation Ownership (Current Decision)

- Active implementation line is `Patch-NEPA`.
- `Query-NEPA` code/docs are retained as legacy reference and reproducibility baseline.
- New feature work for the current line is implemented on Patch-NEPA side first.

Porting note:

- Query-era concepts (`split/interleave`, QA-layout variants, dual-mask variants) are not considered "active by default" in Patch-NEPA until explicitly ported and validated.
- When porting, record it in:
  - `patch_nepa_stage2_active.md` (design/plan)
  - `runlog_patch_nepa_202602.md` (execution/results)
