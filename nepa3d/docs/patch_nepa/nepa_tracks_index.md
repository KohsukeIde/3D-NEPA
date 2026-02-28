# NEPA Tracks Index

Last updated: 2026-02-28

## 1. Purpose

This file separates documentation by model line to avoid mixing conclusions:

- Query-NEPA line: token-level sequence (`1 point = 1 token`)
- Patch-NEPA line: patch-level sequence (`K points = 1 token`)

## 2. Active Line (Patch-NEPA)

- Plan: `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
- Runlog: `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Gap audit: `nepa3d/docs/patch_nepa/gap_audit_query_to_patch_active.md`
- Query chronology audit for porting: `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`
- Benchmark table (shared output surface): `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`

## 3. Legacy Line (Query-NEPA)

- Historical ledger:
  - `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
  - `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- Historical job runlog:
  - `nepa3d/docs/query_nepa/runlog_202602.md`

These are retained for traceability and ablation reference, not as active execution source for Stage-2.

## 4. Mapping Note: Query Settings vs Patch Settings

Query-NEPA settings that do not directly apply to current Patch-NEPA point-only run:

- `qa_layout` (`interleave` / `split` / `split_sep`)
- `sequence_mode` (`block` / `event`)
- `event_order_mode`
- `ray_order_mode`

Current Patch-NEPA point-only baseline recipe uses:

- patch sequence (`fps_knn`, `group_size=32`, `num_groups=64`)
- no ray stream (`N_RAY=0`, `USE_RAY_PATCH=0`)
- sample mode default (`pt_sample_mode=rfps`, `pt_rfps_m=4096`)
- NEPA next-embedding objective on patch tokens

## 5. Migration Policy

- Do not merge Query-NEPA and Patch-NEPA run records into one active document.
- Add new Stage-2 runs only to `runlog_patch_nepa_202602.md`.
- Keep comparisons explicit as cross-line ablations in benchmark tables.

## 6. Implementation Ownership (Current Decision)

- Active implementation line is `Patch-NEPA`.
- `Query-NEPA` code/docs are retained as legacy reference and reproducibility baseline.
- New feature work for the current line is implemented on Patch-NEPA side first.

Porting note:

- Query-era concepts (`split/interleave`, QA-layout variants, dual-mask variants) are not considered "active by default" in Patch-NEPA until explicitly ported and validated.
- When porting, record it in:
  - `patch_nepa_stage2_active.md` (design/plan)
  - `runlog_patch_nepa_202602.md` (execution/results)
