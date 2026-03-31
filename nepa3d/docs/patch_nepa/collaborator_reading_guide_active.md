# PatchNEPA Collaborator Reading Guide

Last updated: 2026-04-01

## Purpose

This is the shortest collaborator-facing entrypoint for the current PatchNEPA
line.

Read this when you want to answer:

- what this project has been doing recently,
- which results are currently considered valid,
- which files contain the real evidence,
- which scripts reproduce the current mainline.

This page is intentionally short. It points to the right documents instead of
duplicating the raw ledger.

## Paper-Facing Direction (2026-04)

Historical mainline remains `recong2` for provenance.

Current paper-facing direction is the geometric-teacher line built on the CQA
codepath.

Do not describe this line as symmetric cross-primitive input learning.

Describe it as point-context pretraining with derived geometric teacher
targets.

## One-Sentence Current Status

Current mainline is PatchNEPA v2 reconstruction `recong2` full300
(`recon_chamfer`, `composite`, generator depth `2`), but the ScanObjectNN
benchmark headline is currently under revalidation because the maintained FT
policy switched on 2026-03-14 from `val_split_mode=file` to official
`test-as-val`.

Current canonical headline status:

- `obj_bg=pending`
- `obj_only=pending`
- `pb_t50_rs=pending`

Historical note:

- previously cited `0.8485 / 0.8589 / 0.8140` was obtained under the earlier
  `file`-split FT policy and is now treated as internal/historical only.

Primary evidence:

- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
- `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`

## 10-Minute Reading Order

1. `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
   - Shortest answer to "what is the current paper-facing direction?"
   - Explains why the paper layer is now geometric-teacher focused rather than
     cross-primitive focused.
2. `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
   - Source of truth for paper-facing split / task / protocol semantics.
   - Use this first when asking whether data processing changed.
3. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
   - Best single document for "what happened from QueryNEPA to PatchNEPA v2?"
   - Includes current comparison snapshot, what failed on the cosine path, and
     why `recong2` is the current mainline.
4. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
   - Canonical benchmark table.
   - Use this for any paper-facing or collaborator-facing headline number.
5. `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
   - Exact list of which older ScanObjectNN FT rows are historical because they
     came from the earlier `file`-split policy.
6. `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
   - First paper-facing geo-teacher hypothesis sheet.
7. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
   - Shows which hypotheses are now supported, unsupported, or still open.

If you only read three files, read items `1-3`.

## 30-Minute Reading Order

After the 10-minute path, read:

8. `nepa3d/docs/patch_nepa/migration_cross_primitive_to_geo_teacher_202604.md`
   - Exact old-claim to new-claim mapping.
   - Use this when deciding what to keep vs downgrade to historical language.
9. `nepa3d/docs/patch_nepa/spec_geo_teacher_vocab_v1.md`
   - Paper-facing canonical task vocabulary.
10. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
   - Detailed active-branch memo.
   - Best source for recent reconstruction, generator, PointGPT-parity, and
     branch-level decisions.
11. `scripts/abci/README.md`
   - Current collaborator-facing ABCI entrypoints.
   - Best answer to "which shell script should I run?"
12. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
   - Raw job ledger.
   - Use only if you need exact job IDs, launch history, or provenance.

## What Each File Answers

- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
  - What changed across eras?
  - Which line is still valid?
  - What is the current mainline and why?
- `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
  - What is the paper-facing direction now?
  - Why is the story no longer centered on cross-primitive symmetry?
- `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
  - Does the data processing change at the raw-cache layer or only above it?
  - What split / task / protocol semantics should be used next?
- `nepa3d/docs/patch_nepa/spec_geo_teacher_vocab_v1.md`
  - Which tasks are canonical for the paper-facing line?
  - Which task names stay supplemental?
- `nepa3d/docs/patch_nepa/migration_cross_primitive_to_geo_teacher_202604.md`
  - Which old claims survive the migration?
  - Which terms should now be treated as historical only?
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
  - What are the benchmark-valid numbers?
  - Which rows are headline-safe versus internal-only?
- `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
  - Which old ScanObjectNN numbers are file-split based?
  - Which rows were demoted to historical/internal?
- `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
  - What did we believe, and what is the current verdict?
- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
  - What exactly happened in the recent branch?
  - What were the short-run and full-run diagnostics?
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
  - Which jobs ran, when, and with which IDs?
- `scripts/abci/README.md`
  - Which maintained wrapper should a collaborator actually run?
- `nepa3d/docs/code_inventory_active.md`
  - Which code path is canonical now versus historical shim-only?

## Code Entry Points

If a collaborator wants to inspect the implementation rather than the docs,
start here:

1. `nepa3d/tracks/patch_nepa/tokens/train/pretrain_patch_nepa_tokens.py`
   - Current PatchNEPA token pretrain implementation.
   - Contains `pretrain_objective`, `recon_loss_mode`,
     `recon_generator_depth`, and objective-aligned reconstruction diagnostics.
   - historical wrapper path `nepa3d/train/pretrain_patch_nepa_tokens.py`
     still exists for older `python -m` launches.
2. `nepa3d/tracks/patch_nepa/mainline/train/finetune_patch_cls.py`
   - Current ScanObjectNN / classification finetune implementation.
   - historical wrapper path `nepa3d/train/finetune_patch_cls.py` still exists.
3. `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
   - Main pretrain launcher.
4. `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
   - PBS submit wrapper for pretrain.
5. `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
   - Main finetune entrypoint.
6. `nepa3d/docs/code_inventory_active.md`
   - Canonical map for `track` vs `shared` vs `compat` paths.

## Reproduction Entry Points

For collaborator-facing ABCI usage, prefer the thin wrappers:

- `scripts/abci/submit_patchnepa_current_pretrain.sh`
- `scripts/abci/submit_patchnepa_current_ft.sh`
- `scripts/abci/submit_patchnepa_current_cpac.sh`

These are the maintained "current mainline" entrypoints. Do not start by
searching the entire `scripts/` tree unless you need an older or non-mainline
recipe.

## What Not To Read First

Do not start from these unless you explicitly need traceback:

- `nepa3d/docs/archive/`
- docs named `*_legacy.md`
- raw log files under `logs/`
- old QueryNEPA ledgers

Those are useful for provenance, not for understanding the current line.

## Recommended Hand-Off

When sending materials to a coauthor, send this exact set first:

1. `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
2. `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
3. `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
4. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
5. `scripts/abci/README.md`

If they then ask for branch detail, add:

6. `nepa3d/docs/patch_nepa/spec_geo_teacher_vocab_v1.md`
7. `nepa3d/docs/patch_nepa/hypothesis_matrix_geo_teacher_v1.md`
8. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

If they ask for provenance, add:

9. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
