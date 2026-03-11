# PatchNEPA Collaborator Reading Guide

Last updated: 2026-03-12

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

## One-Sentence Current Status

Current mainline is PatchNEPA v2 reconstruction `recong2` full300
(`recon_chamfer`, `composite`, generator depth `2`), and the current canonical
ScanObjectNN headline is:

- `obj_bg=0.8485`
- `obj_only=0.8589`
- `pb_t50_rs=0.8140`

Primary evidence:

- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`

## 10-Minute Reading Order

1. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
   - Best single document for "what happened from QueryNEPA to PatchNEPA v2?"
   - Includes current comparison snapshot, what failed on the cosine path, and
     why `recong2` is the current mainline.
2. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
   - Canonical benchmark table.
   - Use this for any paper-facing or collaborator-facing headline number.
3. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
   - Shows which hypotheses are now supported, unsupported, or still open.

If you only read three files, read those three.

## 30-Minute Reading Order

After the 10-minute path, read:

4. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
   - Detailed active-branch memo.
   - Best source for recent reconstruction, generator, PointGPT-parity, and
     branch-level decisions.
5. `scripts/abci/README.md`
   - Current collaborator-facing ABCI entrypoints.
   - Best answer to "which shell script should I run?"
6. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
   - Raw job ledger.
   - Use only if you need exact job IDs, launch history, or provenance.

## What Each File Answers

- `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
  - What changed across eras?
  - Which line is still valid?
  - What is the current mainline and why?
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
  - What are the benchmark-valid numbers?
  - Which rows are headline-safe versus internal-only?
- `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
  - What did we believe, and what is the current verdict?
- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
  - What exactly happened in the recent branch?
  - What were the short-run and full-run diagnostics?
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
  - Which jobs ran, when, and with which IDs?
- `scripts/abci/README.md`
  - Which maintained wrapper should a collaborator actually run?

## Code Entry Points

If a collaborator wants to inspect the implementation rather than the docs,
start here:

1. `nepa3d/train/pretrain_patch_nepa_tokens.py`
   - Current PatchNEPA token pretrain implementation.
   - Contains `pretrain_objective`, `recon_loss_mode`,
     `recon_generator_depth`, and objective-aligned reconstruction diagnostics.
2. `nepa3d/train/finetune_patch_cls.py`
   - Current ScanObjectNN / classification finetune implementation.
3. `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
   - Main pretrain launcher.
4. `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
   - PBS submit wrapper for pretrain.
5. `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
   - Main finetune entrypoint.

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
2. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
3. `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
4. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
5. `scripts/abci/README.md`

If they then ask for branch detail, add:

6. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

If they ask for provenance, add:

7. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
