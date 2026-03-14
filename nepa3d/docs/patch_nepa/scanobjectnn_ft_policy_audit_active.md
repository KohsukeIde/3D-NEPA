# ScanObjectNN FT Policy Audit

Last updated: 2026-03-14

## 1. Purpose

This file freezes the 2026-03-14 ScanObjectNN downstream policy correction:

- maintained benchmark-facing FT now uses official Point-MAE / PointGPT-style
  `test-as-val` (`val_split_mode=pointmae`)
- earlier ScanObjectNN FT rows produced under `val_split_mode=file` or
  equivalent `NO_TEST_AS_VAL=1` settings are no longer canonical benchmark
  evidence

The goal is not to delete those results. The goal is to make their status
explicit:

- `canonical`: can be used as current benchmark evidence
- `historical/internal`: useful for within-project comparison only
- `provenance-only`: keep in raw ledgers, but do not reuse in active claims

## 2. Decision Rule

For ScanObjectNN benchmark-facing classification:

1. if model selection used official test split as validation
   (`val_split_mode=pointmae`, `val=test`, or equivalent), the result is
   benchmark-eligible
2. if model selection used a local file split
   (`val_split_mode=file`, `NO_TEST_AS_VAL=1`, or equivalent), the result is
   historical/internal only
3. inference-only sanity checks that do not choose the checkpoint via a local
   validation split are not invalidated by this policy correction

Operational consequence:

- raw ledgers keep the original numbers unchanged
- active benchmark docs must label affected rows as
  `historical file-split FT`

## 3. Affected Result Families

These are the main ScanObjectNN result families that must now be treated as
historical/internal rather than canonical benchmark evidence.

| family | representative values / evidence | why it is affected | prior practical role | required role now |
|---|---|---|---|---|
| PatchNEPA v1 family reference | `0.8262 / 0.8417 / 0.7845` in `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md` and `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` | this internal reference comes from the earlier ScanObjectNN FT regime tracked throughout `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`, where benchmark-facing FT was launched with `val_split_mode=file` | internal target line and sometimes shorthand headline in discussion | keep only as a historical internal reference for `v1 vs v2`; do not use as a current benchmark row |
| PatchNEPA v2 `reconbest` full300 (`g0`) | `0.8399 / 0.8348 / 0.8102`; FT root `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714`; summarized in `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`, and `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` | this FT set was produced before the 2026-03-14 policy correction and belongs to the earlier file-split FT regime | strongest no-generator transfer baseline | keep as `historical file-split FT baseline`; rerun under `val_split_mode=pointmae` before using as benchmark evidence |
| PatchNEPA v2 `recong2` full300 (`g2`) | `0.8485 / 0.8589 / 0.8140`; FT root `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643`; summarized in `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`, `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`, and `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` | this FT set also belongs to the earlier file-split FT regime | strongest observed PatchNEPA v2 transfer line | keep as `historical file-split FT best line`; official `test-as-val` rerun is required before canonization |
| Point-MAE file-split retry5 scratch | `0.8296 / 0.8399 / 0.8012`; section 2.2.2 in `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`; run note explicitly says `Point-MAE file-split retry5 (NO_TEST_AS_VAL=1)` | `NO_TEST_AS_VAL=1` is exactly the old non-official validation policy | strong scratch reference in some PatchNEPA comparisons | keep as historical/internal scratch reference only; do not compare directly against canonical `test-as-val` rows |
| Early PatchCls / PatchNEPA ScanObjectNN FT diagnostic chains | sections 7-12 in `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` and corresponding blocks in `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md` explicitly record `val_split_mode=file` plus TTA/voting (`100002-100075` lineage and related scratch rechecks) | these runs were diagnostic FT chains under the old file-split policy | engineering diagnostics and early transfer sanity | keep as provenance-only or historical diagnostics; do not reuse in benchmark comparison tables |
| Archived scratch-vs-PatchNEPA comparison matrix | `nepa3d/docs/archive/patch_nepa_scratch_to_patch_comparison_reference.md` states rows are protocol-valid under `val_split_mode=file` unless noted | the old reference matrix is entirely framed in the file-split regime | older comparison cheat-sheet | keep archived only; never promote back into active benchmark claims |

## 4. Families Not Invalidated By This Correction

These lines remain usable within their original scope because they are not
best-checkpoint ScanObjectNN FT under the old file-split policy.

| family | why it is not invalidated | current role |
|---|---|---|
| Point-MAE corrected variant-aligned sanity (`90.1893 / 87.9518 / 84.5940` in %) | this is pretrained-checkpoint inference sanity, not a local-FT model-selection result | external sanity / benchmark context |
| PatchNEPA / PointGPT pretrain diagnostics (`recon_lift_*`, cosine probes, etc.) | these are pretrain-side diagnostics, not ScanObjectNN FT selection results | objective analysis only |
| CPAC / UCPR / completion results | not ScanObjectNN classification model-selection surfaces | unaffected by this policy correction |
| ModelNet40 results | different downstream benchmark | unaffected by ScanObjectNN FT policy correction |
| `classification/results_scanobjectnn_core3_active.md`, `classification/results_scanobjectnn_review_active.md`, and `*_legacy.md` ScanObjectNN review tables | these were already legacy/review/provenance tables rather than the canonical PatchNEPA benchmark surface | keep as historical tracebacks, not as current headline sources |

## 5. Writing Rules For Active Docs

When an affected row is cited in an active doc:

1. prefix it as `historical file-split FT` or `historical/internal`
2. do not place it in the current canonical headline slot
3. if a benchmark-safe number is not yet available, show `pending` rather than
   silently reusing the old file-split value

Allowed uses of historical file-split FT rows:

- `g0 vs g2` internal prioritization
- deciding which rerun to launch next
- explaining why a branch is promising

Disallowed uses:

- paper-facing headline claim
- abstract / summary benchmark statement
- direct fairness claim against official Point-MAE / PointGPT-style downstream
  setups

## 6. Current Canonical Pointers

Use these files together:

- canonical benchmark page:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- cross-line interpretation:
  - `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- raw provenance:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- active branch detail:
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

Short rule:

- if a ScanObjectNN number came from the older `file`-split FT policy, keep it
  for traceback or internal ranking only
- if a number is needed as the current benchmark headline, wait for the
  official `test-as-val` rerun
