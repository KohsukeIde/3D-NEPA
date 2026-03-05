# QueryNEPA -> PatchNEPA v2 Storyline

Last updated: 2026-03-06

## 1. Purpose

This document is the shortest path for answering:

- what changed from Query-NEPA to Patch-NEPA,
- which results are still valid,
- which PatchNEPA baseline is the real reference,
- where Point-MAE and PointGPT fit in the comparison story.

It is a synthesis page, not a raw ledger.

## 2. Canonical Sources

- Query historical raw ledger:
  - `nepa3d/docs/query_nepa/runlog_202602.md`
- Patch active raw ledger:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Patch v2 active investigation memo:
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- Patch benchmark headline table:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- Query-to-patch validity audit:
  - `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

## 3. Line Summary

| line | main objective | data / protocol status | current role | interpretation boundary |
|---|---|---|---|---|
| Query-NEPA | token-level NEPA cosine | mixed validity; some runs include `scanobjectnn_main_split_v2` lineage | historical reference | engineering lessons remain useful, headline metrics do not |
| PatchNEPA v1 family | historical content-target family | strong internal reference, but not the current v2 token-path branch | best historical PatchNEPA reference | use as internal target line, not as proof of current v2 correctness |
| PatchNEPA v2 | token-path QA/reconstruction branch | active branch; run-by-run boundaries recorded in active memo | main current research line | conclusions must come from objective-aligned diagnostics plus strict FT |
| Point-MAE | external baseline / sanity | protocol-correct variant-split sanity exists | external upper baseline and split sanity | not a shared architecture; use for benchmark context only |
| PointGPT | causal reconstruction reference | active ShapeNet pretrain reference added | reconstruction-style control | compare by objective-aligned diagnostics, not by cosine metrics |

## 4. Era-by-Era Reading

### 4.1 Query-NEPA

What remains valid:

- protocol hardening lessons,
- explicit config-print / fail-fast policy,
- fairness rule that mixed `scanobjectnn_main_split_v2` results are not
  headline-safe.

What is not valid as current benchmark evidence:

- mixed-cache ScanObjectNN headline reporting,
- runs later marked provisional/invalid due to launch or protocol mismatch.

Reference:

- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

### 4.2 PatchNEPA v1 Family

Why this line matters:

- it is still the strongest internal PatchNEPA reference used in later v2
  comparisons.
- the current v2 memo treats the prior v1-family baseline as:
  - `obj_bg=0.8262`
  - `obj_only=0.8417`
  - `pb_t50_rs=0.7845`

Why it should not be merged into the v2 active memo:

- v1-family and v2 token-path runs are not the same objective family.
- v1 is the comparison target for "did v2 really improve?" not the same branch.

Reference:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

### 4.3 PatchNEPA v2

What is established so far:

- cosine-space diagnostics collapsed repeatedly in the token-path branch
  (`cos_tgt ~= cos_prev`, tiny gap),
- centered cosine and `skip_k` changes did not fix the core issue,
- reconstruction-space diagnostics are healthier:
  - `recon_lift_q/a` become positive,
  - `recon_mse` and `recon_chamfer` are nearly identical on short-run
    diagnostics,
  - FT still remains below the prior v1-family baseline.

Current interpretation:

- v2 reconstruction does create context-using behavior in objective-aligned
  space,
- but the current transfer path is still insufficient to beat the historical
  v1-family baseline.

Reference:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

### 4.4 Point-MAE

Role in this project:

- protocol sanity that variant-split ScanObjectNN evaluation is wired correctly,
- external benchmark context for the same output surface.

Use:

- benchmark ceiling / sanity only,
- not as architectural evidence for NEPA-specific claims.

Reference:

- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/patch_nepa/baseline_patchcls_scratch.md`

### 4.5 PointGPT

Role in this project:

- reconstruction-style causal reference,
- useful for comparing generator presence, context-only patch reconstruction,
  and reconstruction-aligned diagnostics.

Current value:

- shows what healthy reconstruction training curves look like,
- provides a control axis for `generator_depth`, `pointgpt_ctx_only`, and
  `ChamferL1 + ChamferL2`.

Important boundary:

- PointGPT should be compared to PatchNEPA in reconstruction-aligned space
  (`recon/copy/lift` and latent spread), not by PatchNEPA cosine probes.

Reference:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/classification/results_modelnet40_pointgpt_active.md`

## 5. What Is Currently Established

- Query-NEPA is a historical line, not the active implementation line.
- PatchNEPA v1-family remains the best internal historical reference.
- PatchNEPA v2 reconstruction branch shows positive objective-aligned lift.
- PatchNEPA v2 has not yet exceeded the historical v1-family FT baseline.
- Point-MAE is the protocol sanity / benchmark context.
- PointGPT is the reconstruction-style reference for the next causal tests.

## 6. What Is Still Open

- whether `reconstruction + generator` closes the transfer gap,
- whether PointGPT-style loss parity plus generator depth reproduces the expected
  training dynamics on PatchNEPA,
- whether any cosine-family target on the v2 token path can escape the
  `cos_tgt ~= cos_prev` regime.

## 7. Practical Reading Order for New Analysis

1. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
2. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
3. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
4. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
5. `nepa3d/docs/query_nepa/runlog_202602.md`
