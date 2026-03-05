# QueryNEPA -> PatchNEPA v2 Storyline

Last updated: 2026-03-06

## 1. Purpose

This document is the shortest path for answering:

- what changed from Query-NEPA to Patch-NEPA,
- which results are still valid,
- which PatchNEPA line is the real internal reference,
- how Point-MAE and PointGPT should be compared,
- what is currently established vs still open.

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

## 3. Current Comparison Snapshot

| line | pretrain data | main objective / task | strongest current signal | best known downstream / control readout | current role |
|---|---|---|---|---|---|
| Query-NEPA | mixed historical line | token-level NEPA cosine | useful engineering and protocol lessons only | not headline-safe due to protocol mixups in parts of the line | historical reference |
| PatchNEPA v1 family | historical internal reference line | earlier content-target family | still the strongest internal PatchNEPA FT reference | `obj_bg=0.8262`, `obj_only=0.8417`, `pb_t50_rs=0.7845` | internal target line |
| PatchNEPA v2 cosine path | ShapeNet-family token-path branch | latent cosine with QA stream | repeated `cos_tgt ~= cos_prev`, tiny gap, high copy-win | no evidence it beats v1 | active negative result |
| PatchNEPA v2 recon `mse` | ShapeNet-family token-path branch | `ctx(mse)+q/a(mse)` composite recon | positive `recon_lift_q/a`, but weak FT | `0.7900 / 0.7814 / 0.7620` on `obj_bg / obj_only / pb_t50_rs` | active probe, clearly below v1 |
| PatchNEPA v2 recon `chamfer` | ShapeNet-family token-path branch | `ctx(chamfer)+q/a(mse)` composite recon | best current v2 recon branch | `0.8124 / 0.8313 / 0.7765` on `obj_bg / obj_only / pb_t50_rs` | current best v2 recon line, still below v1 |
| Point-MAE | external baseline | masked point-patch modeling | protocol sanity and benchmark context | use benchmark page only | external benchmark context |
| PointGPT | ShapeNet control line | causal reconstruction with generator | healthy reconstruction-style learning curve | use as pretrain/control reference, not a direct benchmark row here | reconstruction reference |

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

- it is still the comparison line for "did v2 really improve?"
- later v2 recon comparisons are explicitly measured against:
  - `obj_bg=0.8262`
  - `obj_only=0.8417`
  - `pb_t50_rs=0.7845`

Boundary:

- v1-family is not the same objective family as the current v2 token-path
  branch.
- use it as an internal performance reference, not as proof that v2 is solved.

Reference:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`

### 4.3 PatchNEPA v2

This is the active research line. The main result split is now:

- cosine-family path: repeatedly collapses in cosine-space diagnostics,
- reconstruction-family path: shows positive objective-aligned lift, but still
  does not exceed v1 in FT.

### 4.4 Point-MAE

Role in this project:

- protocol sanity that variant-split ScanObjectNN evaluation is wired correctly,
- external benchmark context for the same output surface.

Use:

- benchmark ceiling / sanity only,
- not as architectural evidence for NEPA-specific claims.

Reference:

- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- `nepa3d/docs/archive/patch_nepa_baseline_patchcls_scratch_reference.md`

### 4.5 PointGPT

Role in this project:

- reconstruction-style causal reference,
- useful for comparing generator presence, context-only patch reconstruction,
  and reconstruction-aligned diagnostics.

Current interpretation boundary:

- ShapeNet pretrain domain has been aligned for comparison.
- dual-mask parity patch was also tested on PatchNEPA.
- remaining differences are now smaller, but still include more than one axis:
  - generator presence/depth,
  - `ctx-only` vs composite recon loss,
  - QA split / primitive-conditioned answer stream.

Reference:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/classification/results_modelnet40_pointgpt_active.md`

## 5. Concrete PatchNEPA v2 Findings

### 5.1 Cosine Path: What Was Tried and What Failed

Negative findings now established in the active memo:

- centered-cosine variants (`none / segment / shape`) did not fix the
  `cos_tgt ~= cos_prev` regime,
- `skip_k` sweep (`1 / 2 / 4`) was effectively identical,
- PointGPT-style dual-mask parity also did not fix the collapse.

Representative parity-smoke result (`105852`):

- `loss_total=0.499029`
- `cos_tgt=0.5010`
- `cos_prev=0.5005`
- `gap=+0.0005`
- `copy_win=0.7495`

Interpretation:

- the current bottleneck is not explained by a missing column-mask
  implementation alone.
- cosine-family target behavior remains the main unresolved issue on this path.

### 5.2 Reconstruction Path: Objective-Aligned Diags Are Healthy

Short-run reconstruction re-tests showed almost identical behavior for
`recon_mse` and `recon_chamfer` on objective-aligned diagnostics.

| objective | final `recon_lift_q` | final `recon_lift_a` | final `recon_q_err` | final `recon_a_err` | short-run read |
|---|---:|---:|---:|---:|---|
| `recon_mse` | `0.2626` | `0.1196` | `0.2600` | `0.1200` | clearly better than copy baseline |
| `recon_chamfer` | `0.2626` | `0.1196` | `0.2599` | `0.1200` | clearly better than copy baseline |

Interpretation:

- reconstruction-space prediction is using context,
- the old cosine probes are the wrong readout for this objective family,
- `recon_chamfer` and `recon_mse` are nearly equivalent on short-run diags.

### 5.3 Reconstruction FT: Current Best v2 Is Still Below v1

Full FT results from short recon checkpoints:

| line | `obj_bg` | `obj_only` | `pb_t50_rs` | delta vs v1 family |
|---|---:|---:|---:|---|
| v1 family reference | `0.8262` | `0.8417` | `0.7845` | reference |
| `recon_mse` | `0.7900` | `0.7814` | `0.7620` | `-0.0362 / -0.0603 / -0.0225` |
| `recon_chamfer` | `0.8124` | `0.8313` | `0.7765` | `-0.0138 / -0.0104 / -0.0080` |

Interpretation:

- `recon_chamfer` is the best current v2 reconstruction branch,
- but both reconstruction branches remain below the v1-family internal
  reference,
- therefore: "positive recon lift" is established, "better downstream
  transfer" is not.

### 5.4 PointGPT-Parity Controls: What Is Now Aligned

The active memo already records a stricter PointGPT comparison axis:

- ShapeNet pretrain domain aligned,
- column dual-mask parity tested,
- `pointgpt_ctx_only` loss mode added,
- `ChamferL1 + ChamferL2` (`l12`) added,
- `loss_pointgpt_equiv` logging added,
- strict `pc100` PointGPT-loss-axis run submitted (`105884`).

Interpretation:

- the remaining comparison gap is no longer "PatchNEPA forgot to add dual mask"
  or "PatchNEPA forgot to add CD-L12 logging".
- the main unresolved differences are now objective/task structure and generator
  usage.

## 6. Practical Delta: PatchNEPA v2 vs PointGPT

The practical comparison should currently be read as:

| axis | PatchNEPA v2 current line | PointGPT reference |
|---|---|---|
| sequence/task | QA split with primitive-conditioned answer stream | single point-patch stream |
| main objective | cosine path or composite recon path | reconstruction-dominant path |
| loss surface | `ctx(chamfer or mse) + q/a(mse)` or cosine family | patch reconstruction (`CD-L12`) |
| generator | now configurable in PatchNEPA, but not yet the settled mainline | present and central |
| diagnostic space | `recon/copy/lift` for recon, cosine probes only for cosine path | reconstruction-aligned diagnostics |

This means:

- "PatchNEPA now has dual mask and Chamfer" is true,
- "PatchNEPA is already the same regime as PointGPT" is not yet true.

## 7. What Is Currently Established

- Query-NEPA is a historical line, not the active implementation line.
- PatchNEPA v1-family remains the best internal historical reference.
- PatchNEPA v2 cosine-family target repeatedly collapses in cosine diagnostics.
- PatchNEPA v2 reconstruction branch shows positive objective-aligned lift.
- `recon_chamfer` is the best current v2 reconstruction branch.
- PatchNEPA v2 still has not exceeded the historical v1-family FT baseline.
- Point-MAE is the protocol sanity / benchmark context.
- PointGPT is the reconstruction-style reference for the next causal tests.

## 8. What Is Still Open

- whether `reconstruction + generator` closes the transfer gap,
- whether strict PointGPT-loss parity plus generator depth reproduces the
  expected training dynamics on PatchNEPA,
- whether any cosine-family target on the v2 token path can escape the
  `cos_tgt ~= cos_prev` regime,
- whether Q/A composite structure helps or hurts transfer once generator-based
  recon is working.

## 9. Practical Reading Order for New Analysis

1. `nepa3d/docs/llm_retrieval_index.md`
2. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
3. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
4. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
5. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
6. `nepa3d/docs/query_nepa/runlog_202602.md`
