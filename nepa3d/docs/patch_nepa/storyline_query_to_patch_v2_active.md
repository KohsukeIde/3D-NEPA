# QueryNEPA -> PatchNEPA v2 Storyline

Last updated: 2026-03-30

## 1. Purpose

This document is the shortest path for answering:

- what changed from Query-NEPA to Patch-NEPA,
- which results are still valid,
- which PatchNEPA line is the real internal reference,
- how Point-MAE and PointGPT should be compared,
- what is currently established vs still open.

It is a synthesis page, not a raw ledger.

Important 2026-03-14 correction:

- maintained ScanObjectNN downstream policy has been switched from
  `val_split_mode=file` to official `test-as-val`
- historical FT numbers obtained under the old `file`-split policy are kept as
  internal diagnostics, not as the current canonical benchmark headline

## 2. Canonical Sources

- Query historical raw ledger:
  - `nepa3d/docs/query_nepa/runlog_202602.md`
- Patch active raw ledger:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Patch v2 active investigation memo:
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- Patch benchmark headline table:
  - `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- ScanObjectNN FT policy audit:
  - `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- Patch local-only execution backlog:
  - `nepa3d/docs/patch_nepa/execution_backlog_active.md`
- Query-to-patch validity audit:
  - `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`

## 3. Current Comparison Snapshot

| line | pretrain data | main objective / task | strongest current signal | best known downstream / control readout | current role |
|---|---|---|---|---|---|
| Query-NEPA | mixed historical line | token-level NEPA cosine | useful engineering and protocol lessons only | not headline-safe due to protocol mixups in parts of the line | historical reference |
| PatchNEPA v1 family | historical internal reference line | earlier content-target family | still the strongest internal PatchNEPA FT reference | `obj_bg=0.8262`, `obj_only=0.8417`, `pb_t50_rs=0.7845` | internal target line |
| PatchNEPA v2 cosine path | ShapeNet-family token-path branch | latent cosine with QA stream | repeated `cos_tgt ~= cos_prev`, tiny gap, high copy-win | no evidence it beats v1 | active negative result |
| PatchNEPA v2 recon `g0` full300 | ShapeNet-family token-path branch | `ctx(chamfer)+q/a(mse)` composite recon | positive `recon_lift_q/a` survives full run | historical file-split FT readout `0.8399 / 0.8348 / 0.8102`; official rerun pending | validated no-generator reconstruction baseline |
| PatchNEPA v2 recon `g2` full300 | ShapeNet-family token-path branch | composite recon + generator depth `2` | strongest current pretrain/transfer line | historical file-split FT readout `0.8485 / 0.8589 / 0.8140`; official rerun pending | current main reconstruction line, benchmark under revalidation |
| PatchNEPA v2 explicit-query CQA | frozen `world_v3` CQA branch | typed primitive-conditioned answering / completion | `udf_distance` is now stable on same-context, zero-shot off-diagonal, dense completion, and factorization/format ablations; strongest row is `independent + shuffled`, with `DISTANCE + NORMAL` as the first shared multi-type gate | ScanObjectNN utility is paper-safe; Q-block and continuous ablations show shared context + per-query fixed-target answering is the main driver | active method branch; strongest new line, but multi-type headline is still stabilizing |
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
- reconstruction-family path: shows positive objective-aligned lift, and the
  full300 `g2` line now exceeds the historical v1 FT reference on all three
  ScanObjectNN variants,
- explicit-query CQA path: `udf_distance` now supports same-context,
  zero-shot off-diagonal transfer, dense completion, and systematic
  factorization/format ablations, with `independent` prompting as the strongest
  current setting.

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

### 5.3 Reconstruction FT: Historical File-Split Results Favor `g2`, But Official Rerun Is Still Required

Full FT results from the full300 reconstruction families:

| line | `obj_bg` | `obj_only` | `pb_t50_rs` | read |
|---|---:|---:|---:|---|
| v1 family reference | `0.8262` | `0.8417` | `0.7845` | historical internal reference under earlier file-split FT policy |
| `reconbest` full300 (`g0`) | `0.8399` | `0.8348` | `0.8102` | historical file-split FT readout |
| `recong2` full300 (`g2`) | `0.8485` | `0.8589` | `0.8140` | historical file-split FT readout; best observed transfer line so far |

Interpretation:

- full300 reconstruction is no longer just a positive-diag probe; it transfers,
- the no-generator `g0` line is a valid reconstruction baseline,
- the generator-enabled `g2` line is still the strongest PatchNEPA v2 line,
- however, these specific FT numbers were obtained under the earlier
  `file`-split FT policy,
- therefore the benchmark-valid conclusion is not yet "`g2` is solved", but
  rather "`g2` is the right rerun candidate under official `test-as-val`".

### 5.4 Translation-Loss Screen + Mini-CPAC: Split Signal, Not Mainline Yet

The short translation-centric screen (`MAX_STEPS=2000`, `pc33mesh33udf33`,
`g0`, `recon_chamfer`) now has both pretrain diags and mini-CPAC readout.

Pretrain readout:

| mode | `recon_lift_q` | `recon_lift_a` | `target_std_mean` | short read |
|---|---:|---:|---:|---|
| `composite` | `+0.1371` | `0.1132` | `0.1423` | best balanced reconstruction baseline |
| `answer_only` | `-0.2415` | `0.1133` | `0.3654` | keeps A-side lift, breaks Q-side lift |
| `context_plus_answer` | `-0.1777` | `0.1130` | `0.1443` | also breaks Q-side lift |

Mini-CPAC readout (`PC context -> UDF query`, `64/64` shapes):

| mode | `iou@0.01` | `mae` | `rmse` | short read |
|---|---:|---:|---:|---|
| `composite` | `0.0948` | `0.07585` | `0.09929` | best RMSE |
| `answer_only` | `0.1033` | `0.07584` | `0.09991` | best IoU |
| `context_plus_answer` | `0.0954` | `0.07653` | `0.10043` | weakest overall |

Interpretation:

- classification and full reconstruction currently still favor `composite`,
- mini-CPAC favors `answer_only` on the thresholded IoU readout,
- this is promising for translation-style behavior, but not enough to promote
  `answer_only` to the mainline yet.

### 5.5 PointGPT-Parity Controls: What Is Now Aligned

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

### 5.6 Explicit-Query CQA: `udf_distance` Is Now a Real Method Line, and `independent` Is the Strongest Factorization

What is now established on frozen `world_v3`:

- same-context `udf_distance` is stable and well above majority,
- zero-shot off-diagonal `surf -> pc_bank` transfer is real,
- dense completion and mesh-side readout are both meaningful,
- the seed-pack confirms the read is not a single-seed artifact.

Factorization and ordering study:

- shuffled:
  - `AR`: same/offdiag token acc `0.3173 / 0.1766`
  - `joint non-AR ("parallel")`: `0.3790 / 0.2057`
  - `independent`: `0.3898 / 0.2235`
- ordered-query ablation does **not** rescue AR:
  - ordered AR remains below ordered parallel,
  - shuffled parallel still gives the best overall completion row.

Interpretation:

- the primitive-native **Q/A schema** is doing the main work,
- AR is viable, but the current evidence does **not** show it is necessary,
- answer-to-answer interaction is also not necessary under the present
  `udf_distance` setup.

### 5.7 Q-Block and Target-Design Ablations Narrow the Minimal Interface

Query-block ablation (`full_q / self_q / no_q`) under `independent`:

- `full_q` remains the best row,
- but `self_q` and even `no_q` stay close on same/offdiag controls and
  completion,
- therefore the strongest current read is:
  - shared context + per-query anchor is the main mechanism,
  - full query-list conditioning helps only marginally.

Continuous target-design ablation (`udf_distance` only):

- continuous scalar regression does **not** collapse,
- same-context field metrics are very strong,
- but off-diagonal and mesh-side metrics are mixed versus the discrete
  independent line.

Interpretation:

- fixed-target promptable answering does not inherently require discrete CE,
- but continuous target design is currently best treated as a serious ablation,
  not yet as the canonical mainline.

### 5.8 Format Baselines and the First Shared Multi-Type Gate

Format baselines:

- both `k-plane` and `tri-plane` can solve the task under the same
  `world_v3 + udf_distance` protocol,
- but the strongest completion row still belongs to CQA,
- the best P4a baseline is currently `tri-plane` on MAE/RMSE and `k-plane` on
  IoU, not a full replacement for CQA.

First shared multi-type gate (`DISTANCE + NORMAL`, discrete):

- the shared checkpoint is alive on both tasks:
  - `udf_distance` stays strong,
  - `mesh_normal` stays above majority on same/offdiag, though weaker.
- the TYPE-switch asset path also works technically from the same frozen model
  and same context.
- however, the current `mesh_normal` qualitative quality is too weak for a
  paper-face figure.

Unsigned-normal rescue (`DISTANCE + NORMAL_UNSIGNED`, discrete):

- simple hemisphere folding rescues the mesh branch substantially:
  - same/offdiag `mesh_normal_unsigned acc = 0.5773 / 0.3832`
  - versus signed `mesh_normal acc = 0.3181 / 0.2119`
- `udf_distance` stays stable at the same time:
  - same/offdiag `acc = 0.3877 / 0.2005`
- TYPE-switch assets become visibly stronger for the mesh task, so the
  first shared multi-type line is no longer bottlenecked mainly by signed
  winding noise.

Unsigned-normal rescue (`DISTANCE + NORMAL_UNSIGNED`, continuous):

- the same sign fix also rescues the shared continuous branch:
  - `mesh_normal_unsigned mean_cos = 0.7954 / 0.6829`
    on same/offdiag
  - versus signed shared continuous `mesh_normal mean_cos = 0.0023 / 0.0084`
- `udf_distance` stays viable at the same time
  (`same/offdiag MAE = 0.0305 / 0.1229`).
- interpretation:
  - signed-target pathology, not continuous regression itself, was the main
    reason the earlier shared continuous `mesh_normal` line failed.
  - however, the discrete unsigned shared line is still the cleaner current
    mainline for multi-type promptability.

Shared continuous `DISTANCE + NORMAL`:

- `udf_distance` stays alive,
- `mesh_normal` effectively fails,
- so typed continuous regression is currently task-specific rather than a ready
  shared multi-type replacement.

Thickness rescue (`DISTANCE + THICKNESS_VALID_QBIN`, discrete):

- strict valid-support filtering plus quantile bins fixes the old thickness
  pathology:
  - rescue audit drops the old majority-heavy target to
    `majority_baseline_acc=0.0564`, `entropy_bits=5.45`, `unique_codes=64`
  - same/offdiag shared run then gives
    `udf_thickness_valid_qbin acc = 0.0921 / 0.0449`
    versus majority `0.0241 / 0.0236`
- `udf_distance` remains healthy in the same shared checkpoint
  (`same/offdiag acc = 0.3531 / 0.1890`).
- interpretation:
  - thickness is no longer blocked by zero-dominant support;
  - it is now a viable second UDF-family answer candidate, even though it
    remains weaker than `udf_distance`.

`mesh_viscount` first pass (`DISTANCE + NORMAL_UNSIGNED + VISCOUNT`, discrete):

- the short shared smoke is negative for `mesh_viscount` itself:
  - same/offdiag `acc = 0.5363 / 0.5340`
    versus majority `0.5361 / 0.5364`
  - control deltas are almost inert
    (`delta_ce(no_context) = +0.0925 / +0.0722`)
- `udf_distance` and `mesh_normal_unsigned` stay alive in the same smoke, so
  the failure is specific to viscount rather than the rest of the branch.
- interpretation:
  - naive discrete viscount is not yet a headline-safe second mesh-family
    answer;
  - AO and/or continuous scalar mesh answers remain better next candidates.

`mesh_ao` first pass (continuous scalar):

- unlike `mesh_viscount`, AO survives its first same/offdiag control screen:
  - same/offdiag `MAE = 0.1821 / 0.1985`
  - positive `no_context` deltas on both settings
  - `wrong_shape_other > wrong_shape_same` on both settings
- interpretation:
  - AO is the first smooth mesh-family scalar that looks scientifically alive
    under the current promptable recipe;
  - it is therefore the best current candidate for the second mesh-family
    answer after `normal_unsigned`.
  - however, its prediction variance is still compressed relative to the
    target, so it should be treated as a positive first pass rather than a
    fully mature headline row.

Shared continuous `DISTANCE + NORMAL_UNSIGNED + AO`:

- the first mesh-two-answer shared line now survives:
  - same-context:
    - `udf_distance MAE = 0.0360`, `IoU@0.05 = 0.7030`
    - `mesh_normal_unsigned mean_cos = 0.7776`
    - `mesh_ao MAE = 0.1927`
  - off-diagonal:
    - `udf_distance MAE = 0.0948`, `IoU@0.05 = 0.4858`
    - `mesh_normal_unsigned mean_cos = 0.6828`
    - `mesh_ao MAE = 0.1988`
- interpretation:
  - adding AO slightly weakens same-context `DISTANCE + NORMAL_UNSIGNED`,
    but does not collapse the existing UDF or mesh-normal reads;
  - AO itself remains context-sensitive inside the shared checkpoint;
  - this makes the branch the first viable shared line with two mesh-family
    answers, even if it is not yet the safest publishable mainline.

Loss-balanced shared AO reruns:

- balancing the shared continuous losses confirms that task-scale mismatch was
  real:
  - both `ema_norm` and fixed weights recover same-context `udf_distance`
    strongly relative to the raw shared AO branch.
- but they pay for it on off-diagonal transfer:
  - the raw branch remains best on off-diagonal `udf_distance`,
  - and also best on off-diagonal `mesh_normal_unsigned`.
- among the balanced variants, fixed weights are safer than `ema_norm`:
  - they keep more of the off-diagonal read while still recovering most of the
    same-context loss.

Current safest CQA read:

- strongest single line: `independent + shuffled + full_q` on `udf_distance`,
- first multi-type gate: `DISTANCE + NORMAL_UNSIGNED` discrete shared checkpoint,
- viable second UDF-family candidate:
  `DISTANCE + THICKNESS_VALID_QBIN`,
- thickness-128 codec check (`cqa_v3`) is informative but not promotable:
  it raises rescued-thickness entropy, yet the shared `DISTANCE + THICKNESS`
  line becomes worse than the current 64-bin rescue and should not replace it,
- viable second mesh-family candidate:
  continuous `mesh_ao`,
- first viable mesh-two-answer shared expansion:
  continuous `DISTANCE + NORMAL_UNSIGNED + AO`,
- current best explanation of the AO tradeoff:
  task-loss scale mismatch is real, but the raw unbalanced branch still gives
  the best off-diagonal robustness,
- current limitation: multi-type promptability is now real, but the mesh side
  still needs figure-quality polishing and an additional mesh-family
  answer beyond unsigned normals; continuous multi-type is now clearly
  supportive rather than negative, but the discrete shared
  `DISTANCE + NORMAL_UNSIGNED` branch is still the safer canonical mainline.

### 5.9 Frozen Geometric Probes and the Next Raw-Target Wave

Frozen linear probes from `answer_hidden`:

- curvature is the first clearly positive unseen-quantity probe:
  - `C034` same-context curvature probe:
    - `MAE/RMSE/r = 0.1482 / 0.1988 / 0.4821`
    - versus scalar mean baseline `0.1870 / 0.2219`
  - `C035` same-context curvature probe:
    - `0.1517 / 0.1999 / 0.4514`
    - also above the same baseline
  - both show strong control sensitivity on same-context
    (`delta_mae(no_context) ~= +0.14`)
- off-diagonal curvature is only partial:
  - `C034` offdiag MAE is still worse than the mean baseline
  - `C035` nearly matches the baseline on MAE and improves Pearson `r`
    (`0.2210`), so the 4-task branch may help correlation slightly
  - interpretation:
    - frozen CQA representations do expose unseen scalar geometry,
    - but cross-carrier curvature emergence is still limited

Signed-normal frozen probe on the winding-consistent subset:

- remains effectively absent even after filtering to `is_winding_consistent=1`
  shapes:
  - `C034` same/offdiag `mean_cos = 0.0094 / 0.0032`
  - `C035` same/offdiag `0.0090 / 0.0049`
- several off-diagonal controls improve rather than degrade cosine slightly,
  so the read is not directionally clean
- interpretation:
  - current CQA representations do **not** linearly expose signed orientation;
    keep this as a negative analysis result

AO-HQ and HKS raw-target smoke:

- additive derived-cache augmentation is the correct mechanism for data-side
  target upgrades:
  - canonical `world_v3` stays untouched
- AO-HQ is clearly promising:
  - full `64/64` derived-subset success
  - old AO has only `7` rounded support levels with `std=0.219`
  - AO-HQ expands to `129` rounded levels with `std=0.433`
- HKS is scientifically interesting but operationally fragile:
  - `48/64` subset shapes receive `mesh_surf_hks_t0`
  - ARPACK failures remain common on harder meshes
- interpretation:
  - the next raw-target wave should prioritize **AO-HQ**
  - HKS stays smoke-only until the eigensolver path is stabilized

AO-HQ full build and discrete reruns:

- the full additive AO-HQ build now lands cleanly on the real corpus:
  - `train_mesh 16004/16004`
  - `eval 5241/5241`
  - `overall 21245/21245`
  - `0` errors
  - canonical `world_v3` remains untouched
- `DISTANCE + NORMAL_UNSIGNED + AO_HQ` (`C042`) is supportive but not yet a
  headline-safe mesh-two-answer line:
  - same/offdiag token acc:
    - `udf_distance = 0.1464 / 0.0688`
    - `mesh_normal_unsigned = 0.5288 / 0.3808`
    - `mesh_ao_hq = 0.5650 / 0.5018`
  - `mesh_ao_hq` only barely beats majority on same-context and still fails it
    off-diagonal:
    - same `0.5650` vs majority `0.5624`
    - offdiag `0.5018` vs majority `0.5535`
  - same/offdiag completion:
    - `MAE = 0.0227 / 0.1214`
    - `IoU@0.05 = 0.6776 / 0.3657`
    - `mesh_fscore = 0.0999 / 0.0562`
- the AO-HQ 4-task line (`C043`) is a mild improvement over the raw-AO
  4-task ceiling, but still not a new mainline:
  - same/offdiag token acc:
    - `udf_distance = 0.1144 / 0.0613`
    - `mesh_normal_unsigned = 0.5021 / 0.4005`
    - `udf_thickness_valid_qbin = 0.0778 / 0.0504`
    - `mesh_ao_hq = 0.5624 / 0.5289`
  - same/offdiag completion:
    - `MAE = 0.0324 / 0.1066`
    - `IoU@0.05 = 0.5974 / 0.3402`
    - `mesh_fscore = 0.0604 / 0.0388`
  - `mesh_ao_hq` still stays at/below majority (`0.5624 / 0.5535`)
- interpretation:
  - AO-HQ genuinely improves the target and slightly improves the mesh-side
    branch
  - but it does **not** overturn the safer discrete mainline, which remains
    `prefixlm + cqa_v2 DISTANCE + NORMAL_UNSIGNED`

Current discrete mainline-vs-utility snapshot:

| branch | answers | token acc (`dist`, `normal`) same/offdiag | completion `MAE` same/offdiag | completion `IoU@0.05` same/offdiag | utility (`obj_bg / obj_only / pb_t50_rs`) | current role |
|---|---|---|---|---|---|---|
| `C034` | `dist + normal_unsigned` | `0.1745/0.0797`, `0.5495/0.3566` | `0.0192 / 0.1249` | `0.7030 / 0.3814` | `0.8399 / 0.8503 / 0.7679` | safest core mainline |
| `C035` | `dist + normal_unsigned + thickness + AO(raw)` | `0.1232/0.0601`, `0.5091/0.3841` | `0.0290 / 0.1161` | `0.6320 / 0.3583` | `0.8451 / 0.8520 / 0.7710` | raw 4-task ceiling; utility-up, core-down |
| `C042` | `dist + normal_unsigned + AO_HQ` | `0.1464/0.0688`, `0.5288/0.3808` | `0.0227 / 0.1214` | `0.6776 / 0.3657` | `0.8520 / 0.8485 / 0.7734` | strongest utility-supportive AO-HQ branch |
| `C043` | `dist + normal_unsigned + thickness + AO_HQ` | `0.1144/0.0613`, `0.5021/0.4005` | `0.0324 / 0.1066` | `0.5974 / 0.3402` | `0.8434 / 0.8468 / 0.7797` | 4-task AO-HQ tradeoff ceiling |

Reading of the table:

- adding answer families can improve downstream utility slightly, but the gain
  is **not monotonic** across variants
- none of `C035 / C042 / C043` beats `C034` on the core
  `distance + normal_unsigned` answering/completion anchor
- AO-HQ helps more than raw AO on the mesh-side target, but it still does not
  make the AO-family answer headline-safe off-diagonal
- therefore:
  - keep `C034` as the discrete method mainline
  - treat `C042/C043` as supportive evidence that richer probe sets can
    regularize the encoder and sometimes help utility

## 6. Practical Delta: PatchNEPA v2 vs PointGPT

The practical comparison should currently be read as:

| axis | PatchNEPA v2 current line | PointGPT reference |
|---|---|---|
| sequence/task | QA split with primitive-conditioned answer stream | single point-patch stream |
| main objective | full300 reconstruction path (`ctx(chamfer) + q/a(mse)`) with `g2` now mainline | reconstruction-dominant path |
| loss surface | composite recon on QA stream; translation-centric variants remain screening controls | patch reconstruction (`CD-L12`) |
| generator | present and now empirically useful on FT, but CPAC effect is still open | present and central |
| diagnostic space | `recon/copy/lift` for recon, cosine probes only for cosine path | reconstruction-aligned diagnostics |

This means:

- "PatchNEPA now has dual mask and Chamfer" is true,
- "PatchNEPA is already the same regime as PointGPT" is not yet true.

## 7. What Is Currently Established

- Query-NEPA is a historical line, not the active implementation line.
- PatchNEPA v1-family remains the best internal historical reference.
- PatchNEPA v2 cosine-family target repeatedly collapses in cosine diagnostics.
- PatchNEPA v2 reconstruction branch shows positive objective-aligned lift.
- `recong2` full300 is the best current v2 reconstruction branch.
- PatchNEPA v2 reconstruction + `g2` now exceeds the historical v1-family FT
  baseline on all three ScanObjectNN variants.
- the existing `worldvis` source cache has now been frozen as the `world_v3`
  contract with canonical quality/provenance fields and audit artifacts; this
  removes the need for an immediate full corpus rebuild.
- the strict watertight + winding-consistent subset is only `119 / 52311`, so
  it is a future pivot-side manifest rather than a replacement for the main
  training corpus.
- The explicit-query CQA branch now has one non-degenerate task:
  frozen-`world_v3` full-range single-task `udf_distance` reproduces above
  majority at `2k` and then strengthens sharply by `10k`, including strong
  `wrong_shape_same/other_synset` controls.
- a first discrete encoder-decoder CQA branch is now implemented and
  scientifically alive under the same `cqa_v2` interface, but a strict
  `prefixlm + cqa_v2` rerun confirms that prefixlm still wins the first
  architecture-controlled compare on the main `udf_distance` token/completion
  reads and on utility classification; the only approximately tied read is
  off-diagonal `mesh_normal_unsigned`.
- deeper enc-dec follow-ups do not change that read:
  - `decoder_layers=12` alone does not recover the missing `udf_distance`
    token/completion signal,
  - and adding a decoder-side `full_q` block only raises offdiag
    `mesh_normal_unsigned` slightly without closing the main gap to prefixlm.
- the current raw-target 4-task shared prefixlm line
  (`DISTANCE + NORMAL_UNSIGNED + THICKNESS_VALID_QBIN + AO`) is also now
  established as the present multi-probe ceiling: all probes except raw AO stay
  scientifically alive, and utility classification improves slightly, but the
  branch does **not** show monotonic synergy at `10k` because the main
  `udf_distance` / `mesh_normal_unsigned` token reads and distance completion
  regress relative to the stricter `DISTANCE + NORMAL_UNSIGNED` anchor.
- frozen query-conditioned CQA representations already support a positive
  same-context curvature linear probe from `answer_hidden`, so unseen scalar
  geometry is at least partially present in the learned representation.
- the same frozen representations do **not** linearly expose signed normals,
  even on a winding-consistent subset, so emergent orientation-sensitive
  geometry is not established.
- additive derived-cache augmentation for AO-HQ is now validated on a 64-shape
  subset without touching canonical `world_v3`, and the full AO-HQ additive
  build now also lands cleanly on the real `train_mesh + eval` corpus.
- AO-HQ reruns are supportive and slightly stronger than the old raw-AO branch,
  but they still do not make the mesh-side second answer headline-safe because
  off-diagonal `mesh_ao_hq` remains below majority in both the 3-task and
  4-task reruns.
- the first degraded-input completion suite on the current safest mainline
  (`C044`) is completed and negative as a headline robustness axis:
  CQA field metrics degrade smoothly, but BPA and Poisson remain clearly
  stronger on the chosen mesh metrics, so degraded completion is not yet the
  killer result for the paper.
- common-split protocol isolation is now completed on the
  `DISTANCE + NORMAL_UNSIGNED` anchor:
  - removing the train-split confound does **not** collapse the 2-type row,
  - and `packed all-type-per-shape` is a small positive over common-split
    `mixture` on the main field-facing reads,
  - but the gain is modest enough that it does not by itself explain the full
    2-type versus richer-answer tradeoff story.
- a first frozen external Point-MAE control now lands as a real non-internal
  reference on `pc_bank -> udf_distance`, staying clearly above majority on
  both `surf` and `pc_bank` eval and yielding nontrivial completion, but the
  current row is still a `cqa_v1` single-task distance harness and therefore
  benchmark context rather than a strict apples-to-apples replacement for the
  `cqa_v2 DISTANCE + NORMAL_UNSIGNED` mainline.
- HKS smoke remains only partially successful because eigensolver failures are
  still common.
- the same `surf`-trained `udf_distance` checkpoint now transfers zero-shot to
  `pc_bank -> udf_distance` at eval time only, still above majority and with
  positive `no_context` / `wrong_shape_other_synset` deltas.
- the `udf_distance` mainline and off-diagonal reads now reproduce across
  seeds `0/1/2`, so the branch is no longer a single-seed provisional result.
- a paired `pc_bank -> udf_distance` upper bound is also positive
  (`acc=0.2533` vs majority `0.0375`), but it is best treated as a diagnostic
  ceiling rather than replacing the cleaner zero-shot off-diagonal story.
- `udf_distance` CQA also now has method-native dense-grid completion evidence:
  a held-out `grid_res=12` pilot gives `MAE=0.0479`, `RMSE=0.0638`,
  `IoU@0.05=0.5712`.
- ordered-query ablations now show that simply restoring a spatial query order
  does **not** rescue AR:
  ordered AR stays below ordered parallel, and the strongest completion row is
  still shuffled-parallel. The safest read is that the primitive-native Q/A
  interface matters more than causal factorization under the current design.
- Point-MAE is the protocol sanity / benchmark context.
- PointGPT is the reconstruction-style reference for the next causal tests.

## 8. What Is Still Open

- whether the `g2` gain also carries over to CPAC / cross-primitive evaluation,
- whether the `answer_only` CPAC advantage can be retained without damaging FT
  transfer,
- whether strict PointGPT-loss parity plus generator depth reproduces the
  expected training dynamics on PatchNEPA,
- whether any cosine-family target on the v2 token path can escape the
  `cos_tgt ~= cos_prev` regime,
- whether Q/A composite structure helps or hurts transfer once generator-based
  recon is working,
- whether true train-sampling parity changes the current negative
  `fps_then_sample` ablation once a real `point_all > npoints` path exists,
- whether FT-side recipe follow-ups such as rotation materially change the
  current external-gap readout,
- whether the explicit-query CQA branch can turn the current full-range
  `udf_distance` signal into a broader task family beyond its current anchor
  + rescued-thickness UDF branch, without destabilizing the distance anchor
  when thickness resolution is increased,
- whether encoder-decoder can be made competitive with the current prefixlm
  mainline once training scale or a more substantial query-conditioned design
  is changed, since simply using `decoder_layers=12` and decoder-side `full_q`
  still leaves it clearly below the prefixlm anchor,
- whether AO target quality beyond the current AO-HQ upgrade can finally turn
  the mesh-side second answer into a headline-safe line, since AO-HQ improves
  same-context behavior but off-diagonal `mesh_ao_hq` still sits below its
  majority-heavy baseline,
- whether a redesigned robustness protocol can turn degraded-input completion
  into a real headline axis, since the first `C044` wave is already a
  completed negative result against Open3D baselines,
- whether the small packed-over-mixture gain scales up beyond the current
  common-split `2-type` protocol-isolation row, especially on the cleaner
  `distance + normal + thickness_valid_qbin` bridge before AO-HQ is re-added,
- whether the external Point-MAE control can be moved from the current
  `cqa_v1` single-task distance harness into a stricter `cqa_v2` matched
  compare, so the external baseline is more than benchmark context,
- whether HKS can be stabilized enough to become a real mesh-native follow-up
  probe rather than remaining a numerically fragile smoke result,
- whether a redesigned task/target definition can increase context dependence
  without repeating the negative `near_surface` collapse or the negative
  `pc_bank` retry, especially for second mesh-family answers beyond
  `mesh_normal_unsigned`,
- how far the positive off-diagonal `pc_bank -> udf_distance` read survives
  under stronger completion settings and broader query regimes,
- how to package the current cosine/reconstruction diagnostics into a stable
  paper-ready metric sheet.

Operational execution order for these open items now lives in:

- `nepa3d/docs/patch_nepa/execution_backlog_active.md`

## 9. Practical Reading Order for New Analysis

1. `nepa3d/docs/llm_retrieval_index.md`
2. `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
3. `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`
4. `nepa3d/docs/patch_nepa/execution_backlog_active.md`
5. `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
6. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
7. `nepa3d/docs/query_nepa/runlog_202602.md`
