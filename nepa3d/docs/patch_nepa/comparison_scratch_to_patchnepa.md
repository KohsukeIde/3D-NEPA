# Scratch -> PatchNEPA Comparison Matrix

Last updated: 2026-03-02

## 1. Recommended Axes

For this project, use **two tables** with different axes.

### 1.1 Primary leaderboard (for headline discussion)

- **X-axis (columns):** dataset variants (`obj_bg`, `obj_only`, `pb_t50_rs`, `macro_mean`)
- **Y-axis (rows):** method/recipe line (`scratch`, `PatchNEPA pretrain+FT` variants)

Reason:
- This directly answers "which line is stronger" with the same output surface.
- `macro_mean` prevents overfitting discussion to only one variant.

### 1.2 Factor A/B table (for causal diagnosis)

- **X-axis (columns):** metrics (`delta_obj_bg`, `delta_obj_only`, `delta_pb_t50_rs`, `delta_macro`)
- **Y-axis (rows):** one-factor changes (`ray on/off`, `E100/E300`, `encdec1/dualmask`, etc.)

Reason:
- This isolates what actually moved the score.
- It keeps analysis falsifiable (one change per row).

## 2. Primary Leaderboard (Scratch -> PatchNEPA)

All rows below are protocol-valid (`val_split_mode=file`, `aug_eval=1`, `mc_test=10`) unless explicitly noted.

| line | recipe / source | obj_bg | obj_only | pb_t50_rs | macro_mean | note |
|---|---|---:|---:|---:|---:|---|
| external reference | Point-MAE SSL target row (README) | 0.9002 | 0.8829 | 0.8518 | 0.8783 | benchmark target (`benchmark_scanobjectnn_variant.md`) |
| external sanity | Point-MAE official finetuned ckpt (`--test`, variant-aligned) | 0.9019 | 0.8795 | 0.8459 | 0.8758 | `obj_bg/obj_only` は ckswap 検証 (`100752/100753`) |
| strong scratch ref | Point-MAE strict retry5 scratch | 0.8296 | 0.8399 | 0.8012 | 0.8236 | jobs `99275/99277/99279` + tests `99276/99278/99280` |
| scratch baseline | PatchCls PM-aligned scratch | 0.7831 | 0.8176 | 0.7609 | 0.7872 | run set `patchcls_scan3_scratch_pmalign_20260227_202814` |
| PatchNEPA (ray, E100) | split-x2 dualmask baseline direct FT | 0.7900 | 0.8193 | 0.7519 | 0.7871 | `100194/100195/100196` |
| PatchNEPA (ray, E300) | dualmask split_sep direct FT | 0.8072 | 0.7866 | 0.7720 | 0.7886 | `100425/100426/100427` |
| PatchNEPA (ray, early mainline) | direct FT from ray pretrain | 0.7797 | 0.7952 | 0.7582 | 0.7777 | `100148/100150/100152` |
| PatchNEPA (point-only control) | direct FT from point-only pretrain | 0.7590 | 0.7797 | 0.7380 | 0.7589 | `100155/100156/100157` |

## 3. Factor A/B Table (Single-change deltas)

`delta = (B - A)`.

| factor | A | B | delta_obj_bg | delta_obj_only | delta_pb_t50_rs | delta_macro | interpretation |
|---|---|---|---:|---:|---:|---:|---|
| ray contribution | point-only control (`100155/156/157`) | ray early mainline (`100148/150/152`) | +0.0207 | +0.0155 | +0.0202 | +0.0188 | ray path is net positive on all 3 variants in this pair |
| masking style (E100) | encdec1 baseline (`100188/189/190`) | dualmask baseline (`100194/195/196`) | +0.0103 | +0.0293 | +0.0017 | +0.0138 | dualmask wins in this matched E100 pair |
| pretrain length (ray dualmask) | E100 (`100194/195/196`) | E300 (`100426/425/427`) | +0.0172 | -0.0327 | +0.0201 | +0.0015 | longer pretrain helped `obj_bg/pb_t50_rs`, hurt `obj_only` |

### 3.2 Dual-mask and Loss-Curve Shape (Pretrain-side)

Question:
- does dual-mask change the *initial* loss collapse pattern?

Matched sanity pair (pretrain, `mr0` logs):
- off: `patchnepa_stage2_sanity32_20260301_162811_lS_d0_ta1_tp0_r0`
- on:  `patchnepa_stage2_sanity32_20260301_162811_lS_d1_ta1_tp0_r0`

Observed:
- step0 loss is identical (`0.9750`).
- step100 loss is also almost identical:
  - off: `0.0048597`
  - on:  `0.0048567`
- interpretation: dual-mask is not the dominant cause of early rapid drop.

Later-phase tendency (same pair):
- late segment mean (roughly step `1450-4000`) is higher for dual-mask on.
- this suggests dual-mask influences sustained optimization pressure more than initial collapse.

Caveat:
- strict Q1 random point-only on/off pair is still missing (Q1 mainline has on only).
- so this is a nearest-match inference, not the final Q1-causal verdict.

### 3.1 Serialization (`serial` / `serial_ztrans`) delta summary

These are `PatchCls scratch`-side comparisons for `pb_t50_rs` only (not direct PatchNEPA FT).

| factor | A | B | delta_pb_t50_rs | interpretation |
|---|---|---|---:|---|
| serialization vs fps_knn (main scratch baseline) | `pb_t50_rs` scratch fps_knn (`0.7609`, `patchcls_scan3_scratch_pmalign_20260227_202814`) | `pb_t50_rs_serial` (`0.6541`, `patchcls_pb_t50_rs_serial_fix1_20260228_052424`) | `-0.1068` | large drop vs main scratch baseline |
| serialization(ztrans) vs fps_knn (main scratch baseline) | `pb_t50_rs` scratch fps_knn (`0.7609`) | `pb_t50_rs_serial_ztrans` (`0.6489`, `patchcls_pb_t50_rs_serial_ztrans_20260228_053913`) | `-0.1120` | slightly worse than serial(morton) |
| serialization vs fair-fps16 (same runset control) | `pc_pb_t50_fair_fps16` (`0.6662`, `patchcls_pb_t50_rs_fair_g64s16_file_20260228_064306`) | `pc_pb_t50_fair_ser_m` (`0.6541`) | `-0.0121` | drop remains under fair g64s16 control |
| serialization(ztrans) vs fair-fps16 (same runset control) | `pc_pb_t50_fair_fps16` (`0.6662`) | `pc_pb_t50_fair_ser_zt` (`0.6489`) | `-0.0173` | ztrans also below fair-fps16 |


## 4. Validity Rules for This Table

- Include only runs with:
  - finished pretrain checkpoint path confirmed
  - finished FT with `TEST acc` present
  - same evaluation protocol (`file + TTA10`)
- Exclude rows where upstream pretrain failed (`SIGABRT`, NCCL/ECC, missing checkpoint), even if dependent FT produced a number.

## 5. Source Pointers

- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`


## 6. FPS-Comparison Branch Status (Pretrain Sanity)

This branch is for causal diagnosis of patch sampling / ray grouping; pretrains are complete (`100699`, `100741`) and strict missing-ray A/B for `obj_only` is now available (`100742` vs `100750`).

| branch | job | setting delta | status | usable for final comparison |
|---|---|---|---|---|
| fps baseline (no aug) | `100563` | `pt_sample_mode=fps`, bind ray | completed | yes (pretrain-side) |
| random baseline (no aug) | `100564` | `pt_sample_mode=random`, bind ray | completed | yes (pretrain-side) |
| fps + aug (dualmask off) | `100642` | PM-like aug only | terminated by user (`Exit_status=265`) | no |
| fps + aug + dualmask (old) | `100643` | PM-like aug + dualmask | finished (`Exit_status=97`) | no (launcher marker missing / invalid) |
| fps + aug + dualmask (fixed rerun, bind) | `100741` | `ray_assign_mode=proxy_sphere`, `ray_num_groups=32` | completed (`Exit_status=0`) | yes (matched FT: `100750=0.7367`) |
| independent ray patch + aug + dualmask | `100699` | `ray_assign_mode=independent_fps_knn`, `ray_num_groups=32` | completed (`Exit_status=0`) | yes (matched FT: `100742=0.7762`) |

Step-0 token sanity (sample-0):

- bind rerun (`100741` log): `Q_RAY=21`, `A_RAY=21`, `MISSING_RAY=86`
- independent patch (`100699` log): `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`
- reduction: `MISSING_RAY 86 -> 0` (`-100%`)

Strict missing-ray A/B (same FT recipe: `qa_zeroa + cls + linear + llrd off`):

- independent branch: `100699 -> 100742` (`obj_only=0.7762`)
- bind branch: `100741 -> 100750` (`obj_only=0.7367`)
- delta (`independent - bind`) = `+0.0395`

Comparison caveat:

- the above A/B is strict only for this one FT recipe and `obj_only`.
- commonly cited bind score `0.8193` is from `100181` family (`rfps_cached + no aug` and different FT readout), so it must not be mixed into this A/B.

Note on the augmentation question:

- there is no evidence that augmentation itself is broken.
- the only ended aug-only run (`100642`) was user-terminated; it is excluded from validity.

### 6.1 Newly Completed FT Jobs (2026-03-02)

| job | runset | variant | test_acc | note |
|---|---|---|---:|---|
| `100747` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `obj_bg` | `0.7711` | point-only + EMA rerun, PM-head defaults |
| `100748` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `obj_only` | `0.7659` | point-only + EMA rerun, PM-head defaults |
| `100750` | `patchnepaFT_bindfix_llrdoff_default_20260302_005751` | `obj_only` | `0.7367` | strict bind-side counterpart of `100742` |

Pending:

- `100749` (`pb_t50_rs`, runset `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316`) is still running; no final `TEST acc` yet.

## 7. q_only Completed Results (Direct PatchNEPA FT)

- completed rows: `9`
- includes `cls_token` so BOS代替実験 (`last_q/eos`) が識別可能

| variant | test_acc | pooling | cls_token | is_causal | runset | log |
|---|---:|---|---|---|---|---|
| `obj_bg` | `0.7986` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_bg.out` |
| `obj_bg` | `0.7797` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_bg.out` |
| `obj_bg` | `0.7573` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_bg.out` |
| `obj_only` | `0.8090` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_only.out` |
| `obj_only` | `0.7814` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_only.out` |
| `obj_only` | `0.7797` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_only.out` |
| `pb_t50_rs` | `0.7533` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7512` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7488` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/pb_t50_rs.out` |

## 8. Full Completed Direct-FT Ledger (Raw)

- completed rows: `62`
- this is the full raw ledger for direct PatchNEPA FT outputs; mainline-valid subset remains in Section 2/3

| variant | test_acc | ft_mode | pooling | cls_token | is_causal | runset | log |
|---|---:|---|---|---|---|---|---|
| `obj_bg` | `0.8072` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepa_ft` | `logs/sanity/patchnepa_ft/obj_bg.out` |
| `obj_bg` | `0.7986` | `q_only` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_bg.out` |
| `obj_bg` | `0.7900` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_dualmask_baseline_20260301_040740` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740/obj_bg.out` |
| `obj_bg` | `0.7797` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_bg.out` |
| `obj_bg` | `0.7797` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_encdec1_baseline_20260301_040736` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_baseline_20260301_040736/obj_bg.out` |
| `obj_bg` | `0.7573` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_bg.out` |
| `obj_bg` | `0.7487` | `-` | `cls_max` | `-` | `True` | `patchnepaFT_from_ray_causal_probe_20260301_020406` | `logs/sanity/patchnepa_ft/patchnepaFT_from_ray_causal_probe_20260301_020406/obj_bg.out` |
| `obj_only` | `0.8193` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_dualmask_baseline_20260301_040740` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740/obj_only.out` |
| `obj_only` | `0.8158` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromB768_E300_retry_20260301_230030` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromB768_E300_retry_20260301_230030/obj_only.out` |
| `obj_only` | `0.8090` | `q_only` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_only.out` |
| `obj_only` | `0.8072` | `-` | `cls_max` | `-` | `False` | `patchnepa_refactor_repro1_20260301_021900` | `logs/sanity/patchnepa_ft/patchnepa_refactor_repro1_20260301_021900/obj_only.out` |
| `obj_only` | `0.7969` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromEMA100_ema_20260301_170323` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromEMA100_ema_20260301_170323/obj_only.out` |
| `obj_only` | `0.7900` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_encdec1_baseline_20260301_040736` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_baseline_20260301_040736/obj_only.out` |
| `obj_only` | `0.7883` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromB768_probe_retry_20260301_230010` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromB768_probe_retry_20260301_230010/obj_only.out` |
| `obj_only` | `0.7866` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromE300_dualmask_splitsep_20260301_170815` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromE300_dualmask_splitsep_20260301_170815/obj_only.out` |
| `obj_only` | `0.7814` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_only.out` |
| `obj_only` | `0.7797` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_only.out` |
| `obj_only` | `0.7711` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_objonly_encdec1_lastq_llrd_20260301_154822` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_encdec1_lastq_llrd_20260301_154822/obj_only.out` |
| `obj_only` | `0.7676` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_ablate_objonly_static_lastq_freeze1_20260301_170205` | `logs/sanity/patchnepa_ft/patchnepaFT_ablate_objonly_static_lastq_freeze1_20260301_170205/obj_only.out` |
| `obj_only` | `0.7676` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_objonly_dualmask_lastq_llrd_20260301_154828` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_dualmask_lastq_llrd_20260301_154828/obj_only.out` |
| `obj_only` | `0.7642` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_objonly_interleave_fix_gb128_20260301_173714` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_interleave_fix_gb128_20260301_173714/obj_only.out` |
| `obj_only` | `0.7435` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_from_fpscmp_rand_20260301_205548` | `logs/sanity/patchnepa_ft/patchnepaFT_from_fpscmp_rand_20260301_205548/obj_only.out` |
| `obj_only` | `0.7401` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromEMA100_raw_20260301_170323` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromEMA100_raw_20260301_170323/obj_only.out` |
| `obj_only` | `0.7315` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromB768_E300_20260301_165504` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromB768_E300_20260301_165504/obj_only.out` |
| `obj_only` | `0.7315` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepaFT_objonly_fromB768_probe_20260301_165321` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromB768_probe_20260301_165321/obj_only.out` |
| `obj_only` | `0.7298` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_ablate_objonly_base_llrdcw_lastq_freeze1_20260301_170200` | `logs/sanity/patchnepa_ft/patchnepaFT_ablate_objonly_base_llrdcw_lastq_freeze1_20260301_170200/obj_only.out` |
| `obj_only` | `0.7298` | `qa_zeroa` | `cls` | `bos` | `False` | `patchnepaFT_ablate_objonly_llrdcw_bos_freeze1_20260301_170210` | `logs/sanity/patchnepa_ft/patchnepaFT_ablate_objonly_llrdcw_bos_freeze1_20260301_170210/obj_only.out` |
| `obj_only` | `0.7160` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_objonly_from_skipk4_gb128_20260301_173714` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_skipk4_gb128_20260301_173714/obj_only.out` |
| `obj_only` | `0.7108` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_ablate_objonly_llrdcw_lastq_freeze0_20260301_170215` | `logs/sanity/patchnepa_ft/patchnepaFT_ablate_objonly_llrdcw_lastq_freeze0_20260301_170215/obj_only.out` |
| `obj_only` | `0.7108` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_objonly_from_multik124_gb128_20260301_174748` | `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_multik124_gb128_20260301_174748/obj_only.out` |
| `obj_only` | `0.6833` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_random_off_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_random_off_E12/obj_only.out` |
| `obj_only` | `0.6454` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_fps_off_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_fps_off_E12/obj_only.out` |
| `obj_only` | `0.6420` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepaFT_from_fpscmp_fps_20260301_205540` | `logs/sanity/patchnepa_ft/patchnepaFT_from_fpscmp_fps_20260301_205540/obj_only.out` |
| `obj_only` | `0.6368` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_rfps_cached_off_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_rfps_cached_off_E12/obj_only.out` |
| `obj_only` | `0.6317` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_rfps_cached_on_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_rfps_cached_on_E12/obj_only.out` |
| `obj_only` | `0.6196` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_random_on_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_random_on_E12/obj_only.out` |
| `obj_only` | `0.5938` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp0_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp0_r0/obj_only.out` |
| `obj_only` | `0.5594` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp1_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp1_r0/obj_only.out` |
| `obj_only` | `0.5594` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp1_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp1_r0/obj_only.out` |
| `obj_only` | `0.5594` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp1_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp1_r0/obj_only.out` |
| `obj_only` | `0.5525` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp0_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp0_r0/obj_only.out` |
| `obj_only` | `0.5525` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp0_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp0_r0/obj_only.out` |
| `obj_only` | `0.5318` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_sanity6_20260301_224515_ft_rayind_fps_on_E12` | `logs/sanity/patchnepa_ft/patchnepa_sanity6_20260301_224515_ft_rayind_fps_on_E12/obj_only.out` |
| `obj_only` | `0.5095` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp0_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp0_r1/obj_only.out` |
| `obj_only` | `0.4699` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp0_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp0_r0/obj_only.out` |
| `obj_only` | `0.4406` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp1_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp1_r1/obj_only.out` |
| `obj_only` | `0.4406` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp1_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp1_r1/obj_only.out` |
| `obj_only` | `0.4355` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp1_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp1_r1/obj_only.out` |
| `obj_only` | `0.3993` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp0_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp0_r1/obj_only.out` |
| `obj_only` | `0.3941` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp1_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta1_tp1_r1/obj_only.out` |
| `obj_only` | `0.3838` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp1_r0` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d1_ta0_tp1_r0/obj_only.out` |
| `obj_only` | `0.3769` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp0_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta0_tp0_r1/obj_only.out` |
| `obj_only` | `0.3769` | `qa_zeroa` | `cls` | `last_q` | `False` | `patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp0_r1` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lS_d0_ta1_tp0_r1/obj_only.out` |
| `pb_t50_rs` | `0.7720` | `qa_zeroa` | `cls_max` | `bos` | `False` | `patchnepa_ft` | `logs/sanity/patchnepa_ft/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7533` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7519` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_dualmask_baseline_20260301_040740` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7512` | `q_only` | `cls_max` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7502` | `qa_zeroa` | `cls_max` | `-` | `False` | `patchnepaFT_splitx2_encdec1_baseline_20260301_040736` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_baseline_20260301_040736/pb_t50_rs.out` |
| `pb_t50_rs` | `0.7488` | `q_only` | `mean_q` | `-` | `False` | `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200` | `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/pb_t50_rs.out` |

## 9. Raw TSV Index

- `nepa3d/docs/patch_nepa/patchnepa_ft_completed_results.tsv` (59 rows, direct PatchNEPA FT)
- `nepa3d/docs/patch_nepa/patchcls_completed_results.tsv` (historical PatchCls/adapter/direct mixed)

## 10. Exhaustive Coverage Audit (No Drop Policy)

- policy: every `logs/sanity/patchnepa_ft/**/*.out` entry is tracked (completed + incomplete)
- direct-ft files scanned: `91`
- completed (`TEST acc` present): `59`
- incomplete (`TEST acc` absent): `32`

Coverage (all scanned):
- `patchnepa_ft_mode`: `-`=22, `q_only`=18, `qa_zeroa`=51
- `pooling`: `-`=20, `cls`=36, `cls_max`=29, `mean_q`=6
- `cls_token`: `-`=46, `bos`=10, `last_q`=35

Coverage (completed only):
- `patchnepa_ft_mode`: `-`=2, `q_only`=9, `qa_zeroa`=48
- `pooling`: `cls`=33, `cls_max`=23, `mean_q`=3
- `cls_token`: `-`=17, `bos`=10, `last_q`=32

Explicit check for requested settings:
- `mean_a`: **0 rows found** in scanned logs (未実施)
- `mean_q`: present
- `q_only`: present
- BOS代替 (`cls_token=last_q/eos`) は `cls_token` 列で追跡

## 11. Incomplete / No-TEST Rows (Still Tracked)

| variant | runset | ft_mode | pooling | cls_token | status | log |
|---|---|---|---|---|---|---|
| `obj_bg` | `patchnepaFT_from_ray_interleave_20260301_014535` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_from_ray_interleave_20260301_014535/obj_bg.out` |
| `obj_bg` | `patchnepaFT_splitx2_dualmask_qonly_20260301_040743` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_20260301_040743/obj_bg.out` |
| `obj_bg` | `patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745` | `q_only` | `mean_q` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745/obj_bg.out` |
| `obj_bg` | `patchnepaFT_splitx2_encdec1_qonly_20260301_040738` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_qonly_20260301_040738/obj_bg.out` |
| `obj_bg` | `patchnepa_ft_variants_20260301_223043` | `qa_zeroa` | `cls` | `last_q` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_ft_variants_20260301_223043/obj_bg.out` |
| `obj_only` | `patchnepaFT_from_ray_interleave_20260301_014535` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_from_ray_interleave_20260301_014535/obj_only.out` |
| `obj_only` | `patchnepaFT_splitx2_dualmask_qonly_20260301_040743` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_20260301_040743/obj_only.out` |
| `obj_only` | `patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745` | `q_only` | `mean_q` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745/obj_only.out` |
| `obj_only` | `patchnepaFT_splitx2_encdec1_qonly_20260301_040738` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_qonly_20260301_040738/obj_only.out` |
| `obj_only` | `patchnepa_ft_variants_20260301_223043` | `qa_zeroa` | `cls` | `last_q` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_ft_variants_20260301_223043/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp0_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp0_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp0_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp0_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp1_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp1_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp1_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta0_tp1_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp0_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp0_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp0_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp0_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp1_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp1_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp1_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d0_ta1_tp1_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp0_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp0_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp0_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp0_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp1_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp1_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp1_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta0_tp1_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp0_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp0_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp0_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp0_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp1_r0` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp1_r0/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp1_r1` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp1_r1/obj_only.out` |
| `obj_only` | `patchnepa_stage2_sanity_20260301_162213_ft_layout_interleave` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity_20260301_162213_ft_layout_interleave/obj_only.out` |
| `pb_t50_rs` | `patchnepaFT_from_ray_interleave_20260301_014535` | `-` | `-` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_from_ray_interleave_20260301_014535/pb_t50_rs.out` |
| `pb_t50_rs` | `patchnepaFT_splitx2_dualmask_qonly_20260301_040743` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_20260301_040743/pb_t50_rs.out` |
| `pb_t50_rs` | `patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745` | `q_only` | `mean_q` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745/pb_t50_rs.out` |
| `pb_t50_rs` | `patchnepaFT_splitx2_encdec1_qonly_20260301_040738` | `q_only` | `cls_max` | `-` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_qonly_20260301_040738/pb_t50_rs.out` |
| `pb_t50_rs` | `patchnepa_ft_variants_20260301_223043` | `qa_zeroa` | `cls` | `last_q` | `no_test_acc` | `logs/sanity/patchnepa_ft/patchnepa_ft_variants_20260301_223043/pb_t50_rs.out` |

## 12. Exhaustive Sources

- `nepa3d/docs/patch_nepa/patchnepa_ft_exhaustive_audit.tsv` (all direct-ft `.out`, completed + incomplete)
- `nepa3d/docs/patch_nepa/patchcls_exhaustive_audit.tsv` (all patchcls `.out`, completed + incomplete)

## 13. Latest Completion Snapshot (2026-03-01)

### 13.1 Newly finished jobs in this update

| job | purpose | status | key output |
|---|---|---|---|
| `100643` | bind + aug + dualmask pretrain | finished invalid (`Exit_status=97`) | launcher marker missing |
| `100700` | point-only + EMA E100 pretrain | finished valid (`Exit_status=0`) | ckpt produced |
| `100704`-`100715` | sanity6 short-screen pretrain+FT | all finished valid (`Exit_status=0`) | FT `obj_only` results below |

### 13.2 sanity6 (E12->E120) FT results (`obj_only`)

| condition | pretrain job | FT job | TEST acc |
|---|---|---|---:|
| random, aug off | `100704` | `100705` | `0.6833` |
| random, aug on | `100706` | `100707` | `0.6196` |
| fps, aug off | `100708` | `100709` | `0.6454` |
| fps, aug on | `100710` | `100711` | `0.5318` |
| rfps_cached, aug off | `100712` | `100713` | `0.6368` |
| rfps_cached, aug on | `100714` | `100715` | `0.6317` |

Current best in this short-screen branch: `random + aug off (0.6833)`.

## 14. Cross-Line Scale Comparison (2D-NEPA / Point-MAE / PatchNEPA)

Purpose:
- make FT/pretrain scale differences explicit before interpreting accuracy deltas.

### 14.1 Pretrain recipe scale

| line | script / config | epochs | effective batch | LR | scheduler / warmup | WD | EMA | key defaults |
|---|---|---:|---:|---:|---|---:|---|---|
| 2D-NEPA-B | `scripts/pretrain/nepa_b.sh` | 1600 | 4096 | `3e-4 * 4096/256 = 4.8e-3` | cosine, warmup_ratio=0.025 | 0.05 | (run script has EMA support in `run_nepa.py`) | image pipeline |
| 2D-NEPA-L | `scripts/pretrain/nepa_l.sh` | 1600 | 4096 | `4.8e-3` | cosine, warmup_ratio=0.025 | 0.05 | (same as above) | image pipeline |
| Point-MAE | `Point-MAE/cfgs/pretrain.yaml` | 300 | 128 | 1.0e-3 | CosLR, warmup 10 epochs | 0.05 | no | `mask_ratio=0.6`, `num_group=64`, `group_size=32` |
| PatchNEPA mainline | `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh` | 100 (default) | 128 (policy fixed) | 3.0e-4 | cosine, warmup_ratio=0.025 | 0.05 | optional (`USE_EMA`) | ray on (`N_RAY=1024`), split+sep, rfps_cached |

### 14.2 Fine-tune recipe scale

| line | script / config | epochs | effective batch | LR | scheduler / warmup | LLRD | freeze embed | notes |
|---|---|---:|---:|---:|---|---|---|---|
| 2D-NEPA-B | `scripts/finetune/nepa_b_sft.sh` | 100 | 1024 | `1.5e-3 * 1024/256 = 6.0e-3` | cosine + `llrd_cosine_warmup`, warmup_ratio=0.20 | `0.65` | yes (`freeze_embed=True`) | EMA=0.9999 |
| 2D-NEPA-L | `scripts/finetune/nepa_l_sft.sh` | 100 | 1024 | `1.0e-3 * 1024/256 = 4.0e-3` | cosine + `llrd_cosine_warmup`, warmup_ratio=0.30 | `0.75` | yes | EMA=0.9999 |
| Point-MAE (ScanObjectNN) | `Point-MAE/cfgs/finetune_scan_*.yaml` | 300 | 32 | 5.0e-4 | CosLR, warmup 10 epochs | no | no explicit freeze | train transform=`PointcloudScaleAndTranslate` |
| PatchNEPA direct FT | `scripts/finetune/patchnepa_scanobjectnn_finetune.sh` + `scripts/finetune/patchcls_scanobjectnn_scratch.sh` | 300 | 64 (global mode default) | 5.0e-4 | cosine, warmup 10 epochs (`llrd_cosine_warmup` default in direct path) | `0.35 -> 1.0` | yes (`patchnepa_freeze_patch_embed=1`) | `pooling=cls`, `cls_token=last_q`, `head=linear` defaults |

### 14.3 LLRD interpretation (important)

Conclusion:
- `LLRD=0.65/0.75` itself is not a known failure pattern.
- 2D-NEPA uses these values in working recipes; this alone is not evidence of regression.

Current higher-priority issue in some PatchNEPA FT runs:
- observed effective LR collapse to `1e-11 ~ 1e-10` in logs (for example `patchnepa_ft_variants_20260301_223043`), while normal runs are `~1e-4` scale.
- this mismatch is large enough to dominate performance differences, independent of choosing `0.35->1.0` vs `0.65/0.75`.

Operational rule:
- treat LR-scale-broken runs as non-comparable until scheduler/group scaling is fixed.

### 14.4 Sources

- `scripts/pretrain/nepa_b.sh`
- `scripts/pretrain/nepa_l.sh`
- `scripts/finetune/nepa_b_sft.sh`
- `scripts/finetune/nepa_l_sft.sh`
- `Point-MAE/cfgs/pretrain.yaml`
- `Point-MAE/cfgs/finetune_scan_objonly.yaml`
- `Point-MAE/cfgs/finetune_scan_objbg.yaml`
- `Point-MAE/cfgs/finetune_scan_hardest.yaml`
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
- `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
- `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
- `logs/sanity/patchnepa_ft/patchnepa_ft_variants_20260301_223043/obj_bg.out`
- `logs/sanity/patchnepa_ft/obj_bg.out`

## 15. FT Augmentation Parity Note (Point-MAE)

- Current PatchNEPA FT `aug_preset=pointmae` is **Point-MAE-aligned but not strict-identical**.
- Difference:
  - PatchNEPA: isotropic scalar scale (`s`) in `_apply_point_aug`.
  - Point-MAE: anisotropic per-axis scale (`sx, sy, sz`) in `PointcloudScaleAndTranslate`.
- Interpretation rule:
  - Until this is matched, call the setting "Point-MAE-like" (not strict parity).
- Status (2026-03-02): implemented in PatchNEPA FT augmentation path (per-axis scale enabled for `aug_preset=pointmae`).
- Backward-compatibility switch:
  - `--pointmae_exact_aug 0` restores legacy isotropic-scalar scale behavior for ablation.

## 16. Insight: Why `pb_t50_rs` gap is the key warning

- In matched direct-FT baseline (`splitx2_dualmask_baseline_20260301_040740`):
  - `obj_bg`: gap `0.0620`
  - `obj_only`: gap `0.0731`
  - `pb_t50_rs`: gap `0.1626` (largest)
- This pattern recurs across completed runs: largest val->test gaps are repeatedly `pb_t50_rs`.
- Interpretation:
  - likely not "lack of raw information" only;
  - more consistent with shortcut-fit / split-specific fit that does not transfer to test on the hardest variant.
- Reporting rule:
  - prioritize `pb_t50_rs` when judging objective/aug changes,
  - and avoid using best-val alone for acceptance of a recipe.

## 17. Missing-Ray A/B: What changed and why best is still higher

### 17.1 Strict A/B result (same FT recipe)

- independent (`MISSING_RAY=0`): `obj_only=0.7762`
- bind (`MISSING_RAY=86`): `obj_only=0.7367`
- effect: `+0.0395` for independent

Interpretation:
- under fixed FT recipe, lowering missing-ray clearly helps.

### 17.2 Why this does not automatically beat current-best `0.8193`

The current-best line is from a different recipe family.
Main deltas vs strict A/B line:

- FT readout:
  - best-line: `cls_max + pointmae_mlp`
  - strict A/B line: `cls + linear` (plus patch_embed freeze)
- pretrain regime:
  - best-line: `rfps_cached`, aug off
  - strict A/B line: `fps`, aug on (`scale=[0.667,1.5], translate=0.2`)
- objective controls:
  - strict A/B branch includes `skip_k=[1]` in pretrain header.

Practical conclusion:
- missing-ray is a positive factor, but not the dominant limiter at current gap scale.
- best comparison claims should be made only inside the same recipe family.

## 18. New result append (`100765`~`100774`, 2026-03-02)

### 18.1 Safe indpatch chain FT updates

Completed (`TEST acc` confirmed):

| source pretrain | variant | test_acc | log |
|---|---|---:|---|
| dualmask indpatch-safe (`100765`) | `obj_bg` (`100767`) | `0.8021` | `logs/sanity/patchnepa_ft/patchnepaFT_from_dualmask_indpatch_safe_20260302_031834/obj_bg.out` |
| dualmask indpatch-safe (`100765`) | `obj_only` (`100768`) | `0.7900` | `logs/sanity/patchnepa_ft/patchnepaFT_from_dualmask_indpatch_safe_20260302_031834/obj_only.out` |
| encdec1 indpatch-safe (`100766`) | `obj_bg` (`100770`) | `0.7969` | `logs/sanity/patchnepa_ft/patchnepaFT_from_encdec1_indpatch_safe_20260302_031834/obj_bg.out` |
| encdec1 indpatch-safe (`100766`) | `obj_only` (`100771`) | `0.8038` | `logs/sanity/patchnepa_ft/patchnepaFT_from_encdec1_indpatch_safe_20260302_031834/obj_only.out` |

Still running:
- `100769` (`dualmask`, `pb_t50_rs`), latest observed `ep 246/300`
- `100772` (`encdec1`, `pb_t50_rs`), latest observed `ep 104/300`

### 18.2 Strict FT A/B (`pretrain-init` vs `scratch-init`) update

| arm | job | best_val | test_acc | log |
|---|---|---:|---:|---|
| pretrain-init | `100773` | `0.8610` | `0.8021` | `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_pre_20260302_035338/obj_only.out` |
| scratch-init | `100774` | `0.8475` | `0.8055` | `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_scratch_20260302_035345/obj_only.out` |

Current read:
- with this fixed FT recipe, scratch-init is slightly higher on TEST (`+0.0034`), so this pair does not show a clear pretrain-init gain.

### 18.3 TSV index refresh

Updated files:
- `nepa3d/docs/patch_nepa/patchnepa_ft_completed_results.tsv`
- `nepa3d/docs/patch_nepa/patchnepa_ft_exhaustive_audit.tsv`

Current counts after refresh:
- completed rows: `73`
- exhaustive rows: `113`
- exhaustive status split: `completed=73`, `no_test_acc=40`

## 19. Q1/Q2 hypothesis jobs launched (pretrain+FT chain, 2026-03-02)

Goal split:

- Q1: does sampling/task diversity (`random`) beat fixed order (`rfps_cached`) under point-only pretrain?
- Q2: does modal-specific mixed pretrain (`mesh+udf`) transfer better than point-only pretrain under matched settings?

Launched matrix (all with dependent `obj_only` FT):

| arm | pretrain job | pretrain run_set | dependent FT job | FT run_set |
|---|---|---|---|---|
| Q1-random | `100778` | `patchnepa_q1_ptonly_random_e100_20260302_053501` | `100781` | `patchnepaFT_objonly_from_q1_ptonly_random_e100_20260302_053501` |
| Q1/Q2-control (rfps) | `100779` | `patchnepa_q1q2_ptonly_rfps_e100_20260302_053501` | `100782` | `patchnepaFT_objonly_from_q1q2_ptonly_rfps_e100_20260302_053501` |
| Q2-mix (mesh+udf, rfps) | `100780` | `patchnepa_q2_mix_meshudf_rfps_e100_20260302_053501` | `100783` | `patchnepaFT_objonly_from_q2_mix_meshudf_rfps_e100_20260302_053501` |

Comparison contract:

- Q1 uses `100778 vs 100779` (only `PT_SAMPLE_MODE` differs: `random` vs `rfps_cached`).
- Q2 uses `100780 vs 100779` (same `rfps_cached`; only mixed corpus differs).
- all three branches share:
  - pretrain: `E100`, global batch `128`, dual-mask on, EMA on, point-only token path (`USE_RAY_PATCH=0`, `N_RAY=0`)
  - FT: direct PatchNEPA, `qa_zeroa`, `cls_max + pointmae_mlp`, `cls_token=bos`, strict eval (`file + TTA10`), `CKPT_USE_EMA=1`.

Current status snapshot:

- pretrains `100778/100779/100780`: `R`
- dependent FT `100781/100782/100783`: `H` (`afterok`)

Tracking table:

- `logs/sanity/patchnepa_ft/patchnepa_q1q2_matrix_20260302_053501_jobs.tsv`
