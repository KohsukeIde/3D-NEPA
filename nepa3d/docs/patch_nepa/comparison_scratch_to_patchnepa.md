# Scratch -> PatchNEPA Comparison Matrix

Last updated: 2026-03-01

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
| external reference | Point-MAE SSL target row | 0.9002 | 0.8829 | 0.8510 | 0.8780 | benchmark target (`benchmark_scanobjectnn_variant.md`) |
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

This branch is for causal diagnosis of patch sampling / ray grouping; it is **not** yet a finished leaderboard row because the independent branch (`100699`) is still active and bind+aug+dualmask (`100643`) ended invalid.

| branch | job | setting delta | status | usable for final comparison |
|---|---|---|---|---|
| fps baseline (no aug) | `100563` | `pt_sample_mode=fps`, bind ray | completed | yes (pretrain-side) |
| random baseline (no aug) | `100564` | `pt_sample_mode=random`, bind ray | completed | yes (pretrain-side) |
| fps + aug (dualmask off) | `100642` | PM-like aug only | terminated by user (`Exit_status=265`) | no |
| fps + aug + dualmask | `100643` | PM-like aug + dualmask | finished (`Exit_status=97`) | no (launcher marker missing / invalid) |
| independent ray patch + aug + dualmask | `100699` | `ray_assign_mode=independent_fps_knn`, `ray_num_groups=32` | running | pending |

Step-0 token sanity (sample-0):

- bind baseline (`100643` log): `Q_RAY=21`, `A_RAY=21`, `MISSING_RAY=86`
- independent patch (`100699` log): `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`
- reduction: `MISSING_RAY 86 -> 0` (`-100%`)

Note on the augmentation question:

- there is no evidence that augmentation itself is broken.
- the only ended aug-only run (`100642`) was user-terminated; it is excluded from validity.

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

- completed rows: `59`
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
