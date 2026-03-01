# ScanObjectNN Variant Benchmark (Active Canonical)

Last updated: 2026-03-02

## 1. Scope

This file is the canonical benchmark summary for protocol-correct ScanObjectNN evaluation.

- Variant-split only: `obj_bg`, `obj_only`, `pb_t50_rs`
- Headline metric: `test_acc`
- SOTA-fair and NEPA-full are always reported separately

Detailed job history is split by track:

- Query-NEPA historical: `nepa3d/docs/query_nepa/runlog_202602.md`
- Patch-NEPA Stage-2: `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- Track index: `nepa3d/docs/patch_nepa/nepa_tracks_index.md`

### 1.1 Validity boundary (read first)

To avoid over-interpretation, results are classified as follows:

- `VALID (benchmark-eligible)`:
  - variant-split cache (`obj_bg` / `obj_only` / `pb_t50_rs`)
  - protocol wiring fixed (env propagation + cache guard active)
  - job finished successfully (`Exit_status=0`)
- `INTERNAL/ABLATION ONLY`:
  - mixed cache (`scanobjectnn_main_split_v2`) runs
  - historical checkpoints with known config mismatch for intended design
    (for example `qa_tokens` mismatch in old `runs/pretrain_abcd_1024_run*`)
- `INVALID`:
  - runtime-failed runs (`Exit_status!=0`)
  - known precheck/import failures (for example CPAC mesh precheck/import failures)

Scope note for historical issues:

- Python 3.9 type-hint import failure was a CPAC path issue, not a Scan classification issue.
- Env propagation issue affected specific old submissions and is fixed; canonical tables use re-submitted protocol-correct jobs.

## 2. Non-Negotiable Protocol

- `SCAN_CACHE_ROOT` must be one of:
  - `data/scanobjectnn_obj_bg_v3_nonorm`
  - `data/scanobjectnn_obj_only_v3_nonorm`
  - `data/scanobjectnn_pb_t50_rs_v3_nonorm`
- No mixed-cache headline table (`data/scanobjectnn_main_split_v2`) for fair comparison.
- `scanobjectnn_*_v2` (uni-scale) is legacy-only and excluded from new benchmark rows.
- `test_acc` is benchmark headline.
- `best_val` / `best_ep` are diagnostics only.

### 2.1 Why "same-split sanity" is required

Reason:

- It separates **data/split integrity** from **model/recipe quality**.
- If an external baseline reaches expected range on the same variant protocol
  (`pb_t50_rs` / `obj_bg` / `obj_only`), then low scores in NEPA runs are not
  explained by broken split construction alone.

Operational note:

- Point-MAE sanity in this repo uses the official Point-MAE ScanObjectNN
  layout (`data/ScanObjectNN/h5_files/main_split*`) rather than NEPA NPZ cache.
- This is split/domain-aligned sanity (not shared NPZ-loader sanity).

### 2.2 External sanity already executed (`Point-MAE`)

Status:

- historical as-launched jobs completed (`Exit_status=0`): `97974`, `97977`, `97978`
- corrected ckswap verification jobs completed (`Exit_status=0`): `100752`, `100753`
- important: this block is **pretrained-checkpoint inference sanity** (`--test --ckpts ...`),
  not `scratch` training baseline.

| variant | README target | historical as-launched (`scan_*.pth`) | corrected variant-aligned sanity | evidence |
|---|---:|---:|---:|---|
| `pb_t50_rs` | 85.18 | 84.5940 | 84.5940 | `logs/sanity/pointmae/pointmae_pb_t50_rs_sanity_20260226_183416.log` |
| `obj_bg` | 90.02 | 73.3219 | 90.1893 | old: `logs/sanity/pointmae/pointmae_h5_parity_v3nonorm_fix_20260227_060439/pm_obj_bg_official_pointmae_h5_parity_v3nonorm_fix_20260227_060439.out`, ckswap: `logs/sanity/pointmae/pointmae_ckswap_objbg_from_objonly_20260302_0135.out` |
| `obj_only` | 88.29 | 81.7556 | 87.9518 | old: `logs/sanity/pointmae/pointmae_h5_parity_v3nonorm_fix_20260227_060439/pm_obj_only_official_pointmae_h5_parity_v3nonorm_fix_20260227_060439.out`, ckswap: `logs/sanity/pointmae/pointmae_ckswap_objonly_from_objbg_20260302_0135.out` |

Checkpoint-metadata sanity from old logs (as-launched run):

- `scan_objbg.pth` loaded as `ckpts @ 166 epoch (acc=88.2960)`
- `scan_objonly.pth` loaded as `ckpts @ 250 epoch (acc=90.0172)`
- implication: `obj_bg` / `obj_only` ckpt label mapping was effectively swapped in the historical as-launched sanity.

Reference:

- `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md` section 9

Cache-input follow-up (`USE_NEPA_CACHE=1`) completed:

- purpose: run Point-MAE using data derived from NEPA NPZ cache
  (`scanobjectnn_*_v2`, legacy cache) via NPZ->H5 bridge.
- jobs (`Exit_status=0`):
  - `98326.qjcm` (`obj_bg`)
  - `98327.qjcm` (`pb_t50_rs`)
  - `98328.qjcm` (`obj_only`)
- bridge script:
  - `scripts/sanity/build_scanobjectnn_h5_from_nepa_cache.py`

| variant | Point-MAE official H5 sanity (as-launched ckpt mapping) | Point-MAE cache-derived H5 sanity (as-launched ckpt mapping) | delta |
|---|---:|---:|---:|
| `pb_t50_rs` | 84.5940 | 76.8910 | -7.7030 |
| `obj_bg` | 73.3219 | 61.4458 | -11.8761 |
| `obj_only` | 81.7556 | 81.7556 | +0.0000 |

Why this gap appears:

- This is expected under the current cache pipeline.
- NEPA ScanObjectNN cache stores normalized `pc_xyz` (center + unit-scale) at
  preprocess time, because cache construction is query/pool-oriented.
- reference:
  - `nepa3d/data/preprocess_scanobjectnn.py` (`normalize_points`)
  - `nepa3d/data/preprocess_scanobjectnn.py` (assignment `pc_xyz = normalize_points(pc)`)
- Therefore, `official H5` and `cache-derived H5` are not byte-identical point
  clouds even when split/label counts match; this is a preprocessing-domain
  difference, not a label/split mismatch.

Normalization contract note (important):

- This is primarily an implementation/design contract, not a theoretical limit.
- Non-normalized cache can be trained in principle, but current NEPA pipeline
  assumes normalized coordinate space in multiple stages.
- Practical impact:
  - `SOTA-fair` classification (`pc_xyz`, `ablate_point_dist=1`, no CPAC) can
    still run with non-normalized cache.
  - `NEPA-full` / CPAC paths are scale-sensitive under current code and are
    likely to degrade or become inconsistent unless query/eval coordinate
    assumptions are also refactored.
- main assumption points:
  - query pool generation uses fixed `[-1,1]` support
    (`nepa3d/data/preprocess_scanobjectnn.py`)
  - distance channel is scale-dependent and consumed by model in NEPA-full
    (`nepa3d/data/dataset.py`)
  - CPAC/mesh conversion assumes normalized cube convention
    (`nepa3d/analysis/completion_cpac_udf.py`)

Cache-derived source logs:

- `logs/sanity/pointmae/pointmae_pb_t50_rs_nepacache_20260227_030616.log`
- `logs/sanity/pointmae/pointmae_obj_bg_nepacache_20260227_030616.log`
- `logs/sanity/pointmae/pointmae_obj_only_nepacache_20260227_030616.log`

### 2.2.2 Scratch sanity baseline (`Transformer [54]` reference)

Status:

- completed run set:
  - `logs/sanity/patchcls/patchcls_scan3_scratch_v3nonorm_20260227_055706`
- jobs (`Exit_status=0`):
  - `98390.qjcm` (`pb_t50_rs`)
  - `98391.qjcm` (`obj_bg`)
  - `98392.qjcm` (`obj_only`)

Why it is needed:

- confirms whether current ScanObjectNN variant protocol can reach expected
  supervised-from-scratch range before attributing gaps to NEPA-specific design.
- separates "dataset/protocol viability" from "self-supervised transfer gain".

Reference targets (reported in Point-MAE table, supervised scratch `Transformer [54]`):

| variant | target test_acc |
|---|---:|
| `obj_bg` | 79.86 |
| `obj_only` | 80.55 |
| `pb_t50_rs` | 77.24 |

Note:

- This is the scratch baseline row in the table (not the `Point-MAE` SSL row).
- `Point-MAE` row itself is SSL-pretrained and higher (`90.02 / 88.29 / 85.18` in README).
- this block is `patchcls` scratch baseline (`patchify + transformer`, bidirectional, `is_causal=0`),
  not Point-MAE codepath.

Result (`patchcls_scan3_scratch_v3nonorm_20260227_055706`):

| variant | target (Transformer [54]) | observed `test_acc` | delta |
|---|---:|---:|---:|
| `obj_bg` | 79.86 | 71.08 | -8.78 |
| `obj_only` | 80.55 | 76.76 | -3.79 |
| `pb_t50_rs` | 77.24 | 68.77 | -8.47 |

Source logs:

- `logs/sanity/patchcls/patchcls_scan3_scratch_v3nonorm_20260227_055706/obj_bg.out`
- `logs/sanity/patchcls/patchcls_scan3_scratch_v3nonorm_20260227_055706/obj_only.out`
- `logs/sanity/patchcls/patchcls_scan3_scratch_v3nonorm_20260227_055706/pb_t50_rs.out`

Additional cross-check (`scan_h5` direct input):

- completed run set:
  - `logs/sanity/patchcls/patchcls_scan3_scratch_h5_20260227_061351`
- jobs (`Exit_status=0`):
  - `98407.qjcm` (`pb_t50_rs`)
  - `98408.qjcm` (`obj_bg`)
  - `98409.qjcm` (`obj_only`)

| variant | `v3_nonorm` (`npz`) | official H5 (`scan_h5`) | delta (h5 - npz) |
|---|---:|---:|---:|
| `obj_bg` | 0.7108 | 0.7074 | -0.0034 |
| `obj_only` | 0.7676 | 0.7797 | +0.0121 |
| `pb_t50_rs` | 0.6877 | 0.6731 | -0.0146 |

Obj-only random-sampling parity check (seed-fixed):

- run set:
  - `logs/sanity/patchcls/patchcls_objonly_parity_randomseed_20260227_063811`
- jobs (`Exit_status=0`):
  - `98413.qjcm` (`obj_only_v3_random`)
  - `98414.qjcm` (`obj_only_h5_random`)
- observed:
  - `obj_only_v3_random`: `test_acc=0.7728`
  - `obj_only_h5_random`: `test_acc=0.7745`
  - delta: `+0.0017` (h5 - v3)

Interpretation note:

- random-seed parity looks consistent between `npz(v3_nonorm)` and `scan_h5`.
- mainline policy is now fixed to `val_split_mode=file` only; historical `group_*`/`pointmae` rows below are reference-only.

Obj-only parity recheck (Point-MAE split unified):

- run set:
  - `logs/sanity/patchcls/patchcls_objonly_parity_pointmae_20260227_070409`
- jobs (`Exit_status=0`):
  - `98427.qjcm` (`obj_only_v3_pointmae`)
  - `98428.qjcm` (`obj_only_h5_pointmae`)
- observed:
  - `obj_only_v3_pointmae`: `test_acc=0.7900`
  - `obj_only_h5_pointmae`: `test_acc=0.7900`
  - delta: `+0.0000` (h5 - v3)

Interpretation update:

- this `val_split_mode=pointmae` block is historical/reference only.
- mainline strict comparisons must use `val_split_mode=file`.

Point-MAE scratch ckpt direct test extraction:

- script: `scripts/sanity/pointmae_scan_test_from_ckpt_qf.sh`
- clean run set:
  - `logs/sanity/pointmae_scratch_tests/pointmae_scan3_scratch_test_from_ckpt_fixlog_20260227_164410`
- jobs:
  - `98998.qjcm` (`obj_bg`, `Exit_status=0`)
  - `98999.qjcm` (`obj_only`, `Exit_status=0`)
- results (`--test`, ckpt-best from scratch run `pointmae_scan3_scratch_stdbs32_20260227_142324`):
  - `obj_bg`: `[TEST] acc = 86.7470`
  - `obj_only`: `[TEST] acc = 86.4028`
- unresolved:
  - `pb_t50_rs` test job `98997.qjcm` reached `job_state=F` but no valid output/test log was produced.
  - keep `pb_t50_rs` in this block as **not recorded** until rerun emits `[TEST] acc`.

Strict protocol rerun (`NO_TEST_AS_VAL=1`, `npoints=1024`) status:

- run set:
  - `logs/sanity/pointmae_scratch/pointmae_scan3_scratch_n1024_notestval_fix_20260227_174052`
- jobs:
  - `99105.qjcm` (`obj_bg`)
  - `99106.qjcm` (`obj_only`)
  - `99107.qjcm` (`pb_t50_rs`)
- current scheduler state:
  - `99105`: finished (`F`, `Exit_status=0`)
  - `99106`: finished (`F`, `Exit_status=0`)
  - `99107`: finished with walltime termination (`F`, `Exit_status=271`)

Test-from-ckpt (strict run checkpoints):

- `99171` (`obj_bg`): `[TEST] acc = 83.3046`
  - `logs/sanity/pointmae_scratch_tests/pm_obj_bg_pointmae_scan3_scratch_n1024_notestval_fix_20260227_174052_test.out`
- `99172` (`obj_only`): `[TEST] acc = 83.6489`
  - `logs/sanity/pointmae_scratch_tests/pm_obj_only_pointmae_scan3_scratch_n1024_notestval_fix_20260227_174052_test.out`
- `99173` (`pb_t50_rs`) dependency was unsatisfied (`afterok:99107`); finalized without execution (`Walltime=0`), so no test metric.

PatchCls PM-aligned scratch rerun (1024, all variants finalized):

- run set:
  - `logs/sanity/patchcls/patchcls_scan3_scratch_pmalign_20260227_202814`
- jobs (`Exit_status=0`):
  - `99181.qjcm` (`obj_bg`)
  - `99182.qjcm` (`obj_only`)
  - `99183.qjcm` (`pb_t50_rs`)

| variant | PatchCls PM-aligned `test_acc` |
|---|---:|
| `obj_bg` | 0.7831 |
| `obj_only` | 0.8176 |
| `pb_t50_rs` | 0.7609 |

Point-MAE strict retry5 (`NO_TEST_AS_VAL=1`, `npoints=1024`, BS64 extfix) finalized:

- scratch jobs: `99275` (`obj_bg`), `99277` (`obj_only`), `99279` (`pb_t50_rs`) all `Exit_status=0`
- test jobs: `99276`, `99278`, `99280` all `Exit_status=0`
- source logs:
  - `logs/sanity/pointmae_scratch_tests/obj_bg_pointmae_scan3_scratch_bs64_extfix_retry5_20260227_220742_test.out`
  - `logs/sanity/pointmae_scratch_tests/obj_only_pointmae_scan3_scratch_bs64_extfix_retry5_20260227_220742_test.out`
  - `logs/sanity/pointmae_scratch_tests/pb_t50_rs_pointmae_scan3_scratch_bs64_extfix_retry5_20260227_220742_test.out`

| variant | Point-MAE strict retry5 `test_acc` |
|---|---:|
| `obj_bg` | 0.8296 |
| `obj_only` | 0.8399 |
| `pb_t50_rs` | 0.8012 |

PatchCls split-only parity check (`obj_bg`, PM-aligned recipe fixed):

- run: `99285.qjcm` (`pc_objbg_splitf`), `Exit_status=0`
- log: `logs/sanity/patchcls/patchcls_obj_bg_pmalign_splitfile_20260227_223435/obj_bg.out`
- result: `TEST acc=0.7831`

Interpretation:

- `group_auto` -> `file` split change on patchcls (with other settings fixed) produced `delta=0.0000` on `obj_bg` in this run.
- `obj_bg` gap vs Point-MAE strict retry5 remains about `-4.65pt` (`0.7831 - 0.8296`).

### 2.2.3 Non-normalized cache + dynamic query-bbox follow-up (`v3_nonorm`)

Purpose:

- remove cache-time normalization for `pc_xyz` and avoid fixed `[-1,1]` query-space assumptions.
- confirm whether Point-MAE sanity recovers to official-H5 scores when using cache-derived H5 from non-normalized NPZ.
- this is **not** the same as 2.2.2 scratch baseline; this block is pretrained-checkpoint sanity on `v3_nonorm`.

Build/eval jobs:

- preprocess variants (`normalize_pc=0`, `query_bbox_mode=auto`): `98338` (`Exit_status=0`)
- Point-MAE sanity:
  - `98339` (`pb_t50_rs`, `Exit_status=0`)
  - `98352` (`obj_bg`, `Exit_status=0`)
  - `98355` (`obj_only`, `Exit_status=0`)

Result:

| variant | official H5 sanity (as-launched ckpt mapping) | cache-derived H5 sanity (`v3_nonorm`, as-launched ckpt mapping) | delta |
|---|---:|---:|---:|
| `pb_t50_rs` | 84.5940 | 84.5940 | +0.0000 |
| `obj_bg` | 73.3219 | 73.3219 | +0.0000 |
| `obj_only` | 81.7556 | 81.7556 | +0.0000 |

Interpretation:

- observed gap in the earlier cache-derived run (`v2`) was dominated by cache preprocessing domain shift (normalized vs non-normalized), not by split/label mismatch.
- with non-normalized cache + dynamic bbox sampling, Point-MAE sanity matches official H5 exactly under the same as-launched ckpt mapping for all three variants.

Implementation note:

- `scripts/sanity/pointmae_scan_sanity_qf.sh` now resolves relative cache paths to absolute paths to avoid broken symlink targets during `USE_NEPA_CACHE=1`.

### 2.3 NEPA NPZ cache sanity executed (integrity check)

Purpose:

- verify variant NPZ caches are not corrupted and contain required keys
  before interpreting classifier gaps.
- this is an integrity-only check (no model scoring).
- actual model evaluation using these caches is reported in Section 4+.

Command:

```bash
python scripts/sanity/check_scanobjectnn_variant_cache.py \
  --cache_roots data/scanobjectnn_obj_bg_v3_nonorm,data/scanobjectnn_obj_only_v3_nonorm,data/scanobjectnn_pb_t50_rs_v3_nonorm \
  --out_json logs/sanity/scanobjectnn_variant_cache_sanity_v3_nonorm_20260227.json
```

Summary:

| cache_root | split | n_files | n_classes | bad_npz | missing_key_files |
|---|---|---:|---:|---:|---:|
| `scanobjectnn_obj_bg_v3_nonorm` | train | 2309 | 15 | 0 | 0 |
| `scanobjectnn_obj_bg_v3_nonorm` | test | 581 | 15 | 0 | 0 |
| `scanobjectnn_obj_only_v3_nonorm` | train | 2309 | 15 | 0 | 0 |
| `scanobjectnn_obj_only_v3_nonorm` | test | 581 | 15 | 0 | 0 |
| `scanobjectnn_pb_t50_rs_v3_nonorm` | train | 11416 | 15 | 0 | 0 |
| `scanobjectnn_pb_t50_rs_v3_nonorm` | test | 2882 | 15 | 0 | 0 |

Artifacts (v3_nonorm canonical):

- `logs/sanity/scanobjectnn_variant_cache_sanity_v3_nonorm_20260227.json`
- `logs/sanity/scanobjectnn_variant_cache_sanity_v3_nonorm_20260227.out`
- checker script: `scripts/sanity/check_scanobjectnn_variant_cache.py`

### 2.4 Cache-loaded model eval evidence (`pb_t50_rs_v2`, legacy reference)

This confirms actual classification runs reading variant cache (not integrity-only):

- cache root used: `data/scanobjectnn_pb_t50_rs_v2`
- source logs:
  - `logs/eval/abcd_cls_cpac_scan_pb_nepafull_backfillcheck_meanq_lr1e4_20260226_194113/runA_classification_scan.log`
  - `logs/eval/abcd_cls_cpac_scan_pb_nepafull_backfillcheck_meanq_lr5e4_20260226_194113/runA_classification_scan.log`
  - `logs/eval/abcd_cls_cpac_scan_pb_nepafull_backfillcheck_meanall_lr1e4_20260226_195200/runA_classification_scan.log`
  - `logs/eval/abcd_cls_cpac_scan_pb_nepafull_backfillcheck_meanall_lr5e4_20260226_195200/runA_classification_scan.log`
- log evidence (all four): `num_train=10282 num_val=1134 num_test=2882` and final `best_val=... test_acc=...` lines present.
- policy note: this block is kept as historical troubleshooting evidence only; new benchmark runs use `*_v3_nonorm`.

### 2.5 Point-MAE vs patchcls pipeline diff audit (why gap remains)

Purpose:

- Confirm whether current gap is caused by patch-construction method itself.
- Decide whether to switch from `FPS+kNN` patches to serialization chunks at this stage.

Conclusion (current stage):

- **Do not switch patch construction to serialization yet.**
- Both paths already use `FPS+kNN`-style local grouping as the core patch constructor, and the larger gap is dominated by recipe/backbone/head mismatches.

Confirmed major diffs (code-level):

| item | Point-MAE path | patchcls path | impact |
|---|---|---|---|
| patch token count | `num_group=128` (`Point-MAE/cfgs/finetune_scan_objonly.yaml`) | default `num_groups=64` (`nepa3d/train/finetune_patch_cls.py`) | fewer tokens in patchcls by default |
| points per sample | config default `npoints=2048` (recent strict run overrides to `1024`) | default `n_point=1024` | effective input density differs unless explicitly aligned |
| train sampling | `_fps_then_sample(points, npoints)` with `point_all` pre-pool (`Point-MAE/tools/runner_finetune.py`) | direct random/fps subset from dataset (`nepa3d/data/cls_patch_dataset.py`) | crop distribution differs |
| augmentation | `PointcloudScaleAndTranslate` (`Point-MAE/tools/runner_finetune.py`) | `_apply_point_aug` preset (`scale/shift/jitter/rot`) (`nepa3d/data/cls_patch_dataset.py`) | train-time geometry noise differs |
| backbone block | Point-MAE `TransformerEncoder` (`Point-MAE/models/Point_MAE.py`) | `CausalTransformer(backbone_impl='nepa2d')` (`nepa3d/models/patch_classifier.py`) | attention block implementation differs |
| positional encoding | center-MLP positional embedding (`Point-MAE/models/Point_MAE.py`) | learned absolute `pos_emb` table (`nepa3d/models/patch_classifier.py`) | token position signal differs |
| classifier head | concat(`[CLS]`, token-max) + MLP head (`Point-MAE/models/Point_MAE.py`) | single linear head after pooled feature (`nepa3d/models/patch_classifier.py`) | head capacity differs |
| optimization | `lr=5e-4`, `grad_norm_clip=10` (`Point-MAE/cfgs/finetune_scan_objonly.yaml`) | commonly used parity run used `lr=1e-3`, `grad_clip=1` (`runs/patchcls/.../args.json`) | update scale/clip regime differs |
| validation protocol | legacy test-as-val (`subset=test`) or strict split override (`NO_TEST_AS_VAL=1`) | mainline fixed to `file` (`nepa3d/train/finetune_patch_cls.py`) | comparison target is fixed under current policy |

Practical implication:

- The present gap is not evidence that `FPS+kNN` patch construction is insufficient.
- First close recipe parity (sampling/aug/head/optimizer/selection protocol), then re-measure.
- If serialization is explored, keep it as **patch ordering** (inter-patch sequence) first; avoid replacing local patch membership before parity is established.

## 3. Checkpoint Families in Use

- `A_fps`: `runs/pretrain_abcd_1024_runA/last.pt`
- `B_fps`: `runs/pretrain_abcd_1024_runB/last.pt`
- `A_rfps_aug`: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt`
- `B_rfps_aug`: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt`

## 4. Latest 1024 Variant-Split Snapshot (minimum 12 jobs)

Source: `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md` section 18.1

| pretrain | run | variant | test_acc |
|---|---|---|---:|
| `fps` | A | `obj_bg` | 0.5143 |
| `fps` | B | `obj_bg` | 0.5247 |
| `fps` | A | `obj_only` | 0.5742 |
| `fps` | B | `obj_only` | 0.5742 |
| `fps` | A | `pb_t50_rs` | 0.5264 |
| `fps` | B | `pb_t50_rs` | 0.5143 |
| `rfps` | A | `obj_bg` | 0.5013 |
| `rfps` | B | `obj_bg` | 0.4844 |
| `rfps` | A | `obj_only` | 0.5195 |
| `rfps` | B | `obj_only` | 0.5339 |
| `rfps` | A | `pb_t50_rs` | 0.5098 |
| `rfps` | B | `pb_t50_rs` | 0.4990 |

### 4.1 Newly Finalized Extension Rows (NEPA-full / `pb_t50_rs`)

Source logs:

- `logs/eval/abcd_cls_cpac_variant_ext_20260226_1346_nepafull_a_fps_lr1e4_pb_t50_rs_base/runA_classification_scan.log`
- `logs/eval/abcd_cls_cpac_variant_ext_20260226_1346_nepafull_a_fps_lr5e4_pb_t50_rs_base/runA_classification_scan.log`

| job_id | pretrain | protocol | variant | lr_cls | test_acc |
|---|---|---|---|---:|---:|
| `97653` | `A_fps` | `nepafull` | `pb_t50_rs` | `1e-4` | 0.3190 |
| `97656` | `A_fps` | `nepafull` | `pb_t50_rs` | `5e-4` | 0.3171 |

### 4.2 Post-backfill NEPA-full check (`98018`/`98019`/`98021`/`98022`, legacy cache)

Scope:

- cache: `data/scanobjectnn_pb_t50_rs_v2` (after `pt_fps_order` backfill, legacy)
- checkpoint: `runs/pretrain_abcd_1024_runA/last.pt`
- protocol: `NEPA-full`
- all jobs completed with `Exit_status=0`

| job_id | pooling | lr_cls | test_acc |
|---|---|---:|---:|
| `98018` | `mean_q` | `5e-4` | 0.2367 |
| `98019` | `mean_q` | `1e-4` | 0.2591 |
| `98021` | `mean_all` | `1e-4` | 0.2793 |
| `98022` | `mean_all` | `5e-4` | 0.2292 |

## 5. Latest 256 Query-Rethink Classification Snapshot (18 jobs)

Source run set:

- `a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`
- source table: `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md` section 15

ScanObjectNN (`test_acc`) best variant in this set:

- `b04_split_xanchor_morton_typepos`
  - `sotafair=0.5059`
  - `nepafull=0.3275`

Log-reading caution:

- In this run set, each `*.out` contains both ScanObjectNN and ModelNet40 stages.
- Therefore `test_acc` lines around `0.84~0.86` in `*.out` belong to ModelNet40, not ScanObjectNN.
- For ScanObjectNN benchmark values, read only `*_classification_scan.log`.

## 6. Reporting Checklist

For each new table row, include:

1. checkpoint path
2. pretrain family
3. variant cache root
4. protocol (`sotafair` or `nepafull`)
5. job ID
6. `test_acc`

## 7. PatchNEPA ptonly -> PatchCls finetune snapshot (2026-02-28)

Run set:

- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007`
- jobs:
  - `100002` (`obj_bg_nepa2d`) done
  - `100003` (`obj_only_nepa2d`) done
  - `100004` (`pb_t50_rs_nepa2d`) done

Recipe note for this run set:

- `val_split_mode=file`
- eval uses voting:
  - `aug_eval=True`
  - `mc_test=10`

Current results:

| variant | PatchNEPA-ptonly -> PatchCls (`file + TTA`) |
|---|---:|
| `obj_bg` | 0.8003 |
| `obj_only` | 0.7986 |
| `pb_t50_rs` | 0.7047 |

File-split scratch references (strict line):

| variant | scratch ref (`file`) | delta |
|---|---:|---:|
| `obj_bg` | 0.7831 | +0.0172 |
| `obj_only` | 0.7849 | +0.0137 |
| `pb_t50_rs` | pending | pending |

## 8. New FT set from DDP16 RFPS-bank pretrain ckpt (partial complete)

- source ckpt:
  - `runs/patchnepa_pointonly/patchnepa_ptonly_ddp16_rfpsbank_20260228_223103/ckpt_latest.pt`
- run set:
  - `logs/sanity/patchcls/patchcls_ft_from_patchnepa_pt16_rfpsbank_20260228_231732`
- jobs:
  - `100028` (`obj_bg_nepa2d`) done (`Exit_status=0`)
  - `100029` (`obj_only_nepa2d`) done (`Exit_status=0`)
  - `100030` (`pb_t50_rs_nepa2d`) log has `TEST acc=0.7078` (queue state was `R` at capture time)
- fixed recipe:
  - `val_split_mode=file`, `aug_eval=1`, `mc_test=10`, `backbone_mode=nepa2d`

Current results:

| variant | PatchNEPA-pt16-rfpsbank -> PatchCls (`file + TTA`) |
|---|---:|
| `obj_bg` | 0.7401 |
| `obj_only` | 0.7849 |
| `pb_t50_rs` | 0.7078 (provisional) |

vs scratch reference (`file + TTA`, nepa2d):

| variant | scratch (`file + TTA`) | FT (`file + TTA`) | delta |
|---|---:|---:|---:|
| `obj_bg` | 0.7625 | 0.7401 | -0.0224 |
| `obj_only` | 0.7849 | 0.7849 | +0.0000 |
| `pb_t50_rs` | pending | 0.7078 | pending |

## 9. Scratch recheck (strict `file` + TTA10) partial snapshot

Run sets:

- `logs/sanity/patchcls/patchcls_scratch_file_tta10_recheck_20260228_232553` (backbone=`nepa2d`)
- `logs/sanity/patchcls/patchcls_scratch_file_tta10_vanilla_recheck_20260228_233118` (backbone=`vanilla`)

Jobs:

- nepa2d:
  - `100031` (`obj_bg`) done (`Exit_status=0`)
  - `100032` (`obj_only`) done (`Exit_status=0`)
  - `100033` (`pb_t50_rs`) running
- vanilla:
  - `100035` (`obj_bg`) done (`Exit_status=0`)
  - `100036` (`obj_only`) done (`Exit_status=0`)
  - `100037` (`pb_t50_rs`) running

Current results (completed rows only):

| variant | nepa2d scratch (`file + TTA10`) | vanilla scratch (`file + TTA10`) |
|---|---:|---:|
| `obj_bg` | 0.7625 | 0.7694 |
| `obj_only` | 0.7849 | 0.7298 |
| `pb_t50_rs` | pending | pending |

## 10. LR-scheduler controlled comparison job (pretrain)

To isolate scheduler effect only, one controlled pretrain was launched from the same
point-only encdec split recipe as `100027`, with scheduler change only:

- job: `100042.qjcm` (`patchnepa_ptE16c`) running
- run set:
  - `logs/patch_nepa_pretrain/patchnepa_pointonly_ddp16_encdec_split_cosine_20260301_000000`
- key setting difference:
  - `lr_scheduler=cosine`, `warmup_ratio=0.025` (`warmup_epochs=2.5`)
- fixed settings kept:
  - `N_POINT=1024`, `N_RAY=0`, `patch_embed=fps_knn`, `group_size=32`, `num_groups=64`
  - `qa_tokens=1`, `qa_layout=split_sep`, `encdec_arch=1`
  - `batch_per_proc=8`, `num_processes=16` (`global_batch=128`)

Update:

- `100042` is invalid for comparison due to scheduler stepping bug (LR oscillation).
- replacement run:
  - `100045.qjcm`
  - `patchnepa_pointonly_ddp16_encdec_split_cosine_fixsched_20260301_001200`
  - early epoch 1-5 trend (rank0):
    - `epoch1`: `lr=2.40e-04`, `loss_avg=0.0135`
    - `epoch2`: `lr=3.00e-04`, `loss_avg=0.0844`
    - `epoch3`: `lr=3.00e-04`, `loss_avg=0.1625`
    - `epoch4`: `lr=3.00e-04`, `loss_avg=0.2124`
    - `epoch5`: `lr=3.00e-04`, `loss_avg=0.2176`

## 11. Important comparability note (`ptonly_onepass` vs `pt16_rfpsbank`)

The two Stage-2 FT blocks above are not a strict same-pretrain-recipe A/B.

Pretrain side differs:

- `ptonly_onepass` lineage (`99710`):
  - `PT_XYZ_KEY=pc_xyz`
  - `ABLATE_POINT_DIST=1`
  - `8 GPU x batch16` (`global_batch=128`)
- `pt16_rfpsbank` lineage (`100010`):
  - `PT_XYZ_KEY=pt_xyz_pool`
  - `ABLATE_POINT_DIST=0`
  - `16 GPU x batch8` (`global_batch=128`)

Finetune side check:

- `args.json` comparison for FT runs shows recipe parity; practical difference is ckpt path.

Therefore:

- the observed FT delta is interpreted as **pretrain-ckpt content/config difference**,
  not FT recipe drift.

Source logs:

- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/pb_t50_rs_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_pt16_rfpsbank_20260228_231732/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_pt16_rfpsbank_20260228_231732/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchcls_scratch_file_tta10_recheck_20260228_232553/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchcls_scratch_file_tta10_recheck_20260228_232553/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchcls_scratch_file_tta10_vanilla_recheck_20260228_233118/obj_bg_vanilla.out`
- `logs/sanity/patchcls/patchcls_scratch_file_tta10_vanilla_recheck_20260228_233118/obj_only_vanilla.out`
- `logs/sanity/patchcls/patchcls_obj_bg_pmalign_splitfile_20260227_223435/obj_bg.out`
- `logs/sanity/patchcls/patchcls_objonly_factor4_20260228_070858/pcf_b_file.out`

## 12. Ray-enabled FT (`query-only ray`) result snapshot (2026-03-01)

Run set:

- `logs/sanity/patchcls/patchnepa_rayqa_ft_from100011_queryonly_20260301_002508`
- recipe summary:
  - `USE_RAY_PATCH=1`
  - query-only ray path (`ray_o/ray_d` only; no `ray_t/ray_hit` in ray encoder)
  - `val_split_mode=file`, `aug_eval=1`, `mc_test=10`, backbone=`nepa2d`

Job status/result snapshot:

- `100073` (`obj_bg`): done, `TEST acc=0.7281`
- `100074` (`obj_only`): done, `TEST acc=0.7762`
- `100075` (`pb_t50_rs`): running (`ep 128/300` at snapshot), `TEST acc` pending

Result table (completed rows only):

| variant | PatchNEPA-RayQA -> PatchCls (`file + TTA10`) |
|---|---:|
| `obj_bg` | 0.7281 |
| `obj_only` | 0.7762 |
| `pb_t50_rs` | pending |
