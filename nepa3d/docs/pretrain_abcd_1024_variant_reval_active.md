# 1024 Variant-Split Re-evaluation (Active)

Last updated: 2026-02-26

## 1. Scope

This document is the active plan for protocol-correct ScanObjectNN re-evaluation:

- variant-split benchmark only (`obj_bg`, `obj_only`, `pb_t50_rs`)
- `test_acc` headline reporting
- re-check of previously claimed comparison axes

Historical run logs remain in:

- `nepa3d/docs/pretrain_abcd_1024_multinode_active.md` (legacy ledger)

## 2. Protocol Baseline (non-negotiable)

Variant cache roots:

- `data/scanobjectnn_obj_bg_v2`
- `data/scanobjectnn_obj_only_v2`
- `data/scanobjectnn_pb_t50_rs_v2`

Each variant must be built from one train h5 and one test h5 (per protocol definition).

## 3. Job Count (fine-tune/eval only)

Counting rule for current submit path:

- jobs = `n_variants * n_runs * n_ablations`
- with defaults: `n_variants=3`, `n_runs=4` (`A/B/C/D`)

### 3.1 Operational minimum (start here)

- checkpoint families: `fps`, `rfps` (existing 1024 checkpoints)
- runs: `A/B` only (exclude weaker `C/D` in first pass)
- ablations: `base` only
- variants: `obj_bg,obj_only,pb_t50_rs`
- total eval jobs:
  - per family: `3 * 2 * 1 = 6`
  - two families: `6 * 2 = 12`

This is the current recommended "go" set to reduce queue load and quickly verify
protocol-correct trends.

### 3.2 Expanded fair rerun (second wave)

- checkpoint families: `fps`, `rfps`
- runs: `A/B/C/D`
- ablations: `base,llrd,dp,llrd_dp` (`4`)
- total eval jobs:
  - per family: `3 * 4 * 4 = 48`
  - two families: `48 * 2 = 96`

### 3.3 Full historical-comparison backfill (eval-only estimate)

If all previously compared fine-tune-side groups are repeated per family:

- group A: `base,llrd,dp,llrd_dp` (`4`)
- group B: pooling+LS (`4`)
- group C: reg-ablation (`4`)
- group D: point-order x augmentation (`4`)
- total ablation configs: `16`

Eval-job total:

- one family: `3 * 4 * 16 = 192`
- two families (`fps`,`rfps`): `192 * 2 = 384`
- optional dual-mask AB-only add-on (base): `+12`
- full estimate with add-on: `396`

## 4. Is this eval-only?

Mostly yes, with one important exception:

- Existing checkpoints are sufficient for most reruns (`fps`/`rfps` families).
- `mesh+udf only` corpus comparison needs dedicated checkpoints with
  `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf.yaml`.
- therefore that axis is not eval-only unless matching checkpoints already exist.

## 5. Recommended Execution Order

1. Build variant caches and verify `_meta` (`h5_count=1` per split).
2. Run operational minimum (`12` eval jobs).
3. Freeze a protocol-correct baseline table (`test_acc` only).
4. Run expanded fair rerun (`96`) only if needed.
5. Expand to full backfill (`384`/`396`) only for paper/supporting appendix.
6. Run `mesh+udf only` comparison after preparing dedicated pretrain checkpoints.

Current pruning note:

- query-rethink 18-way variants (including `view_raster`) are not in the first-wave minimum.
- They are promoted only after `A/B` variant-split baseline is stable.

## 6. Reporting Rules

- Publish per-variant tables (`obj_bg`, `obj_only`, `pb_t50_rs`).
- Keep SOTA-fair and NEPA-full separated.
- Keep `best_val`/`best_ep` as diagnostics only; do not headline them.
- For every table row, include:
  - checkpoint path and job id
  - pretrain `pt_sample_mode_train` and `mix_config`
  - eval `SCAN_CACHE_ROOT` and sampling mode

## 7. Execution Record (2026-02-26)

### 7.1 Script updates for reduced first wave

- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - added `RUN_IDS` filter (for example `A,B`) to avoid mandatory A/B/C/D submission.
- `scripts/eval/submit_sotafair_variants_llrd_droppath_ablation_qf.sh`
  - when `QSUB_DEPEND` is set, missing variant cache is warning-only at submit time.
- added preprocess submit helper:
  - `scripts/preprocess/submit_scanobjectnn_protocol_variants_qf.sh`
- added minimal eval submit helper:
  - `scripts/eval/submit_scan_variant_minimal_ab_fps_rfps_qf.sh`

### 7.2 Submitted jobs (operational minimum = 12 eval jobs)

First attempt (failed quickly):

- variant preprocess `97617.qjcm` failed (`Exit_status=1`)
- root cause:
  - `preprocess_scanobjectnn_protocol_variants.sh` resolved root to `/var/spool/pbs`
  - python path became `/var/spool/pbs/.venv/bin/python`
- fix:
  - script now prioritizes `WORKDIR` / `PBS_O_WORKDIR` for root resolution.

Retry1 (active):

- variant preprocess: `97630.qjcm` `preprocess_scanobjectnn_variants_v2_retry1`

Minimal eval submit command profile:

- families: `fps,rfps`
- variants: `obj_bg,obj_only,pb_t50_rs`
- runs: `A,B`
- ablation: `base`
- toggles: `RUN_MODELNET=0`, `RUN_CPAC=0`
- dependency: `afterok:97630.qjcm`

Submitted eval jobs:

- fps family:
  - `97631.qjcm` `obj_bg` runA
  - `97632.qjcm` `obj_bg` runB
  - `97633.qjcm` `obj_only` runA
  - `97634.qjcm` `obj_only` runB
  - `97635.qjcm` `pb_t50_rs` runA
  - `97636.qjcm` `pb_t50_rs` runB
- rfps family:
  - `97637.qjcm` `obj_bg` runA
  - `97638.qjcm` `obj_bg` runB
  - `97639.qjcm` `obj_only` runA
  - `97640.qjcm` `obj_only` runB
  - `97641.qjcm` `pb_t50_rs` runA
  - `97642.qjcm` `pb_t50_rs` runB

Notes:

- current state:
  - `97630` is running (`R`)
  - `97631`..`97642` are held (`H`) on dependency.
- query-rethink 18-way variants (including `view_raster`) are intentionally excluded
  from this first wave.

### 7.3 Coverage status vs requested 26-job matrix (updated)

First-wave + extension submissions:

- first-wave minimum: `97631`..`97642` (`12` jobs)
- extension add-on: `97645`..`97664` (`20` jobs)
- total submitted behind preprocess dependency: `32` eval jobs

Extension submission breakdown (`scripts/eval/submit_scan_variant_extension_qf.sh`):

- `97645`..`97650`: SOTA-fair `A_fps/B_fps`, `LR=5e-4`, 3 variants (`6` jobs)
- `97651`..`97656`: NEPA-full `A_fps`, `LR in {1e-4,5e-4}`, 3 variants (`6` jobs)
- `97657`..`97660`: SOTA-fair `A_rfps_aug/B_rfps_aug`, `pb_t50_rs`, `LR in {1e-4,5e-4}` (`4` jobs)
- `97661`..`97662`: augmentation compare on `B_fps/pb_t50_rs` (`none` vs `scanobjectnn`) (`2` jobs)
- `97663`..`97664`: drop-path compare on `B_fps/pb_t50_rs` (`base` vs `dp`) (`2` jobs)

Coverage conclusion vs requested `26`-job matrix:

- requested `26` combinations are now all submitted.
- additional `6` jobs are also queued (`rfps` non-aug, `obj_bg/obj_only/pb_t50_rs`, `LR=1e-4`) from the first-wave minimum.

Current scheduler snapshot:

- preprocess `97630` is `R`
- eval `97631`..`97664` are `H` on `afterok:97630`

## 8. Pretrain Corpus Fairness Policy (ScanObjectNN-in-pretrain)

### 8.1 Current risk classification

Current 1024 mixed pretrain configs used in many runs:

- `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml`
- `nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml`

Both include ScanObjectNN train-domain data in pretrain corpus (`scanobjectnn_main_split_v2` lineage).
This is acceptable for internal domain-adaptation study, but not ideal as the main
"fair comparison" protocol against methods that pretrain on ShapeNet-only.

### 8.2 Reporting policy (effective immediately)

- Main benchmark table:
  - use ShapeNet-only pretrain checkpoints.
- Secondary/ablation table:
  - report current ShapeNet+Scan mixed-pretrain checkpoints as domain-aware setting.
- For every published row, explicitly state pretrain corpus family:
  - `ShapeNet-only`
  - `ShapeNet+Scan (domain-aware)`

### 8.3 Required additional runs for fair baseline

Minimum additional pretrain work:

1. Add/confirm ShapeNet-only A/B pretrain configs
   - A-side can use existing `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf.yaml`
   - B-side requires an explicit ShapeNet-only XYZ-oriented counterpart (new config if missing)
2. Run ShapeNet-only pretrain for `A/B` (`2` jobs).
3. Run variant-split SOTA-fair eval on those ShapeNet-only A/B checkpoints
   (`obj_bg,obj_only,pb_t50_rs`), reporting `test_acc` as headline.

Interpretation rule:

- Until ShapeNet-only A/B checkpoints are evaluated, current mixed-pretrain
  variant tables should be marked as "internal/ablation", not final fair-comparison
  headline results.

## 9. Additional 256 Evaluation Submission (18 jobs)

Submitted on: `2026-02-26`

Purpose:

- add the requested 256-scale 18-eval bundle (`9 variants x 2 protocols`).
- reuse existing `a256_queryrethink` checkpoints; run CPAC(+mesh/chamfer) only.

Submission:

- script: `scripts/eval/submit_a256_queryrethink_cpac_retry_qf.sh`
- source checkpoints: `SOURCE_RUN_SET=a256_queryrethink_20260226_024537`
- new run set: `a256_queryrethink_cpac_retry2_20260226_135813`
- log root: `logs/eval/a256_queryrethink_cpac_retry2_20260226_135813`
- result root: `results/a256_queryrethink_cpac_retry2_20260226_135813`
- job list:
  - `logs/eval/a256_queryrethink_cpac_retry2_20260226_135813/submitted_jobs_a256_queryrethink_cpac_retry2_20260226_135813.txt`

Submitted jobs (`18`):

- `97667` `b00_interleave_theta_sotafair_cpacfix`
- `97668` `b00_interleave_theta_nepafull_cpacfix`
- `97669` `b01_split_theta_sotafair_cpacfix`
- `97670` `b01_split_theta_nepafull_cpacfix`
- `97671` `b02_split_theta_typepos_sotafair_cpacfix`
- `97672` `b02_split_theta_typepos_nepafull_cpacfix`
- `97673` `b03_split_viewraster_typepos_sotafair_cpacfix`
- `97674` `b03_split_viewraster_typepos_nepafull_cpacfix`
- `97675` `b04_split_xanchor_morton_typepos_sotafair_cpacfix`
- `97676` `b04_split_xanchor_morton_typepos_nepafull_cpacfix`
- `97677` `b05_split_xanchor_fps_typepos_sotafair_cpacfix`
- `97678` `b05_split_xanchor_fps_typepos_nepafull_cpacfix`
- `97679` `b06_split_dirfps_typepos_sotafair_cpacfix`
- `97680` `b06_split_dirfps_typepos_nepafull_cpacfix`
- `97681` `b07_event_xanchor_typepos_sotafair_cpacfix`
- `97682` `b07_event_xanchor_typepos_nepafull_cpacfix`
- `97683` `b08_event_dirfps_typepos_sotafair_cpacfix`
- `97684` `b08_event_dirfps_typepos_nepafull_cpacfix`

Immediate scheduler check:

- `97667`..`97684` all `job_state=R`.

## 10. Additional 256 Classification-Inclusive Eval (18 jobs)

Submitted on: `2026-02-26`

Purpose:

- run the requested 256 `9 variants x 2 protocols = 18` evals with classification enabled.
- keep CPAC/chamfer separate (`RUN_CPAC=0`) because CPAC stage uses pretrain `CKPT`
  directly and does not depend on fine-tune recipe/results.

Implementation:

- added script: `scripts/eval/submit_a256_queryrethink_eval18_qf.sh`
- run set: `a256_queryrethink_eval18_cls_20260226_140614`
- log root: `logs/eval/a256_queryrethink_eval18_cls_20260226_140614`
- result root: `results/a256_queryrethink_eval18_cls_20260226_140614`
- submitted jobs list:
  - `logs/eval/a256_queryrethink_eval18_cls_20260226_140614/submitted_jobs_a256_queryrethink_eval18_cls_20260226_140614.txt`

Submitted jobs (`18`):

- `97688` `b00_interleave_theta_sotafair_eval18`
- `97689` `b00_interleave_theta_nepafull_eval18`
- `97690` `b01_split_theta_sotafair_eval18`
- `97691` `b01_split_theta_nepafull_eval18`
- `97692` `b02_split_theta_typepos_sotafair_eval18`
- `97693` `b02_split_theta_typepos_nepafull_eval18`
- `97694` `b03_split_viewraster_typepos_sotafair_eval18`
- `97695` `b03_split_viewraster_typepos_nepafull_eval18`
- `97696` `b04_split_xanchor_morton_typepos_sotafair_eval18`
- `97697` `b04_split_xanchor_morton_typepos_nepafull_eval18`
- `97698` `b05_split_xanchor_fps_typepos_sotafair_eval18`
- `97699` `b05_split_xanchor_fps_typepos_nepafull_eval18`
- `97700` `b06_split_dirfps_typepos_sotafair_eval18`
- `97701` `b06_split_dirfps_typepos_nepafull_eval18`
- `97702` `b07_event_xanchor_typepos_sotafair_eval18`
- `97703` `b07_event_xanchor_typepos_nepafull_eval18`
- `97704` `b08_event_dirfps_typepos_sotafair_eval18`
- `97705` `b08_event_dirfps_typepos_nepafull_eval18`

Immediate scheduler check (at first submission):

- `97688`..`97705` were `job_state=R` before policy update in `10.1`.

### 10.1 Protocol enforcement update (no `main_split_v2`)

Policy update:

- `data/scanobjectnn_main_split_v2` is no longer allowed for fine-tune/eval benchmarks.
- variant cache must be specified explicitly (`obj_bg` / `obj_only` / `pb_t50_rs`).

Applied script guards:

- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
- `scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh`
- `scripts/eval/submit_ab_dualmask256_2proto_qf.sh`
- `scripts/eval/submit_a256_queryrethink_eval18_qf.sh`
- `scripts/pipeline/submit_a256_queryrethink_ablation_qf.sh`
- `scripts/finetune/run_scanobjectnn_m1_table_local.sh`

Behavior:

- when `RUN_SCAN=1`, `SCAN_CACHE_ROOT` (or `CACHE_ROOT`) is required.
- any `*scanobjectnn_main_split_v2*` path is rejected with error.

### 10.2 18-job re-submit aligned to variant cache

The previously submitted classification 18 jobs (`97688`..`97705`) used `main_split_v2`
and were terminated.

Re-submitted classification 18 jobs with explicit variant cache and dependency:

- run set: `a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`
- setting:
  - `SCAN_CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v2`
  - `RUN_CPAC=0`
  - `QSUB_DEPEND=afterok:97630.qjcm`
- jobs:
  - `97716`..`97733` (`18` jobs), all currently `H` waiting on `97630`.

## 11. Mesh+UDF-only 1024 pretrain submission

Submitted on: `2026-02-26`

Purpose:

- add ShapeNet `mesh+UDF` only pretrain at `1024` to support fair-comparison baseline work.
- exclude ScanObjectNN from pretrain corpus for this run.

Submission:

- script: `scripts/pretrain/submit_pretrain_mesh_udf_only_1024_qf.sh`
- mix config: `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf.yaml`
- run set: `meshudf_only1024_20260226_1435`
- job id record:
  - `logs/pretrain/meshudf_only1024_20260226_1435_job_ids.txt`

Submitted jobs:

- `97738.qjcm` `runA_meshudf_meshudf_only1024_20260226_1435`
  - `job_state=R` (immediate check)
  - save dir:
    - `runs/pretrain_meshudf_only_1024_meshudf_only1024_20260226_1435_runA`

Notes:

- default submit scope is `RUN_IDS=A` (minimal start).
- same submit helper can launch `A,C,D` by:
  - `RUN_IDS=A,C,D bash scripts/pretrain/submit_pretrain_mesh_udf_only_1024_qf.sh`

## 12. CPAC Retry2 Results Extract (18 jobs complete)

Target run set:

- `a256_queryrethink_cpac_retry2_20260226_135813`
- jobs: `97667`..`97684` (all `Exit_status=0`)
- results:
  - `results/a256_queryrethink_cpac_retry2_20260226_135813/cpac_abcd_1024_*.json`

Interpretation:

- This batch is CPAC-only (`RUN_SCAN=0`, `RUN_MODELNET=0`).
- Therefore, for each variant pair (`sotafair` / `nepafull`), CPAC metrics are identical.
- Mesh/chamfer fields were emitted but are invalid in this batch:
  - `mesh_eval.fail_count=800` (`n_eval_shapes=800`) for all variants
  - `chamfer_l2_n=0`, `chamfer_l1_n=0`, `fscore_n=0` for all variants

CPAC metrics by variant (same for both protocols):

| variant | mae | rmse | iou@tau | mesh fail | valid mesh metrics (l2/l1/fscore) |
|---|---:|---:|---:|---:|---:|
| `b00_interleave_theta` | 0.0312 | 0.0430 | 0.7737 | 800/800 | 0/0/0 |
| `b01_split_theta` | 0.2420 | 0.2850 | 0.0002 | 800/800 | 0/0/0 |
| `b02_split_theta_typepos` | 0.2426 | 0.2868 | 0.0001 | 800/800 | 0/0/0 |
| `b03_split_viewraster_typepos` | 0.2420 | 0.2859 | 0.0029 | 800/800 | 0/0/0 |
| `b04_split_xanchor_morton_typepos` | 0.2408 | 0.2867 | 0.0000 | 800/800 | 0/0/0 |
| `b05_split_xanchor_fps_typepos` | 0.2417 | 0.2859 | 0.0004 | 800/800 | 0/0/0 |
| `b06_split_dirfps_typepos` | 0.2406 | 0.2852 | 0.0002 | 800/800 | 0/0/0 |
| `b07_event_xanchor_typepos` | 0.2389 | 0.2840 | 0.0012 | 800/800 | 0/0/0 |
| `b08_event_dirfps_typepos` | 0.2433 | 0.2864 | 0.0008 | 800/800 | 0/0/0 |
