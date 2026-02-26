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
  - `97716`..`97733` (`18` jobs), all completed (`job_state=F`, `Exit_status=0`).
  - final classification metrics are summarized in §15.

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
  - root cause (verified by 10-shape debug with `--mesh_store_per_shape 1`):
    - per-shape error: `No module named 'skimage'`
    - `completion_cpac_udf.py` uses `skimage.measure.marching_cubes` for mesh reconstruction;
      without `scikit-image`, mesh stage fails for every shape.

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

## 13. CPAC seq2seq + mesh pipeline validation patch (2026-02-26)

### 13.1 Root cause confirmed for `split_sep` checkpoints

`completion_cpac_udf.py` had a serialization mismatch for `qa_layout=split_sep`:

- CPAC path built split-style sequence without `[SEP]` token.
- query-position extraction treated only `split` as split-layout and routed `split_sep` through interleave indexing.
- mesh/cpac max-length precheck did not include the extra separator token.

Patch applied:

- `nepa3d/analysis/completion_cpac_udf.py`
  - added `TYPE_SEP` handling in both CPAC feature builders
  - unified split handling for `split` + `split_sep` in query-index extraction
  - included `split_sep` separator in required sequence-length check

Sanity rerun (small subset, same CPAC protocol):

- `b00_interleave_theta`:
  - `--head_train_max_shapes 200 --max_shapes 80` -> `iou@tau=0.8026`
- `b04_split_xanchor_morton_typepos`:
  - `--head_train_max_shapes 200 --max_shapes 80` -> `iou@tau=0.6697`

Interpretation:

- prior near-zero CPAC of `b01`..`b08` in §12 is consistent with evaluation-side layout mismatch,
  not sufficient evidence of model collapse.

### 13.2 Mesh/chamfer path unblocked

Two independent issues were fixed:

1. Runtime dependency:
   - installed `scikit-image` in `.venv` (`skimage.measure.marching_cubes` dependency).
2. API compatibility:
   - `completion_cpac_udf.py`: robust `mesh_metrics(...)` kwargs dispatch by signature.
   - `nepa3d/analysis/mesh_metrics.py`: `trimesh.sample.sample_surface` seed compatibility
     (`int` seed first, fallback to `RandomState`).

Mesh smoke result after fixes:

- `b04_split_xanchor_morton_typepos`, `--head_train_max_shapes 20 --max_shapes 2 --mesh_eval=1`
  - `mesh_eval.fail_count=0`
  - `chamfer_l2_n=2`, `chamfer_l1_n=2`, `fscore_n=2`

Note:

- §12 mesh fail (`800/800`) and `b01..b08` near-zero CPAC should now be treated as
  **pre-fix historical artifact**.

## 14. CPAC/mesh official rerun submission (post-fix, 18 jobs)

Submitted on: `2026-02-26`

Purpose:

- run the full 18-job `a256_queryrethink` CPAC-only matrix with the fixed evaluator:
  - `split_sep` serialization/query-index/max-len fixes in `completion_cpac_udf.py`
  - mesh path fixes (`skimage` dependency + `mesh_metrics` API compatibility + trimesh seed compatibility)

Submission:

- script: `scripts/eval/submit_a256_queryrethink_cpac_retry_qf.sh`
- run set: `a256_queryrethink_cpac_retry3_20260226_152442`
- log root:
  - `logs/eval/a256_queryrethink_cpac_retry3_20260226_152442`
- result root:
  - `results/a256_queryrethink_cpac_retry3_20260226_152442`
- submitted jobs record:
  - `logs/eval/a256_queryrethink_cpac_retry3_20260226_152442/submitted_jobs_a256_queryrethink_cpac_retry3_20260226_152442.txt`

Submitted jobs (`18`):

- `97791.qjcm` `b00_interleave_theta_sotafair_cpacfix`
- `97792.qjcm` `b00_interleave_theta_nepafull_cpacfix`
- `97793.qjcm` `b01_split_theta_sotafair_cpacfix`
- `97794.qjcm` `b01_split_theta_nepafull_cpacfix`
- `97795.qjcm` `b02_split_theta_typepos_sotafair_cpacfix`
- `97796.qjcm` `b02_split_theta_typepos_nepafull_cpacfix`
- `97797.qjcm` `b03_split_viewraster_typepos_sotafair_cpacfix`
- `97798.qjcm` `b03_split_viewraster_typepos_nepafull_cpacfix`
- `97799.qjcm` `b04_split_xanchor_morton_typepos_sotafair_cpacfix`
- `97800.qjcm` `b04_split_xanchor_morton_typepos_nepafull_cpacfix`
- `97801.qjcm` `b05_split_xanchor_fps_typepos_sotafair_cpacfix`
- `97802.qjcm` `b05_split_xanchor_fps_typepos_nepafull_cpacfix`
- `97803.qjcm` `b06_split_dirfps_typepos_sotafair_cpacfix`
- `97804.qjcm` `b06_split_dirfps_typepos_nepafull_cpacfix`
- `97805.qjcm` `b07_event_xanchor_typepos_sotafair_cpacfix`
- `97806.qjcm` `b07_event_xanchor_typepos_nepafull_cpacfix`
- `97807.qjcm` `b08_event_dirfps_typepos_sotafair_cpacfix`
- `97808.qjcm` `b08_event_dirfps_typepos_nepafull_cpacfix`

Immediate scheduler check:

- all `97791`..`97808` are `job_state=R`.

## 15. 256 classification results (eval18, `pb_t50_rs`, 18 jobs complete)

Status check (`97716`..`97733`):

- all jobs: `job_state=F`, `Exit_status=0`
- run set: `a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`
- log root:
  - `logs/eval/a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`
- result root:
  - `results/a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`

Run settings:

- `SCAN_CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v2`
- `RUN_SCAN=1`, `RUN_MODELNET=1`, `RUN_CPAC=0`

Final metrics (from `*_classification_scan.log` and `*_classification_modelnet.log`):

| variant | protocol | scan `best_val` | scan `best_ep` | scan `test_acc` | modelnet `best_val` | modelnet `best_ep` | modelnet `test_acc` |
|---|---|---:|---:|---:|---:|---:|---:|
| `b00_interleave_theta` | `sotafair` | 0.5443 | 70 | 0.4342 | 0.8677 | 54 | 0.8568 |
| `b00_interleave_theta` | `nepafull` | 0.3220 | 88 | 0.2861 | 0.6206 | 93 | 0.6084 |
| `b01_split_theta` | `sotafair` | 0.5208 | 51 | 0.4261 | 0.8682 | 69 | 0.8551 |
| `b01_split_theta` | `nepafull` | 0.3351 | 61 | 0.2998 | 0.7104 | 98 | 0.7122 |
| `b02_split_theta_typepos` | `sotafair` | 0.5833 | 76 | 0.4753 | 0.8594 | 71 | 0.8555 |
| `b02_split_theta_typepos` | `nepafull` | 0.3151 | 86 | 0.2904 | 0.6152 | 96 | 0.6003 |
| `b03_split_viewraster_typepos` | `sotafair` | 0.5095 | 86 | 0.4245 | 0.8750 | 65 | 0.8630 |
| `b03_split_viewraster_typepos` | `nepafull` | 0.3351 | 64 | 0.3044 | 0.7036 | 93 | 0.7344 |
| `b04_split_xanchor_morton_typepos` | `sotafair` | 0.6441 | 89 | 0.5059 | 0.8721 | 87 | 0.8652 |
| `b04_split_xanchor_morton_typepos` | `nepafull` | 0.3663 | 71 | 0.3275 | 0.7563 | 90 | 0.7738 |
| `b05_split_xanchor_fps_typepos` | `sotafair` | 0.5920 | 79 | 0.4785 | 0.8672 | 91 | 0.8607 |
| `b05_split_xanchor_fps_typepos` | `nepafull` | 0.3255 | 61 | 0.2894 | 0.7363 | 84 | 0.7588 |
| `b06_split_dirfps_typepos` | `sotafair` | 0.5964 | 76 | 0.4863 | 0.8657 | 50 | 0.8652 |
| `b06_split_dirfps_typepos` | `nepafull` | 0.3429 | 70 | 0.3079 | 0.7402 | 97 | 0.7562 |
| `b07_event_xanchor_typepos` | `sotafair` | 0.4575 | 85 | 0.4206 | 0.8472 | 59 | 0.8438 |
| `b07_event_xanchor_typepos` | `nepafull` | 0.3212 | 96 | 0.2884 | 0.5933 | 94 | 0.5827 |
| `b08_event_dirfps_typepos` | `sotafair` | 0.5113 | 81 | 0.4554 | 0.8647 | 48 | 0.8545 |
| `b08_event_dirfps_typepos` | `nepafull` | 0.3220 | 85 | 0.2975 | 0.5952 | 94 | 0.5765 |

Quick read:

- ScanObjectNN (`test_acc`) best in both protocols: `b04_split_xanchor_morton_typepos`
  - `sotafair=0.5059`, `nepafull=0.3275`
- ModelNet40 (`test_acc`) best:
  - `sotafair`: `b04_split_xanchor_morton_typepos` and `b06_split_dirfps_typepos` tie at `0.8652`
  - `nepafull`: `b04_split_xanchor_morton_typepos=0.7738`

## 16. ShapeNet pointcloud-only 1024 pretrain (new fairness baseline)

Submitted on: `2026-02-26`

Purpose:

- add missing `ShapeNet-only pointcloud` pretrain baseline (no ScanObjectNN in pretrain corpus).
- use this as additional fair-comparison baseline alongside `mesh+UDF-only`.

Added files:

- config:
  - `nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only.yaml`
  - dataset: `shapenet_pc` only (`data/shapenet_cache_v0`, `backend=pointcloud_noray`, `weight=1.0`)
- submit helper:
  - `scripts/pretrain/submit_pretrain_shapenet_pointonly_1024_qf.sh`
  - default scope: `RUN_IDS=B` (XYZ-only profile)

Submission:

- command:
  - `RUN_SET=shapenet_pointonly1024_20260226_154047 JOB_IDS_OUT=logs/pretrain/shapenet_pointonly1024_job_ids.txt bash scripts/pretrain/submit_pretrain_shapenet_pointonly_1024_qf.sh`
- submitted job:
  - `97814.qjcm` `runB_shapenetpc_shapenet_pointonly1024_20260226_154047`
- immediate state:
  - `job_state=R`

Runtime args (from `qstat -fx`):

- `MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only.yaml`
- `N_POINT=1024`, `N_RAY=0`, `QA_TOKENS=0`, `MAX_LEN=2500`
- `PT_XYZ_KEY=pc_xyz`, `ABLATE_POINT_DIST=1`, `POINT_ORDER_MODE=morton`
- checkpoint dir:
  - `runs/pretrain_shapenet_pointonly_1024_shapenet_pointonly1024_20260226_154047_runB`

## 17. 1024 eval status update (`97659`..`97664`)

Checked on: `2026-02-26`

Status:

- `97659` (`eval_runA`): `job_state=R` (running)
- `97660`..`97664` (`eval_runB`): all `job_state=F`, `Exit_status=0`

Scope confirmation:

- these jobs are `1024` eval runs (`N_POINT_CLS=1024`, `N_RAY_CLS=0`)
- protocol: `SOTA-fair` on `pb_t50_rs`
  - `PT_XYZ_KEY_CLS=pc_xyz`, `ABLATE_POINT_DIST=1`
  - `SCAN_CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v2`
  - `RUN_CPAC=0` (classification-only)

Completed metrics (`runB_classification_scan.log`):

| job | setting | best_val | best_ep | test_acc |
|---|---|---:|---:|---:|
| `97660` | `B_rfps_aug`, `LR=5e-4`, `aug=scanobjectnn` | 0.2474 | 230 | 0.2402 |
| `97661` | `B_fps`, `LR=5e-4`, `aug=none` | 0.7049 | 234 | 0.5716 |
| `97662` | `B_fps`, `LR=5e-4`, `aug=scanobjectnn` | 0.7361 | 245 | 0.5505 |
| `97663` | `B_fps`, `LR=5e-4`, `drop_path=0.0` | 0.7422 | 254 | 0.5544 |
| `97664` | `B_fps`, `LR=5e-4`, `drop_path=0.1` | 0.7526 | 281 | 0.5641 |
