# 1024 Variant-Split Re-evaluation (Active)

Last updated: 2026-02-26

> Transition note (2026-02-26):
> This file is now a compact policy/protocol document.
> Canonical benchmark summary moved to:
> `nepa3d/docs/active/benchmark_scanobjectnn_variant.md`
> Job-by-job updates moved to:
> `nepa3d/docs/active/runlog_202602.md`

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

## 7. Pretrain Corpus Fairness Policy (ScanObjectNN-in-pretrain)

### 7.1 Current risk classification

Current 1024 mixed pretrain configs used in many runs:

- `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml`
- `nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml`

Both include ScanObjectNN train-domain data in pretrain corpus (`scanobjectnn_main_split_v2` lineage).
This is acceptable for internal domain-adaptation study, but not ideal as the main
"fair comparison" protocol against methods that pretrain on ShapeNet-only.

### 7.2 Reporting policy (effective immediately)

- Main benchmark table:
  - use ShapeNet-only pretrain checkpoints.
- Secondary/ablation table:
  - report current ShapeNet+Scan mixed-pretrain checkpoints as domain-aware setting.
- For every published row, explicitly state pretrain corpus family:
  - `ShapeNet-only`
  - `ShapeNet+Scan (domain-aware)`

### 7.3 Required additional runs for fair baseline

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


## 8. Execution History (migrated)

- Detailed execution history sections were moved to:
  - `nepa3d/docs/active/runlog_202602.md`
- Canonical benchmark tables are maintained in:
  - `nepa3d/docs/active/benchmark_scanobjectnn_variant.md`
- This file now keeps only policy, protocol, and run-scope decisions.
