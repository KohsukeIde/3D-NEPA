# ScanObjectNN Variant Benchmark (Active Canonical)

Last updated: 2026-02-26

## 1. Scope

This file is the canonical benchmark summary for protocol-correct ScanObjectNN evaluation.

- Variant-split only: `obj_bg`, `obj_only`, `pb_t50_rs`
- Headline metric: `test_acc`
- SOTA-fair and NEPA-full are always reported separately

Detailed job history is tracked in `nepa3d/docs/active/runlog_202602.md`.

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
  - `data/scanobjectnn_obj_bg_v2`
  - `data/scanobjectnn_obj_only_v2`
  - `data/scanobjectnn_pb_t50_rs_v2`
- No mixed-cache headline table (`data/scanobjectnn_main_split_v2`) for fair comparison.
- `test_acc` is benchmark headline.
- `best_val` / `best_ep` are diagnostics only.

## 3. Checkpoint Families in Use

- `A_fps`: `runs/pretrain_abcd_1024_runA/last.pt`
- `B_fps`: `runs/pretrain_abcd_1024_runB/last.pt`
- `A_rfps_aug`: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt`
- `B_rfps_aug`: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt`

## 4. Latest 1024 Variant-Split Snapshot (minimum 12 jobs)

Source: `nepa3d/docs/pretrain_abcd_1024_variant_reval_active.md` section 18.1

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

### 4.2 Post-backfill NEPA-full check (`98018`/`98019`/`98021`/`98022`)

Scope:

- cache: `data/scanobjectnn_pb_t50_rs_v2` (after `pt_fps_order` backfill)
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
- source table: `nepa3d/docs/pretrain_abcd_1024_variant_reval_active.md` section 15

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
