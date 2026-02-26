# ScanObjectNN Variant Benchmark (Active Canonical)

Last updated: 2026-02-26

## 1. Scope

This file is the canonical benchmark summary for protocol-correct ScanObjectNN evaluation.

- Variant-split only: `obj_bg`, `obj_only`, `pb_t50_rs`
- Headline metric: `test_acc`
- SOTA-fair and NEPA-full are always reported separately

Detailed job history is tracked in `nepa3d/docs/active/runlog_202602.md`.

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

## 5. Latest 256 Query-Rethink Classification Snapshot (18 jobs)

Source run set:

- `a256_queryrethink_eval18_cls_pb_t50_rs_20260226_141335`
- source table: `nepa3d/docs/pretrain_abcd_1024_variant_reval_active.md` section 15

ScanObjectNN (`test_acc`) best variant in this set:

- `b04_split_xanchor_morton_typepos`
  - `sotafair=0.5059`
  - `nepafull=0.3275`

## 6. Reporting Checklist

For each new table row, include:

1. checkpoint path
2. pretrain family
3. variant cache root
4. protocol (`sotafair` or `nepafull`)
5. job ID
6. `test_acc`
