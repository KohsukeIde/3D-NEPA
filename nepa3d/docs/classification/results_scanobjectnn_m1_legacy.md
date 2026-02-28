# ScanObjectNN M1 Legacy Snapshot

This file stores the legacy/internal M1 few-shot table that was previously embedded in `nepa3d/README.md`.

- Source cache: `data/scanobjectnn_cache_v2` (legacy naming)
- Status snapshot date: February 14, 2026

## Legacy table (`75/75` complete)

Status (as of February 14, 2026):

- completed jobs: `75 / 75`
- completion by method:
  - `scratch`: `15/15`
  - `shapenet_nepa`: `15/15`
  - `shapenet_mesh_udf_nepa`: `15/15`
  - `shapenet_mix_nepa`: `15/15`
  - `shapenet_mix_mae`: `15/15`

Dataset/caching note for this table:

- This completed `75/75` table was produced on `CACHE_ROOT=data/scanobjectnn_cache_v2` (legacy cache naming).
- Repro runs for paper should use a split-specific cache root (Section 2.2).

Table below is computed from `runs/scan_<method>_k<K>_s<seed>/last.pt`.
`n(seed)=3` for all rows.

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.8204 +- 0.0039 |
| `scratch` | 1 | 3 | 0.1314 +- 0.0222 |
| `scratch` | 5 | 3 | 0.1549 +- 0.0134 |
| `scratch` | 10 | 3 | 0.1115 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1115 +- 0.0000 |
| `shapenet_nepa` | 0 | 3 | 0.8076 +- 0.0023 |
| `shapenet_nepa` | 1 | 3 | 0.1553 +- 0.0289 |
| `shapenet_nepa` | 5 | 3 | 0.2259 +- 0.0102 |
| `shapenet_nepa` | 10 | 3 | 0.2569 +- 0.0145 |
| `shapenet_nepa` | 20 | 3 | 0.3197 +- 0.0079 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.8170 +- 0.0011 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1413 +- 0.0177 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2524 +- 0.0122 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.3016 +- 0.0166 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.3562 +- 0.0069 |
| `shapenet_mix_nepa` | 0 | 3 | 0.8264 +- 0.0018 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1408 +- 0.0153 |
| `shapenet_mix_nepa` | 5 | 3 | 0.2501 +- 0.0047 |
| `shapenet_mix_nepa` | 10 | 3 | 0.3128 +- 0.0149 |
| `shapenet_mix_nepa` | 20 | 3 | 0.3753 +- 0.0054 |
| `shapenet_mix_mae` | 0 | 3 | 0.7883 +- 0.0028 |
| `shapenet_mix_mae` | 1 | 3 | 0.1611 +- 0.0182 |
| `shapenet_mix_mae` | 5 | 3 | 0.2588 +- 0.0125 |
| `shapenet_mix_mae` | 10 | 3 | 0.2750 +- 0.0107 |
| `shapenet_mix_mae` | 20 | 3 | 0.3107 +- 0.0077 |

Readout:

- Full (`K=0`) best: `shapenet_mix_nepa` (`0.8264`)
- Low-shot (`K=1,5`) best: `shapenet_mix_mae`
- Mid/high-shot (`K=10,20`) best: `shapenet_mix_nepa`

Protocol details for this table:

- optimizer/eval seeds:
  - `SEED in {0,1,2}`
  - `VAL_SEED=0` (fixed)
  - `EVAL_SEED=0` (fixed)
- few-shot subset seed:
  - `K=0`: `fewshot_seed=0`
  - `K>0`: `fewshot_seed=SEED`
- MC evaluation:
  - `mc_eval_k_val=1`
  - `mc_eval_k_test=4`

Method/run-name mapping (exact):

- `scratch` -> `runs/scan_scratch_k<K>_s<seed>/`
- `shapenet_nepa` -> `runs/scan_shapenet_nepa_k<K>_s<seed>/`
- `shapenet_mesh_udf_nepa` -> `runs/scan_shapenet_mesh_udf_nepa_k<K>_s<seed>/`
- `shapenet_mix_nepa` -> `runs/scan_shapenet_mix_nepa_k<K>_s<seed>/`
- `shapenet_mix_mae` -> `runs/scan_shapenet_mix_mae_k<K>_s<seed>/`

## Notes carried from README

- The `75/75` M1 table in Section 6 is a legacy/internal snapshot from `CACHE_ROOT=data/scanobjectnn_cache_v2`.
- For camera-ready tables, use split-specific caches (Section 2.2) and protocol-variant tables (Section 6.1).
- Current tables are aggregated from `last.pt`.
- In `finetune_cls.py`, `last.pt` is saved after loading the best-val model state, so `last.pt` and `best.pt` are consistent for final test readout.
- `scratch K=10/20` shows majority-class-collapse behavior in current seeds.
- This collapse is not a K-shot sampling bug: `stratified_kshot` gives exact per-class counts (`K x 15 classes`), observed `test_acc=0.1115` matches test majority ratio (`2440/21889=0.11147`), and observed best-val around `0.1249` matches val majority ratio (`743/5948=0.12492`).
- For paper, include collapse diagnostics (train/val curves and prediction distribution), and optionally scratch linear-probe/head-only runs.
- ScanObjectNN task here is not raw point-set classification; model input is query-token sequence with `POINT xyz + dist` (`pt_dist_pool`) and optional ray channels.
- MC evaluation is used (`mc_eval_k_test=4` in current setup).
- Comparisons to raw-point baselines must explicitly note this representation/evaluation difference.
- ScanObjectNN classification is a downstream/supporting benchmark; main evidence for unpaired cross-primitive capability should be UCPR/CPAC (Section 8).
