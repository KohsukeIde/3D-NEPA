# ScanObjectNN Review Tables (Active)

This page tracks active ScanObjectNN classification reruns and keeps the run lineage (`v0 -> v1 -> v2`) explicit.

Legacy mixed/causal history:

- `nepa3d/docs/results_scanobjectnn_review_legacy.md`

Snapshot time: `2026-02-19 01:27 JST`

## Run lineage (`v0 -> v1 -> v2`)

| Version | Purpose | Key protocol | Status | Run root | Main log |
|---|---|---|---|---|---|
| `v0` | poolfix baseline | `n_point=256`, `pt_xyz_key=pt_xyz_pool`, `pt_dist` enabled, `cls_is_causal=0`, `cls_pooling=mean_no_special`, `mc_eval_k_test=10` | FT complete (`225/225`) | `runs/scan_variants_review_ft_bidir_poolfix_v1` | `logs/finetune/scan_variants_review_lp_bidir_poolfix_resume/pipeline.log` |
| `v1` | fair FT (pc-xyz 2k, xyz-only) | `n_point=2048`, `allow_scale_up=1`, `pt_xyz_key=pc_xyz`, `ablate_point_dist=1`, `cls_is_causal=0`, `cls_pooling=mean_pts`, `mc_eval_k_test=1` | `obj_bg` complete (`75/75`), others not run | `runs/scan_variants_review_ft_fair_pcxyz2k_v1` | `logs/finetune/scan_variants_review_fair_ft_chain/pipeline.log` |
| `v2` | fair FT + explicit FPS eval | `v1` + `pt_sample_mode_train=random`, `pt_sample_mode_eval=fps`, `pt_fps_key=pt_fps_order` | running (`obj_bg 0/75` at snapshot) | `runs/scan_variants_review_ft_fair_pcxyz2k_fps_v2` | `logs/finetune/scan_variants_review_fair_ft_chain_v2_objbg/pipeline.log` |

## What changed between versions

- `v0 -> v1`:
  - input tokens switched from pool query (`pt_xyz_pool`) to observed point cloud (`pc_xyz`)
  - point count scaled from `256` to `2048` (`allow_scale_up=1`)
  - switched to xyz-only (`ablate_point_dist=1`)
  - voting changed from `mc_eval_k_test=10` to `mc_eval_k_test=1`
- `v1 -> v2`:
  - sampling behavior made explicit with FPS at eval (`pt_sample_mode_eval=fps`)
  - train sampling kept random (`pt_sample_mode_train=random`)

## Current run status

- `v1` started at `2026-02-18 15:36` and finished `obj_bg` (`75/75`) at `2026-02-19 00:47`.
- `v1` stage-2 did not proceed (pipeline script syntax error after stage-1 completion), so `obj_only` / `pb_t50_rs` remain `0/75`.
- `v2` started at `2026-02-19 00:49` for `obj_bg`; this is the current comparison run against `v1`.
- queued next chain (auto after `v2 obj_bg`): dist-enabled `v1(obj_bg)` then dist-enabled `v2(obj_bg)` at `logs/finetune/scan_variants_review_fair_ft_dist_after_v2_objbg_chain/pipeline.log`.

## Quick result gap (`obj_bg`, best-by-K)

| K | `v0` poolfix best | `v1` fair-random best | Delta (`v1-v0`) |
|---:|---:|---:|---:|
| 0 | 0.6644 | 0.4550 | -0.2094 |
| 20 | 0.4274 | 0.2266 | -0.2008 |

## Fair FT v1 update (`pcxyz2k`, xyz-only, random sampling)

### Fair FT v1 (`obj_bg`, complete)

Source: `runs/scan_variants_review_ft_fair_pcxyz2k_v1/obj_bg/scan_<method>_ablate_dist_k<K>_s<seed>/last.pt` (`n(seed)=3`).

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1612 +- 0.0201 |
| `scratch` | 1 | 3 | 0.1325 +- 0.0024 |
| `scratch` | 5 | 3 | 0.1268 +- 0.0072 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1343 +- 0.0000 |
| `shapenet_nepa` | 0 | 3 | 0.4274 +- 0.0123 |
| `shapenet_nepa` | 1 | 3 | 0.1228 +- 0.0297 |
| `shapenet_nepa` | 5 | 3 | 0.1767 +- 0.0191 |
| `shapenet_nepa` | 10 | 3 | 0.2020 +- 0.0008 |
| `shapenet_nepa` | 20 | 3 | 0.2238 +- 0.0446 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.4550 +- 0.0168 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1348 +- 0.0149 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1738 +- 0.0243 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.2020 +- 0.0239 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.2266 +- 0.0556 |
| `shapenet_mix_nepa` | 0 | 3 | 0.4360 +- 0.0133 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1113 +- 0.0138 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1756 +- 0.0049 |
| `shapenet_mix_nepa` | 10 | 3 | 0.2123 +- 0.0154 |
| `shapenet_mix_nepa` | 20 | 3 | 0.2197 +- 0.0404 |
| `shapenet_mix_mae` | 0 | 3 | 0.4131 +- 0.0156 |
| `shapenet_mix_mae` | 1 | 3 | 0.1262 +- 0.0179 |
| `shapenet_mix_mae` | 5 | 3 | 0.1595 +- 0.0059 |
| `shapenet_mix_mae` | 10 | 3 | 0.1836 +- 0.0296 |
| `shapenet_mix_mae` | 20 | 3 | 0.1910 +- 0.0169 |

### Fair FT v1 best-by-K (`obj_bg`)

| K | best method | test_acc mean +- std |
|---:|---|---:|
| 0 | `shapenet_mesh_udf_nepa` | 0.4550 +- 0.0168 |
| 1 | `shapenet_mesh_udf_nepa` | 0.1348 +- 0.0149 |
| 5 | `shapenet_nepa` | 0.1767 +- 0.0191 |
| 10 | `shapenet_mix_nepa` | 0.2123 +- 0.0154 |
| 20 | `shapenet_mesh_udf_nepa` | 0.2266 +- 0.0556 |

## TODO (next steps)

- [x] Add per-epoch training diagnostics to classification logs (`train_loss`, `train_acc`) in `nepa3d/train/finetune_cls.py`.
- [x] Merge v05 patch features needed for fair classification reruns:
  - `--aug_preset` / augmentation knobs in `finetune_cls.py`
  - dataset-side point augmentation and key-select support
  - tokenizer robustness for point-only (`n_ray=0`) runs while keeping legacy token ids
- [x] Pause LP for now (FT-first policy).
- [x] Run fair FT `v1` (`pc_xyz`, xyz-only, `n_point=2048`, bidirectional) for `obj_bg`.
- [x] Launch fair FT `v2` (`obj_bg`) with explicit FPS eval sampling.
- [ ] Aggregate `v2 obj_bg` results and compare against `v1 obj_bg` (same K/seed grid).
- [ ] Decide go/no-go for expanding `v2` to `obj_only` and `pb_t50_rs`.
- [ ] If FT still underperforms after `v2`, then re-open LP and model/head-side comparisons.
- [ ] Keep completion-side TODOs tracked separately (this page is classification-focused).
- [ ] Keep launcher references up to date:
  - launcher: `scripts/finetune/launch_scanobjectnn_review_fair_ft_chain_local.sh`
  - runner: `scripts/finetune/run_scanobjectnn_review_fair_ft_chain_local.sh`

## Protocol reference (`v0` poolfix rerun)

- attention: `cls_is_causal=0` (bidirectional)
- backend: `pointcloud_noray`
- input: `n_point=256`, `n_ray=0`
- eval voting: `mc_eval_k_test=10` (`mc_eval_k_val=1`)
- pooling: `cls_pooling=mean_no_special` (explicit)
- point keys: `pt_xyz_key=pt_xyz_pool`, `pt_dist_key=pt_dist_pool`

Run/log roots:

- full FT: `runs/scan_variants_review_ft_bidir_poolfix_v1`
- linear probe: `runs/scan_variants_review_lp_bidir_poolfix_v1`
- resume log: `logs/finetune/scan_variants_review_lp_bidir_poolfix_resume/pipeline.log`

## Completeness (snapshot)

- Full FT: `225/225` (`obj_bg=75`, `obj_only=75`, `pb_t50_rs=75`)
- Linear probe: `182/225` (`obj_bg=75`, `obj_only=75`, `pb_t50_rs=32`)
- LP was intentionally stopped at this snapshot (`FT-first` policy); `pb_t50_rs` LP remains partial.

## A) Full FT (poolfix)

### Full FT (`obj_bg`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5026 +- 0.0330 |
| `scratch` | 1 | 3 | 0.1400 +- 0.0081 |
| `scratch` | 5 | 3 | 0.1847 +- 0.0361 |
| `scratch` | 10 | 3 | 0.1360 +- 0.0024 |
| `scratch` | 20 | 3 | 0.1503 +- 0.0114 |
| `shapenet_nepa` | 0 | 3 | 0.6483 +- 0.0136 |
| `shapenet_nepa` | 1 | 3 | 0.1916 +- 0.0177 |
| `shapenet_nepa` | 5 | 3 | 0.2937 +- 0.0295 |
| `shapenet_nepa` | 10 | 3 | 0.3242 +- 0.0138 |
| `shapenet_nepa` | 20 | 3 | 0.4039 +- 0.0339 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6558 +- 0.0106 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1515 +- 0.0212 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2748 +- 0.0261 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.3270 +- 0.0156 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4125 +- 0.0298 |
| `shapenet_mix_nepa` | 0 | 3 | 0.6644 +- 0.0049 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1733 +- 0.0252 |
| `shapenet_mix_nepa` | 5 | 3 | 0.2490 +- 0.0242 |
| `shapenet_mix_nepa` | 10 | 3 | 0.3500 +- 0.0236 |
| `shapenet_mix_nepa` | 20 | 3 | 0.4274 +- 0.0221 |
| `shapenet_mix_mae` | 0 | 3 | 0.6122 +- 0.0091 |
| `shapenet_mix_mae` | 1 | 3 | 0.1870 +- 0.0289 |
| `shapenet_mix_mae` | 5 | 3 | 0.2628 +- 0.0178 |
| `shapenet_mix_mae` | 10 | 3 | 0.2846 +- 0.0150 |
| `shapenet_mix_mae` | 20 | 3 | 0.3723 +- 0.0133 |

### Full FT (`obj_only`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5800 +- 0.0106 |
| `scratch` | 1 | 3 | 0.1492 +- 0.0144 |
| `scratch` | 5 | 3 | 0.1291 +- 0.0193 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1423 +- 0.0059 |
| `shapenet_nepa` | 0 | 3 | 0.6575 +- 0.0126 |
| `shapenet_nepa` | 1 | 3 | 0.2203 +- 0.0356 |
| `shapenet_nepa` | 5 | 3 | 0.2754 +- 0.0074 |
| `shapenet_nepa` | 10 | 3 | 0.3804 +- 0.0123 |
| `shapenet_nepa` | 20 | 3 | 0.4188 +- 0.0215 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6707 +- 0.0155 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1721 +- 0.0282 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2719 +- 0.0316 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.3792 +- 0.0210 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4584 +- 0.0107 |
| `shapenet_mix_nepa` | 0 | 3 | 0.6512 +- 0.0155 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1658 +- 0.0081 |
| `shapenet_mix_nepa` | 5 | 3 | 0.2834 +- 0.0345 |
| `shapenet_mix_nepa` | 10 | 3 | 0.3523 +- 0.0197 |
| `shapenet_mix_nepa` | 20 | 3 | 0.4670 +- 0.0035 |
| `shapenet_mix_mae` | 0 | 3 | 0.6173 +- 0.0239 |
| `shapenet_mix_mae` | 1 | 3 | 0.1870 +- 0.0264 |
| `shapenet_mix_mae` | 5 | 3 | 0.2289 +- 0.0313 |
| `shapenet_mix_mae` | 10 | 3 | 0.2972 +- 0.0160 |
| `shapenet_mix_mae` | 20 | 3 | 0.4223 +- 0.0192 |

### Full FT (`pb_t50_rs`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5451 +- 0.0140 |
| `scratch` | 1 | 3 | 0.1371 +- 0.0164 |
| `scratch` | 5 | 3 | 0.1537 +- 0.0318 |
| `scratch` | 10 | 3 | 0.1425 +- 0.0101 |
| `scratch` | 20 | 3 | 0.1521 +- 0.0142 |
| `shapenet_nepa` | 0 | 3 | 0.5349 +- 0.0039 |
| `shapenet_nepa` | 1 | 3 | 0.1601 +- 0.0202 |
| `shapenet_nepa` | 5 | 3 | 0.2046 +- 0.0137 |
| `shapenet_nepa` | 10 | 3 | 0.2358 +- 0.0127 |
| `shapenet_nepa` | 20 | 3 | 0.2836 +- 0.0079 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.5452 +- 0.0048 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1323 +- 0.0020 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2001 +- 0.0290 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.2279 +- 0.0041 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.2787 +- 0.0074 |
| `shapenet_mix_nepa` | 0 | 3 | 0.5642 +- 0.0035 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1247 +- 0.0176 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1956 +- 0.0237 |
| `shapenet_mix_nepa` | 10 | 3 | 0.2436 +- 0.0166 |
| `shapenet_mix_nepa` | 20 | 3 | 0.2693 +- 0.0087 |
| `shapenet_mix_mae` | 0 | 3 | 0.5365 +- 0.0066 |
| `shapenet_mix_mae` | 1 | 3 | 0.1501 +- 0.0351 |
| `shapenet_mix_mae` | 5 | 3 | 0.1886 +- 0.0160 |
| `shapenet_mix_mae` | 10 | 3 | 0.2388 +- 0.0183 |
| `shapenet_mix_mae` | 20 | 3 | 0.2678 +- 0.0095 |

### Best-by-K (Full FT, complete n=3)

| Variant | K | best method | n(seed) | test_acc mean +- std |
|---|---:|---|---:|---:|
| `obj_bg` | 0 | `shapenet_mix_nepa` | 3 | 0.6644 +- 0.0049 |
| `obj_bg` | 1 | `shapenet_nepa` | 3 | 0.1916 +- 0.0177 |
| `obj_bg` | 5 | `shapenet_nepa` | 3 | 0.2937 +- 0.0295 |
| `obj_bg` | 10 | `shapenet_mix_nepa` | 3 | 0.3500 +- 0.0236 |
| `obj_bg` | 20 | `shapenet_mix_nepa` | 3 | 0.4274 +- 0.0221 |
| `obj_only` | 0 | `shapenet_mesh_udf_nepa` | 3 | 0.6707 +- 0.0155 |
| `obj_only` | 1 | `shapenet_nepa` | 3 | 0.2203 +- 0.0356 |
| `obj_only` | 5 | `shapenet_mix_nepa` | 3 | 0.2834 +- 0.0345 |
| `obj_only` | 10 | `shapenet_nepa` | 3 | 0.3804 +- 0.0123 |
| `obj_only` | 20 | `shapenet_mix_nepa` | 3 | 0.4670 +- 0.0035 |
| `pb_t50_rs` | 0 | `shapenet_mix_nepa` | 3 | 0.5642 +- 0.0035 |
| `pb_t50_rs` | 1 | `shapenet_nepa` | 3 | 0.1601 +- 0.0202 |
| `pb_t50_rs` | 5 | `shapenet_nepa` | 3 | 0.2046 +- 0.0137 |
| `pb_t50_rs` | 10 | `shapenet_mix_nepa` | 3 | 0.2436 +- 0.0166 |
| `pb_t50_rs` | 20 | `shapenet_nepa` | 3 | 0.2836 +- 0.0079 |

## B) Linear Probe (poolfix snapshot)

### Linear Probe (`obj_bg`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1452 +- 0.0154 |
| `scratch` | 1 | 3 | 0.1205 +- 0.0195 |
| `scratch` | 5 | 3 | 0.1239 +- 0.0223 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1354 +- 0.0016 |
| `shapenet_nepa` | 0 | 3 | 0.2255 +- 0.0028 |
| `shapenet_nepa` | 1 | 3 | 0.1107 +- 0.0324 |
| `shapenet_nepa` | 5 | 3 | 0.1348 +- 0.0219 |
| `shapenet_nepa` | 10 | 3 | 0.1629 +- 0.0053 |
| `shapenet_nepa` | 20 | 3 | 0.1629 +- 0.0096 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.2352 +- 0.0035 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1038 +- 0.0455 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1492 +- 0.0317 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.1314 +- 0.0165 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.1991 +- 0.0126 |
| `shapenet_mix_nepa` | 0 | 3 | 0.1991 +- 0.0085 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1038 +- 0.0293 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1102 +- 0.0171 |
| `shapenet_mix_nepa` | 10 | 3 | 0.1216 +- 0.0120 |
| `shapenet_mix_nepa` | 20 | 3 | 0.1561 +- 0.0203 |
| `shapenet_mix_mae` | 0 | 3 | 0.1870 +- 0.0082 |
| `shapenet_mix_mae` | 1 | 3 | 0.1348 +- 0.0403 |
| `shapenet_mix_mae` | 5 | 3 | 0.1452 +- 0.0450 |
| `shapenet_mix_mae` | 10 | 3 | 0.1394 +- 0.0244 |
| `shapenet_mix_mae` | 20 | 3 | 0.2037 +- 0.0296 |

### Linear Probe (`obj_only`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1383 +- 0.0082 |
| `scratch` | 1 | 3 | 0.1205 +- 0.0195 |
| `scratch` | 5 | 3 | 0.1205 +- 0.0195 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1360 +- 0.0024 |
| `shapenet_nepa` | 0 | 3 | 0.1928 +- 0.0098 |
| `shapenet_nepa` | 1 | 3 | 0.1262 +- 0.0323 |
| `shapenet_nepa` | 5 | 3 | 0.1239 +- 0.0184 |
| `shapenet_nepa` | 10 | 3 | 0.1285 +- 0.0187 |
| `shapenet_nepa` | 20 | 3 | 0.1606 +- 0.0164 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.2065 +- 0.0037 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1176 +- 0.0165 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1239 +- 0.0084 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.1182 +- 0.0160 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.1583 +- 0.0158 |
| `shapenet_mix_nepa` | 0 | 3 | 0.2255 +- 0.0268 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1182 +- 0.0049 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1119 +- 0.0366 |
| `shapenet_mix_nepa` | 10 | 3 | 0.1234 +- 0.0231 |
| `shapenet_mix_nepa` | 20 | 3 | 0.1624 +- 0.0113 |
| `shapenet_mix_mae` | 0 | 3 | 0.1819 +- 0.0086 |
| `shapenet_mix_mae` | 1 | 3 | 0.1262 +- 0.0325 |
| `shapenet_mix_mae` | 5 | 3 | 0.1256 +- 0.0256 |
| `shapenet_mix_mae` | 10 | 3 | 0.1371 +- 0.0268 |
| `shapenet_mix_mae` | 20 | 3 | 0.1819 +- 0.0276 |

### Linear Probe (`pb_t50_rs`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 2 | 0.1766 +- 0.0056 |
| `scratch` | 1 | 2 | 0.1355 +- 0.0002 |
| `scratch` | 5 | 1 | 0.1353 +- 0.0000 |
| `scratch` | 10 | 1 | 0.1353 +- 0.0000 |
| `scratch` | 20 | 1 | 0.1353 +- 0.0000 |
| `shapenet_nepa` | 0 | 2 | 0.2686 +- 0.0014 |
| `shapenet_nepa` | 1 | 2 | 0.0942 +- 0.0120 |
| `shapenet_nepa` | 5 | 1 | 0.1256 +- 0.0000 |
| `shapenet_nepa` | 10 | 1 | 0.1513 +- 0.0000 |
| `shapenet_nepa` | 20 | 1 | 0.1718 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 0 | 2 | 0.2621 +- 0.0019 |
| `shapenet_mesh_udf_nepa` | 1 | 1 | 0.1353 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 5 | 1 | 0.1353 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 10 | 1 | 0.1353 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 20 | 1 | 0.1520 +- 0.0000 |
| `shapenet_mix_nepa` | 0 | 2 | 0.2613 +- 0.0014 |
| `shapenet_mix_nepa` | 1 | 1 | 0.0510 +- 0.0000 |
| `shapenet_mix_nepa` | 5 | 1 | 0.0916 +- 0.0000 |
| `shapenet_mix_nepa` | 10 | 1 | 0.0899 +- 0.0000 |
| `shapenet_mix_nepa` | 20 | 1 | 0.1364 +- 0.0000 |
| `shapenet_mix_mae` | 0 | 2 | 0.2951 +- 0.0002 |
| `shapenet_mix_mae` | 1 | 1 | 0.1128 +- 0.0000 |
| `shapenet_mix_mae` | 5 | 1 | 0.1398 +- 0.0000 |
| `shapenet_mix_mae` | 10 | 1 | 0.1416 +- 0.0000 |
| `shapenet_mix_mae` | 20 | 1 | 0.1777 +- 0.0000 |

### Best-by-K (Linear Probe, complete n=3 only)

| Variant | K | best method | n(seed) | test_acc mean +- std |
|---|---:|---|---:|---:|
| `obj_bg` | 0 | `shapenet_mesh_udf_nepa` | 3 | 0.2352 +- 0.0035 |
| `obj_bg` | 1 | `shapenet_mix_mae` | 3 | 0.1348 +- 0.0403 |
| `obj_bg` | 5 | `shapenet_mesh_udf_nepa` | 3 | 0.1492 +- 0.0317 |
| `obj_bg` | 10 | `shapenet_nepa` | 3 | 0.1629 +- 0.0053 |
| `obj_bg` | 20 | `shapenet_mix_mae` | 3 | 0.2037 +- 0.0296 |
| `obj_only` | 0 | `shapenet_mix_nepa` | 3 | 0.2255 +- 0.0268 |
| `obj_only` | 1 | `shapenet_nepa` | 3 | 0.1262 +- 0.0323 |
| `obj_only` | 5 | `shapenet_mix_mae` | 3 | 0.1256 +- 0.0256 |
| `obj_only` | 10 | `shapenet_mix_mae` | 3 | 0.1371 +- 0.0268 |
| `obj_only` | 20 | `shapenet_mix_mae` | 3 | 0.1819 +- 0.0276 |
| `pb_t50_rs` | 0 | - | - | - |
| `pb_t50_rs` | 1 | - | - | - |
| `pb_t50_rs` | 5 | - | - | - |
| `pb_t50_rs` | 10 | - | - | - |
| `pb_t50_rs` | 20 | - | - | - |

## Notes

- This page is a **live snapshot**; LP `pb_t50_rs` values are provisional until `225/225` is reached.
- Use `nepa3d/docs/results_scanobjectnn_review_legacy.md` for older causal/mixed snapshots.
