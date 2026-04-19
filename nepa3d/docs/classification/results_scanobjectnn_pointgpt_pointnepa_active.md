# ScanObjectNN PointGPT / pointNEPA Sidecar Results (Active)

Snapshot time: `2026-04-19 JST` (verified against local summaries through `2026-04-12T15:28:59+09:00`)

Scope:

- This page tracks local PointGPT / pointNEPA sidecar experiments on ScanObjectNN.
- This page is **not** the canonical PatchNEPA benchmark headline ledger.
- Use `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` for current PatchNEPA headline tables.

Definitions used on this page:

- `pointNEPA`
  - PointGPT scaffold with `nepa_cosine` pretrain loss only
  - no decoder / Chamfer pretrain loss
  - fine-tune with `cls-only` (`ft_recon_weight=0`)
- `pointNEPA-S (mask-on)`
  - `PointGPT-S` + `nepa_cosine` + `mask_ratio=0.7` + `cls-only FT`
- `pointNEPA-S (mask-off)`
  - `PointGPT-S` + `nepa_cosine` + `mask_ratio=0.0` + `cls-only FT`
- `pointNEPA-S (vit-shift)`
  - `PointGPT-S` variant that keeps the PointGPT causal extractor but moves the one-step shift to the loss side to match `models/vit_nepa` more closely

Primary source summaries:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_b_20260312_summary.md`
- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_s_20260318_summary.md`
- `logs/local/pointgpt_ft_recipe_matrix_2x2/pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointnepa_s_maskoff_20260403_212525_summary.md`
- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointnepa_s_vitshift_maskoff_20260403_221453_summary.md`

## Official checkpoint protocol compare

### PointGPT-B official checkpoint

Source summary:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_b_20260312_summary.md`

Checkpoint:

- `PointGPT/checkpoints/official/pointgpt_b_post_pretrain_official.pth`

| Variant | `test-as-val` test_acc_plain | `strict(train->val)` test_acc_plain |
|---|---:|---:|
| `obj_bg` | `96.7298` | `96.7298` |
| `objonly` | `94.4923` | `94.4923` |
| `hardest` | `91.6031` | `90.5968` |

### PointGPT-S official checkpoint

Source summary:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_s_20260318_summary.md`

Checkpoint:

- `PointGPT/checkpoints/official/pointgpt_s_pretrain_official.pth`

| Variant | `test-as-val` test_acc_plain | `strict(train->val)` test_acc_plain |
|---|---:|---:|
| `obj_bg` | `90.0172` | `89.8451` |
| `objonly` | `91.222` | `87.2633` |
| `hardest` | `86.086` | `85.6697` |

## Local PointGPT-B objective x FT recipe matrix (complete)

Master summary:

- `logs/local/pointgpt_ft_recipe_matrix_2x2/pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md`

| Pretrain source | FT recipe | `obj_bg` best_acc | `objonly` best_acc | `hardest` best_acc | Arm summary |
|---|---|---:|---:|---:|---|
| `nepa_cosine` | `cls-only` | `89.17526245117188` | `89.0034408569336` | `84.76751708984375` | `logs/local/pointgpt_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `nepa_cosine` | `PointGPT FT (cls+recon)` | `89.86254119873047` | `89.5188980102539` | `84.6634292602539` | `logs/local/pointgpt_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `cdl12` | `cls-only` | `88.83161926269531` | `89.0034408569336` | `84.21234893798828` | `logs/local/pointgpt_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `cdl12` | `PointGPT FT (cls+recon)` | `90.03436279296875` | `89.17526245117188` | `85.14920043945312` | `logs/local/pointgpt_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |

## Local PointGPT-S objective x FT recipe matrix (complete)

Run tag:

- `pointgpt_s_ft_recipe_matrix_2x2_20260318`

Completed arm summaries:

- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`

| Pretrain source | FT recipe | `obj_bg` best_acc | `objonly` best_acc | `hardest` best_acc | Status |
|---|---|---:|---:|---:|---|
| `nepa_cosine` | `cls-only` | `90.03436279296875` | `90.20618438720703` | `85.39208984375` | complete |
| `nepa_cosine` | `PointGPT FT (cls+recon)` | `91.06529235839844` | `90.03436279296875` | `85.46147918701172` | complete |
| `cdl12` | `cls-only` | `90.03436279296875` | `89.5188980102539` | `83.83067321777344` | complete |
| `cdl12` | `PointGPT FT (cls+recon)` | `91.58075714111328` | `89.86254119873047` | `84.62872314453125` | complete |

## pointNEPA-S readout

### pointNEPA-S (mask-on) baseline

This is the already-completed `PointGPT-S + nepa_cosine + mask_ratio=0.7 + cls-only FT` run.

Source summary:

- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`

Results:

- `obj_bg`: `90.03436279296875`
- `objonly`: `90.20618438720703`
- `hardest`: `85.39208984375`

### pointNEPA-S (mask-off, current-shift)

Config:

- `PointGPT/cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0_nomask.yaml`

Runtime metadata:

- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525.meta.env`

Experiment path:

- `PointGPT/experiments/pretrain_nepa_cosine_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525`

Pretrain log:

- `PointGPT/experiments/pretrain_nepa_cosine_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525/20260403_212535.log`

Source summary:

- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointnepa_s_maskoff_20260403_212525_summary.md`

Results:

- `obj_bg`: `90.37801361083984`
- `objonly`: `90.37801361083984`
- `hardest`: `84.52462768554688`

Relative to the mask-on baseline:

- `obj_bg`: `+0.34365081787109`
- `objonly`: `+0.17182922363281`
- `hardest`: `-0.86746215820312`

### pointNEPA-S (vit-shift, mask-off)

Intent:

- keep the PointGPT patch encoder and causal extractor
- remove the input-side one-step shift
- move the one-step shift into the loss to match `models/vit_nepa` more closely

Config:

- `PointGPT/cfgs/PointGPT-S/pretrain_nepa_cosine_vitshift_shapenet_cache_v0_nomask.yaml`

Runtime metadata:

- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointgpt_s_nepa_cosine_vitshift_shapenet_cache_v0_nomask_pointnepa_s_vitshift_maskoff_20260403_221453.meta.env`

Experiment path:

- `PointGPT/experiments/pretrain_nepa_cosine_vitshift_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_vitshift_shapenet_cache_v0_nomask_pointnepa_s_vitshift_maskoff_20260403_221453`

Source summary:

- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointnepa_s_vitshift_maskoff_20260403_221453_summary.md`

Results:

- `obj_bg`: `90.72164916992188`
- `objonly`: `89.0034408569336`
- `hardest`: `84.21234893798828`

Relative to the mask-off current-shift arm:

- `obj_bg`: `+0.34363555908204`
- `objonly`: `-1.37457275390624`
- `hardest`: `-0.31227874755860`

## Current readout

- `PointGPT-S + nepa_cosine` is operational on ScanObjectNN; the `mask-on` `cls-only` arm is already at:
  - `obj_bg=90.03436279296875`
  - `objonly=90.20618438720703`
  - `hardest=85.39208984375`
- `pointNEPA-S (mask-off, current-shift)` is complete:
  - `obj_bg=90.37801361083984`
  - `objonly=90.37801361083984`
  - `hardest=84.52462768554688`
- `pointNEPA-S (vit-shift, mask-off)` is complete:
  - `obj_bg=90.72164916992188`
  - `objonly=89.0034408569336`
  - `hardest=84.21234893798828`
- `PointGPT-S` official checkpoint protocol compare is complete and recorded here.
- `PointGPT-B` local 2x2 matrix is complete and recorded here.
- `PointGPT-S` local 2x2 matrix is complete; `cdl12 x PointGPT FT` reached `obj_bg=91.58075714111328`, `objonly=89.86254119873047`, `hardest=84.62872314453125`.
- Do not treat this page as the PatchNEPA benchmark headline; use it as the active PointGPT / pointNEPA sidecar ledger.
