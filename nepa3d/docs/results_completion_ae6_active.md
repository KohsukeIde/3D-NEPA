# Completion A-E/6 Active Results

This file isolates completion-side ablations and scaling runs (A-E, 6) from the main UCPR/CPAC log.

- Scope: Progress-8 merge, A/B/C/D/E/6 runs, and A-1 coarse-to-fine
- Main track (NEPA/MAE): `results_ucpr_cpac_active.md`
- Planning doc: `plan_completion_ae6.md`
- Source archive: `results_ucpr_cpac_active_mixed_archive.md`

## Progress-8 Merge + Scaling-6 Smoke (Feb 16, 2026)

This cycle starts integration of `/home/cvrt/Downloads/nepa_progress_8` into the active repo (`/home/cvrt/Desktop/dev/3D-NEPA`) and validates scaling hooks with small smoke runs.

Merged code scope:

- max-length/resizing utility:
  - `nepa3d/utils/ckpt_utils.py` (new)
- pretrain/finetune scaling:
  - `nepa3d/train/pretrain.py`
  - `nepa3d/train/finetune_cls.py`
  - `nepa3d/data/dataset.py`
  - `nepa3d/data/mixed_pretrain.py`
- eval max-len override:
  - `nepa3d/analysis/retrieval_ucpr.py` (merged while keeping tie-aware ranking + sanity flags)
  - `nepa3d/analysis/completion_cpac_udf.py`
  - `nepa3d/analysis/qualitative_cpac_marching_cubes.py`
- wrappers:
  - `scripts/pretrain/nepa3d_pretrain.sh`
  - `scripts/analysis/nepa3d_ucpr.sh`
  - `scripts/analysis/nepa3d_cpac_udf.sh`

### Commands used (smoke)

```bash
# 1) pretrain scaling schedule smoke (2 epochs)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 128 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 \
  --dual_mask_warmup_frac 0.05 \
  --epochs 2 --batch 8 \
  --n_point 64 --n_ray 32 \
  --max_len 512 \
  --n_point_schedule "0:64,1:96" \
  --n_ray_schedule "0:32,1:64" \
  --num_workers 2 \
  --save_every 1 --save_last 1 \
  --resume_optimizer 1 \
  --save_dir runs/_tmp_scale_sched_smoke_fresh \
  --seed 0

# 2) CPAC pool/grid with and without max_len scaling (small subset)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --baseline nn_copy \
  --max_shapes 120 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --n_context 256 --n_query 256 --query_source pool \
  --out_json results/cpac_nepa_qa_dualmask_ep049_pc2udf_120_nontrans_pool_n256_smoke.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --max_len 4096 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 120 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --n_context 512 --n_query 512 \
  --out_json results/cpac_nepa_qa_dualmask_ep049_pc2udf_120_nontrans_n512_maxlen4096_smoke.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid --baseline nn_copy \
  --max_shapes 120 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --n_context 256 --n_query 256 \
  --out_json results/cpac_nepa_qa_dualmask_ep049_pc2udf_120_nontrans_grid_n256_smoke.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --max_len 4096 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid --baseline nn_copy \
  --max_shapes 120 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --n_context 512 --n_query 512 \
  --out_json results/cpac_nepa_qa_dualmask_ep049_pc2udf_120_nontrans_grid_n512_maxlen4096_smoke.json

# 3) UCPR smoke with max_len override and larger n_point/n_ray
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --max_len 4096 \
  --query_backend mesh --gallery_backend udfgrid \
  --n_point 512 --n_ray 512 \
  --eval_seed 0 --eval_seed_gallery 999 \
  --max_files 200 --pooling mean_a \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_mesh2udf_200_indep_mean_a_n512_maxlen4096_smoke.json
```

### Results (smoke)

CPAC small-subset comparison (`max_shapes=120`, `head_train_max_shapes=1000`):

| Query source | n_context/n_query | MAE | RMSE | IoU@0.03 | NN-copy MAE/RMSE/IoU@0.03 |
|---|---:|---:|---:|---:|---|
| pool | 256/256 | 0.02507 | 0.03322 | 0.78617 | `0.06244 / 0.09688 / 0.78598` |
| pool | 512/512 (`max_len=4096`) | 0.02845 | 0.03839 | 0.77282 | - |
| grid | 256/256 | 0.03329 | 0.04153 | 0.36364 | `0.10837 / 0.13583 / 0.13222` |
| grid | 512/512 (`max_len=4096`) | 0.04011 | 0.05154 | 0.34259 | `0.08690 / 0.10867 / 0.18620` |

UCPR smoke (`mesh->udfgrid`, `max_files=200`, `pooling=mean_a`, `n_point=n_ray=512`, `max_len=4096`):

- `R@1=0.010`, `R@5=0.050`, `R@10=0.080`, `mAP=0.04454`

Pretrain schedule smoke:

- run completed with dynamic size update (`[schedule] epoch 1: n_point=96, n_ray=64`)
- artifacts: `runs/_tmp_scale_sched_smoke_fresh/{ckpt_ep000.pt,ckpt_ep001.pt,last.pt}`

### Readout (smoke)

- Progress-8 merge is now active in the main repo for scaling hooks (`max_len`, schedule, pos-emb resize).
- No regression observed in retrieval tie-aware path (tie-aware args and sanity flag preserved).
- On this small subset, naive inference-time scaling (`256 -> 512`) did not improve CPAC; retraining with scaling curriculum is required for fair assessment.
- Grid remains substantially above NN-copy in these smoke settings, but is still far below pool in absolute IoU.

## A-Pilot: Query Mix / Grid Sampler / Target Transform (Feb 16, 2026)

This cycle adds minimal A-style controls to `completion_cpac_udf.py`:

- `--query_source {pool,grid,hybrid}`
- `--query_pool_frac` (used when `hybrid`)
- `--grid_sample_mode {uniform,near_surface,stratified}`
- `--grid_near_tau`, `--grid_near_frac`
- `--target_transform {none,trunc,log1p}`
- `--target_trunc_max`, `--target_log_scale`
- `--report_near_tau` (`near@tau_report` metrics block)

Wrapper update:

- `scripts/analysis/nepa3d_cpac_udf.sh` now forwards all options above.

### Commands used (smoke, `max_shapes=40`)

```bash
# pool control
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_pool_control_a6_smoke.json

# hybrid (pool 50% + grid 50%), uniform grid sampling
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source hybrid --query_pool_frac 0.5 \
  --grid_sample_mode uniform --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_hybrid50_uniform_a6_smoke.json

# hybrid + near-surface grid sampling
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source hybrid --query_pool_frac 0.5 \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_hybrid50_near08_notrans_a6_smoke.json

# hybrid + near-surface + trunc target transform
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source hybrid --query_pool_frac 0.5 \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --target_transform trunc --target_trunc_max 0.1 \
  --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_hybrid50_near08_trunc01_a6_smoke.json

# grid-only comparison: uniform vs near-surface
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid --grid_sample_mode uniform \
  --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_grid_uniform_a6_smoke.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 800 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid --grid_sample_mode near_surface \
  --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --baseline nn_copy \
  --max_shapes 40 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_40_grid_near08_a6_smoke.json
```

### Results (smoke)

| Setting | Probe MAE / RMSE / IoU@0.03 | Near-only (`y<=0.05`) MAE / RMSE / IoU@0.03 | NN-copy MAE / RMSE / IoU@0.03 |
|---|---|---|---|
| pool control | `0.02463 / 0.03280 / 0.79701` | `0.01746 / 0.02292 / 0.79807` | `0.06199 / 0.09604 / 0.80096` |
| hybrid 50/50 + uniform | `0.02930 / 0.03728 / 0.71381` | `0.02172 / 0.02840 / 0.71540` | `0.08456 / 0.11662 / 0.70852` |
| hybrid 50/50 + near-surface | `0.02337 / 0.03076 / 0.66440` | `0.01880 / 0.02426 / 0.66492` | `0.05048 / 0.08256 / 0.58683` |
| hybrid + near-surface + trunc(0.1) | `0.16416 / 0.30722 / 0.76923` | `0.01199 / 0.01497 / 0.77024` | `0.05048 / 0.08256 / 0.58683` |
| grid + uniform | `0.03299 / 0.04098 / 0.41451` | `0.03587 / 0.04522 / 0.43011` | `0.10856 / 0.13542 / 0.13086` |
| grid + near-surface | `0.02168 / 0.02844 / 0.53708` | `0.01880 / 0.02384 / 0.53729` | `0.03885 / 0.06578 / 0.42571` |

### Readout (A-pilot)

- Grid query improves substantially with near-surface sampling (`IoU 0.4145 -> 0.5371` in this smoke).
- Hybrid sampling lowers absolute IoU vs pool-only, but clearly exceeds NN-copy under the same mixed query distribution.
- Trunc transform is useful for near-surface optimization (`near MAE` strongly improved), but raw MAE is no longer directly comparable to non-trunc settings.
- Next step should be full-scale rerun (`max_shapes=800`, `head_train_max_shapes=4000`, eval-seed sweep) before drawing main-table conclusions.

### Full matrix (`max_shapes=800`, `head_train_max_shapes=4000`, `eval_seed=0,1,2`)

Runner:

- `scripts/analysis/run_cpac_a_pilot_full.sh`

Executed:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_SEEDS="0 1 2" \
MAX_SHAPES=800 HEAD_TRAIN_MAX_SHAPES=4000 \
bash scripts/analysis/run_cpac_a_pilot_full.sh
```

Result JSONs:

- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_pool_uniform_seed{0,1,2}.json`
- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_grid_uniform_seed{0,1,2}.json`
- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_grid_near08_seed{0,1,2}.json`
- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_hybrid50_uniform_seed{0,1,2}.json`
- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_hybrid50_near08_seed{0,1,2}.json`
- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_hybrid50_near08_trunc01_seed{0,1,2}.json`
- aggregated summary: `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_a_pilot_seed012_summary.json`

Mean ± std over `eval_seed=0,1,2`:

| Setting | Probe MAE | Probe RMSE | Probe IoU@0.03 | Near-MAE (`y<=0.05`) | Near-IoU@0.03 | NN-copy MAE/RMSE/IoU@0.03 |
|---|---:|---:|---:|---:|---:|---|
| pool uniform | 0.02653 ± 0.00001 | 0.03528 ± 0.00009 | 0.72218 ± 0.00151 | 0.01946 ± 0.00003 | 0.72398 ± 0.00150 | `0.06149 ± 0.00001 / 0.09454 ± 0.00014 / 0.70786 ± 0.00075` |
| grid uniform | 0.03563 ± 0.00011 | 0.04477 ± 0.00018 | 0.29243 ± 0.00247 | 0.03666 ± 0.00027 | 0.30348 ± 0.00328 | `0.10416 ± 0.00013 / 0.13173 ± 0.00018 / 0.13060 ± 0.00069` |
| grid near08 | 0.02307 ± 0.00008 | 0.03058 ± 0.00010 | 0.46286 ± 0.00027 | 0.02015 ± 0.00007 | 0.46328 ± 0.00030 | `0.03867 ± 0.00006 / 0.06566 ± 0.00012 / 0.37682 ± 0.00061` |
| hybrid50 uniform | 0.03162 ± 0.00006 | 0.04051 ± 0.00013 | 0.62275 ± 0.00139 | 0.02416 ± 0.00007 | 0.62579 ± 0.00143 | `0.08282 ± 0.00009 / 0.11464 ± 0.00015 / 0.60840 ± 0.00047` |
| hybrid50 near08 | 0.02532 ± 0.00001 | 0.03357 ± 0.00006 | 0.59457 ± 0.00155 | 0.02068 ± 0.00002 | 0.59556 ± 0.00162 | `0.05025 ± 0.00008 / 0.08165 ± 0.00024 / 0.52811 ± 0.00087` |
| hybrid50 near08 + trunc0.1 | 0.14299 ± 0.00031 | 0.27323 ± 0.00057 | 0.70613 ± 0.00073 | 0.01344 ± 0.00001 | 0.70780 ± 0.00082 | `0.05025 ± 0.00008 / 0.08165 ± 0.00024 / 0.52811 ± 0.00087` |

Readout (full matrix):

- `grid` bottleneck is reduced by near-surface sampling (`IoU 0.292 -> 0.463`) while keeping a stable gain over NN-copy.
- `pool` remains strongest on this protocol (`IoU ~0.722`), consistent with earlier runs.
- `hybrid` improves over NN-copy but does not yet exceed pool-only; query distribution design still needs tuning.
- `trunc` strongly improves near-surface fit and IoU, but raw MAE/RMSE become non-comparable; report trunc rows as auxiliary.

## B-1 Pilot: Lipschitz-Regularized Probe (Feb 16, 2026)

This cycle adds an optional Lipschitz penalty to the CPAC ridge probe fitting path in:

- `nepa3d/analysis/completion_cpac_udf.py`

Added args:

- `--ridge_lipschitz_lambda`
- `--ridge_lipschitz_pairs`
- `--ridge_lipschitz_steps`
- `--ridge_lipschitz_lr`
- `--ridge_lipschitz_batch`
- `--ridge_lipschitz_max_points`
- `--ridge_lipschitz_seed`

Wrapper forwarding added in:

- `scripts/analysis/nepa3d_cpac_udf.sh` (`RIDGE_LIPSCHITZ_*`)

Implementation note (stability fix in this cycle):

- The Lipschitz refinement now uses a ridge solution computed on the same capped training set.
- If sampled violations are already zero at init, refinement is skipped (`zero_violation_at_init`) to avoid unnecessary drift.

### Commands used

Quick lambda sweep (`max_shapes=200`, `grid_near08`):

```bash
for L in 0 1e-4 3e-4 1e-3; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid \
    --head_train_max_shapes 1000 \
    --n_context 256 --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid \
    --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --max_shapes 200 --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --ridge_lipschitz_lambda ${L} --ridge_lipschitz_pairs 1024 \
    --ridge_lipschitz_steps 80 --ridge_lipschitz_lr 1e-2 \
    --ridge_lipschitz_batch 4096 --ridge_lipschitz_max_points 50000 \
    --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_200_grid_near08_lip${L}_v2.json

done
```

High-pair check (`max_shapes=200`):

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 200 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --ridge_lipschitz_lambda 1e-3 --ridge_lipschitz_pairs 16384 \
  --ridge_lipschitz_steps 120 --ridge_lipschitz_lr 5e-3 \
  --ridge_lipschitz_batch 4096 --ridge_lipschitz_max_points 100000 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_200_grid_near08_lip1e-3_pairs16k_v2.json
```

Full-size comparison (`max_shapes=800`, `htrain4k`, `eval_seed=0,1,2`):

```bash
for S in 0 1 2; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid \
    --head_train_max_shapes 4000 \
    --n_context 256 --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid \
    --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --max_shapes 800 --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed ${S} \
    --ridge_lipschitz_lambda 1e-3 --ridge_lipschitz_pairs 16384 \
    --ridge_lipschitz_steps 120 --ridge_lipschitz_lr 5e-3 \
    --ridge_lipschitz_batch 4096 --ridge_lipschitz_max_points 200000 \
    --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_grid_near08_lip1e-3_pairs16k_v2$( [ ${S} -eq 0 ] && echo '' || echo "_seed${S}" ).json

done
```

### Results

Quick sweep (`max_shapes=200`, seed0):

| Setting | MAE | RMSE | IoU@0.03 | Notes |
|---|---:|---:|---:|---|
| `lip=0` | 0.02223 | 0.02929 | 0.52529 | baseline |
| `lip=1e-4, pairs=1024` | 0.02222 | 0.02935 | 0.52519 | `zero_violation_at_init` |
| `lip=3e-4, pairs=1024` | 0.02222 | 0.02935 | 0.52519 | `zero_violation_at_init` |
| `lip=1e-3, pairs=1024` | 0.02222 | 0.02935 | 0.52519 | `zero_violation_at_init` |
| `lip=1e-3, pairs=16384` | 0.02235 | 0.02939 | 0.52106 | `init_lip=1.85e-5` |

Full-size (`max_shapes=800`, `eval_seed=0,1,2`) vs baseline `grid_near08`:

| Setting | MAE mean ± std | RMSE mean ± std | IoU@0.03 mean ± std | Near-IoU@0.03 mean ± std |
|---|---:|---:|---:|---:|
| baseline (`lip=0`) | 0.02307 ± 0.00008 | 0.03058 ± 0.00010 | 0.46286 ± 0.00027 | 0.46328 ± 0.00030 |
| Lipschitz (`lip=1e-3`, `pairs=16384`) | 0.02329 ± 0.00015 | 0.03087 ± 0.00020 | 0.45720 ± 0.00233 | 0.45765 ± 0.00230 |

Delta (`Lipschitz - baseline`):

- MAE: `+0.000216`
- RMSE: `+0.000284`
- IoU@0.03: `-0.005654`
- Near-IoU@0.03: `-0.005626`

Summary artifact:

- `results/cpac_nepa_qa_dualmask_s0_pc2udf_800_grid_near08_b1_lipschitz_summary.json`

### Readout (B-1 pilot)

- Under current probe-time formulation, B-1 does **not** improve CPAC grid-near performance.
- With moderate pair sampling, violations are often already near-zero at initialization; with large pair sampling, penalties activate but still slightly hurt IoU.
- Keep `ridge_lipschitz_lambda=0` as default for main runs.
- B-1 hooks remain in code for future use (e.g., pretrain-time regularization instead of probe-time only).

## B-2/B-3 Integration Smoke (Feb 16, 2026)

This cycle integrates B-2/B-3 in pretrain (fixed-size smoke only, before C-E):

- `nepa3d/train/pretrain.py`
  - B-2 aux: ray hit/depth supervision + pairwise depth-rank hinge
  - B-3 aux: near-surface point-distance supervision (`dist <= aux_b3_near_tau`)
  - new args: `--aux_b2_*`, `--aux_b3_*` (default OFF)
  - aux heads are checkpointed under `aux_heads` (separate from backbone `model`)
- `scripts/pretrain/nepa3d_pretrain.sh`
  - forward env vars for `AUX_B2_*`, `AUX_B3_*`

### Commands used

Pretrain smoke (`mix_num_samples=256`, `epochs=1`, `qa_tokens=1`):

```bash
# B-2 only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 1 --batch 8 --n_point 64 --n_ray 64 --max_len 320 \
  --num_workers 2 --save_every 1 --save_last 1 \
  --save_dir runs/_tmp_b2_smoke --seed 0 \
  --aux_b2_weight 0.2 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 64 --aux_b2_rank_margin 0.0

# B-3 only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 1 --batch 8 --n_point 64 --n_ray 64 --max_len 320 \
  --num_workers 2 --save_every 1 --save_last 1 \
  --save_dir runs/_tmp_b3_smoke --seed 0 \
  --aux_b3_weight 0.2 --aux_b3_near_tau 0.05

# B-2 + B-3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 1 --batch 8 --n_point 64 --n_ray 64 --max_len 320 \
  --num_workers 2 --save_every 1 --save_last 1 \
  --save_dir runs/_tmp_b23_smoke --seed 0 \
  --aux_b2_weight 0.2 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 64 --aux_b2_rank_margin 0.0 \
  --aux_b3_weight 0.2 --aux_b3_near_tau 0.05
```

CPAC quick check (`head_train_split=train_udf`, `head_train_max_shapes=1000`, `max_shapes=120`):

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_qa_dualmask_smoke/ckpt_ep000.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 1000 \
  --n_context 64 --n_query 64 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 120 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_qa_dualmask_smoke_pc2udf_120.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b2_smoke/ckpt_ep000.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 1000 \
  --n_context 64 --n_query 64 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 120 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2_smoke_pc2udf_120.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b3_smoke/ckpt_ep000.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 1000 \
  --n_context 64 --n_query 64 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 120 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b3_smoke_pc2udf_120.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b23_smoke/ckpt_ep000.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 1000 \
  --n_context 64 --n_query 64 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 120 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b23_smoke_pc2udf_120.json
```

### Results (quick check)

| Model | MAE | RMSE | IoU@0.03 | near-IoU@0.03 (`y<=0.05`) |
|---|---:|---:|---:|---:|
| QA dualmask smoke (reference) | 0.07258 | 0.09549 | 0.38448 | 0.38666 |
| B-2 only smoke | 0.07113 | 0.09349 | 0.39434 | 0.39648 |
| B-3 only smoke | 0.07151 | 0.09403 | 0.39203 | 0.39416 |
| B-2 + B-3 smoke | 0.07154 | 0.09391 | 0.39702 | 0.39917 |

### Readout (B-2/B-3 smoke)

- B-2/B-3 hooks are integrated and train/eval pipelines run end-to-end.
- Even in 1-epoch smoke, all three aux variants outperformed the reference smoke checkpoint on this tiny CPAC subset.
- This is only a sanity run (`mix_num_samples=256`, `max_shapes=120`); next step is fixed-256 full pilot before C.

## B-2/B-3 Fixed-256 Quick Pilot (Feb 16, 2026)

This cycle runs a quick fixed-size continuation from the main checkpoint:

- base ckpt: `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt`
- continuation: `epoch 50` only (`epochs=51`) with `mix_num_samples=8000`
- same base pretrain setup (`qa_tokens=1`, dual-mask on, `n_point=n_ray=256`)

### Commands used

Pretrain continuation:

```bash
# B-2 only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/_tmp_b2_ep050_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0

# B-3 only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/_tmp_b3_ep050_s0 --seed 0 \
  --aux_b3_weight 0.1 --aux_b3_near_tau 0.05

# B-2 + B-3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/_tmp_b23_ep050_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --aux_b3_weight 0.1 --aux_b3_near_tau 0.05
```

CPAC quick eval (`pool`, non-transductive):

```bash
# reference ep049
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 2000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 400 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_ep049_ref_pc2udf_400_htrain2k.json

# b2/b3/b23 ep050
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b2_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 2000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 400 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2_ep050_pc2udf_400_htrain2k.json
```

### Results

CPAC (`max_shapes=400`, `head_train_max_shapes=2000`, `eval_seed=0`):

| Model | MAE | RMSE | IoU@0.03 | Near-IoU@0.03 |
|---|---:|---:|---:|---:|
| ref `ep049` | 0.02501 | 0.03345 | 0.80078 | 0.80163 |
| B-2 `ep050` | 0.02322 | 0.03166 | 0.86838 | 0.86966 |
| B-3 `ep050` | 0.02507 | 0.03320 | 0.78490 | 0.78582 |
| B-2+B-3 `ep050` | 0.02357 | 0.03212 | 0.86066 | 0.86177 |

Delta vs ref `ep049`:

- B-2: `dMAE=-0.00179`, `dRMSE=-0.00179`, `dIoU=+0.06760`
- B-3: `dMAE=+0.00006`, `dRMSE=-0.00025`, `dIoU=-0.01588`
- B-2+B-3: `dMAE=-0.00144`, `dRMSE=-0.00133`, `dIoU=+0.05988`

### Readout (fixed-256 quick pilot)

- Under this quick continuation setup, B-2 gives the clearest gain.
- B-3 alone does not help in this setting; keep B-3 as tunable/secondary for now.
- B-2+B-3 is still better than ref, but below B-2-only.
- Next: run seed expansion (`s0/s1/s2`) for B-2-only before moving to C.

## B-2 Longer Confirm (Seed0, Feb 16, 2026)

Following the “no multi-seed for now” policy, we ran a slightly longer seed0-only B-2 continuation:

- base: `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt`
- run: `runs/eccv_upmix_nepa_qa_dualmask_b2quick_s0/ckpt_ep054.pt`
- setup: `mix_num_samples=20000`, `epochs=55` (resume from `ep049`), B-2 only (`aux_b2_weight=0.1`)

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 20000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 55 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2quick_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0
```

CPAC compare (`max_shapes=800`, `head_train_max_shapes=4000`, `query_source=pool`):

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_ep049_ref_pc2udf_800_normal_h_htrain4k_b2check.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2quick_s0/ckpt_ep054.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2_ep054_pc2udf_800_normal_h_htrain4k.json
```

UCPR hard-pair quick check (`max_files=1000`, `pooling=mean_a`):

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2quick_s0/ckpt_ep054.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_b2_ep054_mesh2udf_1k_indep_mean_a.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2quick_s0/ckpt_ep054.pt \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_b2_ep054_mesh2pc_1k_indep_mean_a.json
```

### Results

CPAC (pool, non-transductive):

| CKPT | MAE | RMSE | IoU@0.03 | Near-IoU@0.03 |
|---|---:|---:|---:|---:|
| `ep049` ref | 0.02653 | 0.03531 | 0.72281 | 0.72445 |
| `b2quick ep054` | 0.02505 | 0.03376 | 0.76543 | 0.76721 |

UCPR hard-pair (`b2quick ep054`, pooling=`mean_a`):

| Pair | R@1 | R@5 | R@10 | MRR (=single-positive mAP) |
|---|---:|---:|---:|---:|
| `mesh -> udfgrid` | 0.005 | 0.030 | 0.044 | 0.02532 |
| `mesh -> pointcloud_noray` | 0.011 | 0.033 | 0.064 | 0.03332 |

### Readout (B-2 longer confirm)

- CPAC (main track) improved further with longer B-2 continuation.
- Hard-pair UCPR dropped vs earlier dualmask baseline; treat this as a trade-off to watch during C.
- With current priority on completion, proceed to C prototype while monitoring UCPR side effects.

## C-0 Start: Teacher Refresh Hook (Feb 16, 2026)

Minimal C entry point implemented:

- `nepa3d/train/pretrain.py`
  - `--teacher_ckpt`
  - `--teacher_distill_weight`
  - optional answer-token distillation (`student z_hat` vs `teacher z_hat`) added to NEPA objective
- `scripts/pretrain/nepa3d_pretrain.sh`
  - added `TEACHER_CKPT`, `TEACHER_DISTILL_WEIGHT`

Smoke command (passed):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 1 --batch 8 --n_point 64 --n_ray 64 --max_len 320 \
  --num_workers 2 --save_every 1 --save_last 1 \
  --save_dir runs/_tmp_c0_teacher_smoke --seed 0 \
  --teacher_ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --teacher_distill_weight 0.1
```

Smoke artifact:

- `runs/_tmp_c0_teacher_smoke/ckpt_ep000.pt`

## C-0 Quick Compare (Seed0, Feb 16, 2026)

C-0 (teacher refresh distillation) was evaluated as a minimal C prototype:

- run: `runs/eccv_upmix_nepa_qa_dualmask_c0quick_s0/ckpt_ep050.pt`
- protocol: seed0 quick continuation (`mix_num_samples=8000`, `ep049 -> ep050`)

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_c0quick_s0 --seed 0 \
  --teacher_ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --teacher_distill_weight 0.1
```

CPAC quick compare:

```bash
# C-0 pool
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c0quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c0_ep050_pc2udf_800_normal_h_htrain4k.json

# C-0 grid_near08
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c0quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c0_ep050_pc2udf_800_grid_near08_h_htrain4k.json
```

### Results

Pool (`max_shapes=800`, `htrain4k`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02653 | 0.03531 | 0.72281 |
| B-2 `ep054` | 0.02505 | 0.03376 | 0.76543 |
| C-0 `ep050` | 0.02610 | 0.03441 | 0.71176 |

Grid near08 (`max_shapes=800`, `htrain4k`, `grid_near_frac=0.8`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02317 | 0.03072 | 0.46266 |
| B-2 `ep054` | 0.02184 | 0.02918 | 0.48853 |
| C-0 `ep050` | 0.02295 | 0.03026 | 0.45324 |

### Readout (C-0 quick compare)

- C-0 is weaker than B-2 on both pool and grid_near08.
- C-0 is also below the reference on IoU in this quick setup.
- Keep C-0 as a plumbing baseline only; prioritize B-2 line and move to stronger C (pseudo-parallel refresh) if needed.

## C-1/C-2 Quick Compare (Seed0, Feb 16, 2026)

Stronger C variants were run under the same quick protocol as C-0:

- base resume: `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt`
- quick continuation: `mix_num_samples=8000`, `ep049 -> ep050`, `seed=0`
- C-1: teacher refresh + answer-drop in student input
- C-2: cycle consistency across two answer-drop views

### Commands used

```bash
# C-1: teacher refresh + answer-drop
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_c1quick_s0 --seed 0 \
  --teacher_ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --teacher_distill_weight 0.1 --teacher_answer_drop_prob 0.4

# C-2: cycle consistency
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_c2quick_s0 --seed 0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4
```

CPAC compare (`pool` and `grid_near08`):

```bash
# C-1 pool
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c1quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c1_ep050_pc2udf_800_normal_h_htrain4k.json

# C-1 grid_near08
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c1quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c1_ep050_pc2udf_800_grid_near08_h_htrain4k.json

# C-2 pool
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c2quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c2_ep050_pc2udf_800_normal_h_htrain4k.json

# C-2 grid_near08
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c2quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_c2_ep050_pc2udf_800_grid_near08_h_htrain4k.json
```

UCPR hard-pair guardrail (C-2 only):

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c2quick_s0/ckpt_ep050.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_c2_ep050_mesh2udf_1k_indep_mean_a.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_c2quick_s0/ckpt_ep050.pt \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_c2_ep050_mesh2pc_1k_indep_mean_a.json
```

### Results

CPAC pool (`max_shapes=800`, `htrain4k`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02653 | 0.03531 | 0.72281 |
| B-2 `ep054` | 0.02505 | 0.03376 | 0.76543 |
| C-0 `ep050` | 0.02610 | 0.03441 | 0.71176 |
| C-1 `ep050` | 0.02391 | 0.03165 | 0.76084 |
| C-2 `ep050` | 0.02341 | 0.03134 | 0.78458 |

CPAC grid near08 (`max_shapes=800`, `htrain4k`, `grid_near_frac=0.8`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02317 | 0.03072 | 0.46266 |
| B-2 `ep054` | 0.02184 | 0.02918 | 0.48853 |
| C-0 `ep050` | 0.02295 | 0.03026 | 0.45324 |
| C-1 `ep050` | 0.02170 | 0.02846 | 0.46409 |
| C-2 `ep050` | 0.02102 | 0.02784 | 0.47582 |

UCPR hard-pair (`C-2 ep050`, pooling=`mean_a`):

| Pair | R@1 | R@5 | R@10 | MRR (=single-positive mAP) |
|---|---:|---:|---:|---:|
| `mesh -> udfgrid` | 0.011 | 0.021 | 0.029 | 0.02244 |
| `mesh -> pointcloud_noray` | 0.013 | 0.041 | 0.073 | 0.03596 |

### Readout (C-1/C-2 quick compare)

- C-1/C-2 はともに C-0 を上回り、C-0 の弱さは「C方向そのもの」の失敗ではなく実装強度不足だった可能性が高い。
- C-2 が現時点の seed0 最良（pool）で、`B-2 ep054` を上回った。
- C-2 は grid_near08 でも ref/C-0 を上回るが、`B-2 ep054` には未到達。
- UCPR side effect は mixed（`mesh->pc` は改善、`mesh->udf` mAP は B-2 より低下）なので、以降も hard-pair を副指標として併記する。

## B-2 + C-2 Joint Confirm (Seed0, Feb 16, 2026)

To resolve the B-2 vs C-2 trade-off, we ran a fixed-size joint setting:

- base resume: `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt`
- run: `runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt`
- protocol: `mix_num_samples=8000`, `ep049 -> ep050`, `seed=0`
- enabled losses:
  - B-2 ray supervision (`aux_b2_weight=0.1`, rank/hit/depth defaults)
  - C-2 cycle consistency (`cycle_weight=0.1`, `cycle_answer_drop_prob=0.4`)

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4
```

CPAC eval:

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_ep050_pc2udf_800_grid_near08_h_htrain4k.json
```

UCPR hard-pair guardrail:

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_b2c2_ep050_mesh2udf_1k_indep_mean_a.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_tmp_b2c2_ep050_mesh2pc_1k_indep_mean_a.json
```

### Results

CPAC pool (`max_shapes=800`, `htrain4k`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| B-2 `ep054` | 0.02505 | 0.03376 | 0.76543 |
| C-2 `ep050` | 0.02341 | 0.03134 | 0.78458 |
| B-2+C-2 `ep050` | 0.02310 | 0.03117 | 0.80672 |

CPAC grid near08 (`max_shapes=800`, `htrain4k`, `grid_near_frac=0.8`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| B-2 `ep054` | 0.02184 | 0.02918 | 0.48853 |
| C-2 `ep050` | 0.02102 | 0.02784 | 0.47582 |
| B-2+C-2 `ep050` | 0.02132 | 0.02792 | 0.47802 |

UCPR hard-pair (`pooling=mean_a`):

| Model | Pair | R@1 | MRR (=single-positive mAP) |
|---|---|---:|---:|
| B-2 `ep054` | `mesh -> udfgrid` | 0.005 | 0.02532 |
| C-2 `ep050` | `mesh -> udfgrid` | 0.011 | 0.02244 |
| B-2+C-2 `ep050` | `mesh -> udfgrid` | 0.009 | 0.02745 |
| B-2 `ep054` | `mesh -> pointcloud_noray` | 0.011 | 0.03332 |
| C-2 `ep050` | `mesh -> pointcloud_noray` | 0.013 | 0.03596 |
| B-2+C-2 `ep050` | `mesh -> pointcloud_noray` | 0.009 | 0.03760 |

### Readout (B-2 + C-2 joint confirm)

- Joint setting is now the strongest on CPAC pool (`IoU@0.03=0.80672`), exceeding both B-2 and C-2 alone.
- On grid_near08, joint improves over C-2 but remains below B-2.
- UCPR hard-pair stays mixed, but `mesh->udfgrid` mAP improves vs both B-2 and C-2; keep this as guardrail while moving to D/E.

## A Status + D/E Quick Compare (Seed0, Feb 16, 2026)

### A status

- A-1/A-2/A-3 matrix was already complete (`seed=0,1,2`), so this cycle did not re-run A.
- Existing artifacts count check: 18/18 JSON present for:
  - `pool_uniform`, `grid_uniform`, `grid_near08`
  - `hybrid50_uniform`, `hybrid50_near08`
  - `hybrid50_near08_trunc01`

### D/E implementation (single-factor, NEPA base loss unchanged)

Added to `nepa3d/train/pretrain.py`:

- D: hard-query mining aux term (top-error answer tokens)
  - `--d_hard_weight`
  - `--d_hard_top_frac`
  - `--d_hard_min_tokens`
- E: decoder-side point-distance head aux term
  - `--aux_e_weight`

Wrapper forwarding added in:

- `scripts/pretrain/nepa3d_pretrain.sh`

### Commands used

```bash
# D-only quick continuation
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_dquick_s0 --seed 0 \
  --d_hard_weight 0.1 --d_hard_top_frac 0.25 --d_hard_min_tokens 32

# E-only quick continuation
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_equick_s0 --seed 0 \
  --aux_e_weight 0.1
```

CPAC eval:

```bash
# D
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_dquick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_d_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_dquick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_d_ep050_pc2udf_800_grid_near08_h_htrain4k.json

# E
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_equick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_e_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_equick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_e_ep050_pc2udf_800_grid_near08_h_htrain4k.json
```

### Results

CPAC pool (`max_shapes=800`, `htrain4k`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02653 | 0.03531 | 0.72281 |
| B-2+C-2 `ep050` | 0.02310 | 0.03117 | 0.80672 |
| D-only `ep050` | 0.02517 | 0.03365 | 0.74835 |
| E-only `ep050` | 0.02497 | 0.03324 | 0.74602 |

CPAC grid near08 (`max_shapes=800`, `htrain4k`, `grid_near_frac=0.8`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| ref `ep049` | 0.02317 | 0.03072 | 0.46266 |
| B-2+C-2 `ep050` | 0.02132 | 0.02792 | 0.47802 |
| D-only `ep050` | 0.02204 | 0.02936 | 0.47069 |
| E-only `ep050` | 0.02257 | 0.02975 | 0.47045 |

### Readout (D/E quick compare)

- D/E single-factor quick pilots both improve over ref `ep049`.
- Both are below the current best (`B-2+C-2 ep050`) in pool and grid settings.
- Keep D/E as secondary candidates for now; proceed with 6 on top of `B-2+C-2`.

## 6 Scale Quick (B-2+C-2 Base, Seed0, Feb 16, 2026)

Quick scale pilot on top of `B-2+C-2`:

- base: `runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt`
- scaled run: `runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt`
- schedule: `n_point 256 -> 512` at epoch 51 (`n_ray=256` fixed)
- auto max_len expanded to 1538; resume-time `pos_emb` resize confirmed

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 52 --batch 64 --n_point 256 --n_ray 256 \
  --n_point_schedule '0:256,51:512' --n_ray_schedule '0:256' --max_len -1 \
  --num_workers 6 --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4
```

CPAC eval at scaled context (`n_context=512`, `n_query=256`):

```bash
# pre-scale ckpt (needs eval-time pos_emb resize)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_ep050_pc512q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt \
  --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_ep050_pc512q256_grid_near08_h_htrain4k.json

# scaled ckpt
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
  --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_ep051_pc512q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
  --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_ep051_pc512q256_grid_near08_h_htrain4k.json
```

### Results

CPAC pool (`n_context=512`, `n_query=256`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| B-2+C-2 `ep050` (pre-scale) | 0.02368 | 0.03281 | 0.80673 |
| B-2+C-2 `scale ep051` | 0.02181 | 0.02952 | 0.82563 |

CPAC grid near08 (`n_context=512`, `n_query=256`):

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| B-2+C-2 `ep050` (pre-scale) | 0.02181 | 0.02929 | 0.46511 |
| B-2+C-2 `scale ep051` | 0.02042 | 0.02657 | 0.48901 |

### Readout (6 scale quick)

- Even in a short 1-stage scale pilot, increasing training `n_point` to 512 improved both pool and grid under scaled eval context.
- Grid near08 improved from 0.46511 to 0.48901 and reached/parity with earlier B-2 best levels.
- This supports moving to a longer scale run as the next completion-focused track.

## 6 Scale Longer Attempt (Seed0, Feb 16, 2026)

We ran a longer continuation from the quick-scaled checkpoint to include `n_point=1024`.

- resume base: `runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt`
- run: `runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0/ckpt_ep054.pt`
- schedule: `0:256,51:512,53:1024` (`n_ray` fixed at 256)
- note: `pos_emb` was resized `1538 -> 2562` at resume time (optimizer resume skipped by design)

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 12000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 55 --batch 16 --n_point 256 --n_ray 256 \
  --n_point_schedule '0:256,51:512,53:1024' --n_ray_schedule '0:256' --max_len -1 \
  --num_workers 6 --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4
```

CPAC eval:

```bash
# long-scale ckpt, context=512
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0/ckpt_ep054.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_long_ep054_pc512q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0/ckpt_ep054.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_long_ep054_pc512q256_grid_near08_h_htrain4k.json

# long-scale ckpt, context=1024
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0/ckpt_ep054.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 1024 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_long_ep054_pc1024q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_long_s0/ckpt_ep054.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 1024 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_long_ep054_pc1024q256_grid_near08_h_htrain4k.json
```

### Results

Compared to quick-scale best (`scale ep051`):

`n_context=512, n_query=256`

| Model | Query | MAE | RMSE | IoU@0.03 |
|---|---|---:|---:|---:|
| `scale ep051` | pool | 0.02181 | 0.02952 | 0.82563 |
| `scale long ep054` | pool | 0.02895 | 0.03833 | 0.67921 |
| `scale ep051` | grid_near08 | 0.02042 | 0.02657 | 0.48901 |
| `scale long ep054` | grid_near08 | 0.02636 | 0.03432 | 0.36344 |

`n_context=1024, n_query=256` (long-scale ckpt)

| Query | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| pool | 0.02707 | 0.03522 | 0.70228 |
| grid_near08 | 0.02550 | 0.03257 | 0.37814 |

### Readout (6 scale longer attempt)

- This longer attempt degraded substantially vs the quick-scale checkpoint.
- Degradation appears on both pool and grid, and persists at `n_context=1024`.
- For now, keep `scale ep051` as the best scale candidate and treat this long attempt as unstable.

## 6 Scale Stability-Adjusted Retry (Seed0, Feb 16, 2026)

After the long attempt regression, we ran a stability-adjusted retry from `scalequick(ep051)`.

- resume base: `runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt`
- run: `runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0/ckpt_ep053.pt`
- effective train args: `epochs=54`, `batch=16`, `lr=1e-4`, `mix_num_samples=8000`
- schedule: `n_point 0:256,51:512,53:1024` (`n_ray=256` fixed)
- resume note: `pos_emb 1538 -> 2562`, optimizer resume skipped by design

### Commands used

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 54 --batch 16 --lr 1e-4 \
  --n_point 256 --n_ray 256 \
  --n_point_schedule '0:256,51:512,53:1024' --n_ray_schedule '0:256' \
  --max_len 2562 \
  --num_workers 6 --save_every 1 --save_last 1 \
  --resume runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4
```

CPAC eval:

```bash
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0/ckpt_ep053.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_stab_ep053_pc512q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0/ckpt_ep053.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_stab_ep053_pc512q256_grid_near08_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0/ckpt_ep053.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 1024 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_stab_ep053_pc1024q256_pool_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_stab_s0/ckpt_ep053.pt --max_len 2562 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 1024 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2c2_scale_stab_ep053_pc1024q256_grid_near08_h_htrain4k.json
```

### Results

`n_context=512, n_query=256`:

| Model | Query | MAE | RMSE | IoU@0.03 |
|---|---|---:|---:|---:|
| `scale quick ep051` | pool | 0.02181 | 0.02952 | 0.82563 |
| `scale long ep054` | pool | 0.02895 | 0.03833 | 0.67921 |
| `scale stab ep053` | pool | 0.02788 | 0.03550 | 0.65371 |
| `scale quick ep051` | grid_near08 | 0.02042 | 0.02657 | 0.48901 |
| `scale long ep054` | grid_near08 | 0.02636 | 0.03432 | 0.36344 |
| `scale stab ep053` | grid_near08 | 0.02527 | 0.03196 | 0.39260 |

`n_context=1024, n_query=256`:

| Model | Query | MAE | RMSE | IoU@0.03 |
|---|---|---:|---:|---:|
| `scale long ep054` | pool | 0.02707 | 0.03522 | 0.70228 |
| `scale stab ep053` | pool | 0.02444 | 0.03118 | 0.74580 |
| `scale long ep054` | grid_near08 | 0.02550 | 0.03257 | 0.37814 |
| `scale stab ep053` | grid_near08 | 0.02265 | 0.02857 | 0.46703 |

### Readout (6 scale stability retry)

- Stability retry reduced error metrics vs `scale long ep054` and recovered `grid_near08` at both contexts.
- At `n_context=512/pool`, IoU did not recover (`0.65371`, below both `ep051` and `ep054`), so recovery is incomplete.
- `scale quick ep051` remains the strongest single-seed checkpoint overall for promotion.

## B-3 Full-Protocol Fill (Seed0, Feb 17, 2026)

To close the remaining gap in the A-E+6 set, we evaluated B-3 and B-2+B-3 under the same full CPAC protocol used in later cycles:

- protocol: `max_shapes=800`, `head_train_max_shapes=4000`, `head_train_split=train_udf`
- context/query: `n_context=256`, `n_query=256`
- readout: `pool` and `grid_near08`
- seed: `eval_seed=0`

### Commands used

```bash
# B-2 ep050 (fair-length reference for B-3 quick line)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b2_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b2_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b2_ep050_pc2udf_800_grid_near08_h_htrain4k.json

# B-3 ep050
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b3_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b3_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b3_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b3_ep050_pc2udf_800_grid_near08_h_htrain4k.json

# B-2 + B-3 ep050
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b23_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b23_ep050_pc2udf_800_normal_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_b23_ep050_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_tmp_b23_ep050_pc2udf_800_grid_near08_h_htrain4k.json
```

### Results

CPAC full protocol (`n_context=256`, `n_query=256`, `max_shapes=800`, `htrain4k`):

| Model | Query | MAE | RMSE | IoU@0.03 |
|---|---|---:|---:|---:|
| ref `ep049` | pool | 0.02653 | 0.03531 | 0.72281 |
| B-2 `ep050` | pool | 0.02408 | 0.03271 | 0.79652 |
| B-3 `ep050` | pool | 0.02655 | 0.03494 | 0.70627 |
| B-2+B-3 `ep050` | pool | 0.02444 | 0.03321 | 0.79017 |
| ref `ep049` | grid_near08 | 0.02317 | 0.03072 | 0.46266 |
| B-2 `ep050` | grid_near08 | 0.02178 | 0.02887 | 0.47760 |
| B-3 `ep050` | grid_near08 | 0.02359 | 0.03093 | 0.44383 |
| B-2+B-3 `ep050` | grid_near08 | 0.02248 | 0.02976 | 0.46994 |

### Readout (B-3 full-protocol fill)

- B-3 alone underperforms both ref and B-2 on this full protocol.
- B-2+B-3 remains better than ref but does not beat B-2-only.
- With this fill, A-E+6 now all have at least one full-protocol result set in the active log.

## A-1 Octree/Coarse-to-Fine Grid Query (Feb 17, 2026)

This cycle implements A-1 in `completion_cpac_udf.py` and runs it under the same full CPAC protocol.

Added flags:

- `--grid_sample_mode coarse_to_fine`
- `--grid_res_schedule` (e.g., `16,32,64`)
- `--grid_c2f_expand`
- `--grid_c2f_stage_weights`

### Commands used

```bash
for MODE in uniform near_surface coarse_to_fine; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
    --n_context 512 --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid \
    --grid_sample_mode ${MODE} \
    --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --grid_res_schedule 16,32,64 --grid_c2f_expand 1 \
    --max_shapes 800 --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --out_json results/cpac_nepa_qa_dualmask_b2c2scalequick_ep051_pc2udf_800_grid_${MODE}_h_htrain4k_seed0.json
done
```

### Results

CPAC full protocol (`max_shapes=800`, `htrain4k`, `eval_seed=0`, `n_context=512`, `n_query=256`):

| grid_sample_mode | MAE | RMSE | IoU@0.03 | Near-IoU@0.03 (`y<=0.05`) |
|---|---:|---:|---:|---:|
| `uniform` | 0.03079 | 0.03889 | 0.28000 | 0.28817 |
| `near_surface` | 0.02042 | 0.02657 | 0.48901 | 0.48943 |
| `coarse_to_fine (16->32->64)` | 0.02318 | 0.02950 | 0.48444 | 0.49127 |

### Readout (A-1)

- A-1 is now **implemented and running** in this branch.
- `coarse_to_fine` gives a large gain over uniform sampling on grid completion.
- On this checkpoint/protocol, tuned `near_surface` remains slightly better on global IoU, while `coarse_to_fine` is comparable on near-surface IoU.
