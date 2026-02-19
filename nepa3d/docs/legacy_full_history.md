# NEPA-3D: Query-Interface Experiments (ShapeNet/ScanObjectNN M1)

This document tracks what was implemented in `nepa3d/`, what was executed, and current results.

## 0) Current active track (M1)

Primary track is now:

- pretrain: ShapeNet mesh + ShapeNet UDF + ScanObjectNN pointcloud_noray (mixed)
- downstream: ScanObjectNN full/few-shot (`K=0,1,5,10,20`)
- methods: `scratch`, `shapenet_nepa`, `shapenet_mesh_udf_nepa`, `shapenet_mix_nepa`, `shapenet_mix_mae`

Current launch flow in this repo:

```bash
# 1) M1 pretrains (2 GPUs, queued 3 jobs)
bash scripts/pretrain/launch_shapenet_m1_pretrains_local.sh

# 2) (optional) auto-launch M1 ScanObjectNN table after pretrain succeeds
bash scripts/finetune/launch_scanobjectnn_m1_after_pretrain.sh

# 3) or launch table manually
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

Resume behavior for pretrain:
- `pretrain.py` now supports `--resume <ckpt>` and `--auto_resume 1` (default).
- current launchers pass `--resume <save_dir>/last.pt`, so interrupted jobs resume automatically from saved epoch when relaunched.

Current log locations:

- pretrain: `logs/pretrain/m1/`
- finetune table: `logs/finetune/scan_m1_table/`
- auto-chain watcher: `logs/finetune/m1_after_pretrain/`
- status/cleanup helpers: `scripts/logs/show_pipeline_status.sh`, `scripts/logs/cleanup_stale_pids.sh`

Legacy ModelNet40-era experiments are kept for reference but are no longer the primary experimental path.

## 1) Research target

The goal is to unify the *interface* (`query -> answer` tokens), not to force a single 3D representation.

- Backends: `mesh`, `pointcloud`, `pointcloud_meshray` (legacy v0), `voxel`.
- Shared model: `QueryNepa` with NEPA next-embedding objective.
- Shared tokenizer: fixed token schema + explicit ordering.

If transfer works across backend switches, the interface abstraction is useful.

## 2) Implemented changes in this repo

### Data and preprocessing

- Added ModelNet40 download script: `nepa3d/data/download_modelnet40.sh`.
- Added preprocessing cache builder: `nepa3d/data/preprocess_modelnet40.py`.
- Added multiprocessing and resume-friendly behavior.
- Added pointcloud ray rendering with voxel occupancy + 3D DDA (`v1`):
  - additional arrays in cache: `ray_hit_pc_pool`, `ray_t_pc_pool`, `ray_n_pc_pool`.
  - preprocess args: `--pc_grid`, `--pc_dilate`, `--pc_max_steps`, `--no_pc_rays`.

### Backends and tokenizer

- Backend APIs under `nepa3d/backends/`.
- `PointCloudBackend` now prefers pointcloud-derived ray pools (v1) and falls back to mesh-ray pools if not available.
- Added `pointcloud_meshray` backend for strict v0 compatibility.
- Added `voxel` backend (occupancy grid from cached points; voxel point/ray answers).
- Tokenizer update to support deterministic sampling via injected RNG.

### Training / evaluation

- Pretrain entrypoint: `nepa3d/train/pretrain.py`.
- Finetune entrypoint: `nepa3d/train/finetune_cls.py`.
- Added pretrain objective switch: `--objective {nepa,mae}`.
- Added deterministic evaluation (`--eval_seed`) and Monte Carlo eval (`--mc_eval_k`).
- Added seed plumbing in PBS wrappers:
  - `scripts/pretrain/nepa3d_pretrain.sh`
  - `scripts/finetune/nepa3d_finetune.sh`
- Added from-scratch baseline wrapper:
  - `scripts/finetune/nepa3d_finetune_scratch.sh`

## 3) Model and objective

### Token schema

Each token has `(feature, type_id)`.

- feature dim `F=15`: `x(3), o(3), d(3), t(1), dist(1), hit(1), n(3)`.
- type ids: `BOS`, `POINT`, `RAY`.

### Ordering

- point tokens: Morton sort in 3D.
- ray tokens: direction-based sorting.

### Pretrain

```
z = token_mlp(feat) + type_emb + pos_emb
h = causal_transformer(z)
z_hat = pred_head(h)
loss = mean(1 - cosine(z_hat_t, stopgrad(z_{t+1})))
```

Alternative baseline objective (`--objective mae`):
- Random token masking on input features (`--mask_ratio`).
- Reconstruct masked token embeddings with MSE in prediction space.

### Finetune

- backbone: `QueryNepa`.
- head: linear classifier on last hidden state.
- loss: cross-entropy over 40 classes.

## 4) Experimental setup used in runs

### Common training defaults

- `n_point=256`, `n_ray=256`
- `d_model=384`, `layers=8`, `heads=6`
- pretrain: `epochs=50`, `lr=3e-4`
- finetune: `epochs=100`, `lr=1e-4`

### ECCV v2 preprocessing profile (legacy reference)

This profile was used for the earlier ModelNet40-centered mixed-pretrain plan.
Current main path is ShapeNet-based M1 (see section `0)` above).

- ModelNet40 cache root: `data/modelnet40_cache_v2`
- ScanObjectNN cache root: `data/scanobjectnn_cache_v2`

ModelNet40 v2 parameters:
- `PC_POINTS=2048`
- `PT_POOL=20000`
- `RAY_POOL=8000`
- `N_VIEWS=20`
- `RAYS_PER_VIEW=400`
- `PC_GRID=64`
- `PC_DILATE=1`
- `DF_GRID=64`
- `DF_DILATE=1`
- `PT_SURFACE_RATIO=0.5`
- `PT_SURFACE_SIGMA=0.02`
- `PT_QUERY_CHUNK=2048`
- `RAY_QUERY_CHUNK=2048`

ScanObjectNN v2 parameters:
- `PT_POOL=4000`
- `RAY_POOL=256`
- `PT_SURFACE_RATIO=0.5`
- `PT_SURFACE_SIGMA=0.02`

v2 cache completion status (current local run):
- ModelNet40: `train=9843`, `test=2468` (`data/modelnet40_cache_v2`)
- ScanObjectNN: `train=59542`, `test=21889` (`data/scanobjectnn_cache_v2`)

### Local stability profile (important)

When running preprocessing from a desktop session, memory growth in long-lived `trimesh` workers can trigger OOM.
To avoid this, `preprocess_modelnet40.py` now supports worker recycling:

- `--max_tasks_per_child N` (0 disables recycling)
- `--pt_query_chunk` / `--ray_query_chunk` to cap per-call memory spikes
- `--pt_dist_mode {mesh,kdtree}`:
  - `mesh`: exact `trimesh.proximity.closest_point` (slow)
  - `kdtree`: nearest sampled surface point approximation (fast)
- UDF generation now fails fast if SciPy (`distance_transform_edt`) is unavailable.

Recommended local values on this machine (24 CPU threads, 125 GiB RAM):
- standard: `workers=4`, `chunk_size=1`, `max_tasks_per_child=2`
- safe fallback (if desktop apps are open or OOM appears): `workers=2`, `chunk_size=1`, `max_tasks_per_child=1`
- run `train` then `test` sequentially (not two concurrent ModelNet jobs)

Observed bottlenecks on this machine:
- `closest_point` (exact pt distance) dominates wall time.
- ray-mesh intersection is fast only when Embree backend is available (`embreex` installed); otherwise `ray_triangle` is extremely slow.
- verify Embree availability with:
  - `python -c "from trimesh.ray.ray_pyembree import RayMeshIntersector; print('ok')"`
- quick UDF sanity check:
  - sample `*.npz` and confirm `udf_grid` is not `(1,1,1)` and `pt_dist_udf_pool` has low zero-ratio.

### Reproducible local commands for v2 caches

ModelNet40 v2:

```bash
python -u nepa3d/data/preprocess_modelnet40.py \
  --modelnet_root data/ModelNet40 \
  --out_root data/modelnet40_cache_v2 \
  --split train \
  --pc_points 2048 --pt_pool 20000 --ray_pool 8000 \
  --n_views 20 --rays_per_view 400 \
  --pc_grid 64 --pc_dilate 1 \
  --df_grid 64 --df_dilate 1 \
  --pt_surface_ratio 0.5 --pt_surface_sigma 0.02 \
  --pt_query_chunk 2048 --ray_query_chunk 2048 \
  --pt_dist_mode kdtree --dist_ref_points 8192 \
  --workers 4 --chunk_size 1 --max_tasks_per_child 2

python -u nepa3d/data/preprocess_modelnet40.py \
  --modelnet_root data/ModelNet40 \
  --out_root data/modelnet40_cache_v2 \
  --split test \
  --pc_points 2048 --pt_pool 20000 --ray_pool 8000 \
  --n_views 20 --rays_per_view 400 \
  --pc_grid 64 --pc_dilate 1 \
  --df_grid 64 --df_dilate 1 \
  --pt_surface_ratio 0.5 --pt_surface_sigma 0.02 \
  --pt_query_chunk 2048 --ray_query_chunk 2048 \
  --pt_dist_mode kdtree --dist_ref_points 8192 \
  --workers 4 --chunk_size 1 --max_tasks_per_child 2
```

ScanObjectNN v2:

```bash
python -u nepa3d/data/preprocess_scanobjectnn.py \
  --scan_root data/ScanObjectNN/h5_files \
  --out_root data/scanobjectnn_cache_v2 \
  --split all \
  --pt_pool 4000 --ray_pool 256 \
  --pt_surface_ratio 0.5 --pt_surface_sigma 0.02
```

ShapeNet (simple replacement pretrain corpus):

- expected mesh layout: `ShapeNetCore.v2/<synset>/<model>/models/model_normalized.obj`
- cache output: `data/shapenet_cache_v0/{train,test}/<synset>/*.npz`

```bash
SHAPENET_ROOT=data/ShapeNetCore.v2 \
OUT_ROOT=data/shapenet_cache_v0 \
SPLIT=all \
PT_DIST_MODE=kdtree \
bash scripts/preprocess/preprocess_shapenet.sh
```

Simple mesh-only pretrain on ShapeNet cache (NEPA + MAE in parallel):

```bash
CACHE_ROOT=data/shapenet_cache_v0 \
SAVE_EVERY=10 SAVE_LAST=1 \
bash scripts/pretrain/run_shapenet_simple_local.sh
```

Checkpoint pruning helper (keep every 10 epochs + last):

```bash
KEEP_EVERY=10 KEEP_LAST=1 \
bash scripts/pretrain/prune_pretrain_checkpoints.sh
```

### Reproducible local commands for mixed pretrain (ECCV v2, legacy)

Run both on 2 GPUs in parallel:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/pretrain_mixed_eccv.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective nepa --drop_ray_prob 0.3 \
  --batch 32 --epochs 50 --num_workers 6 --seed 0 \
  --save_dir runs/eccv_mix_nepa_s0

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/pretrain_mixed_eccv.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective mae --mask_ratio 0.4 \
  --batch 32 --epochs 50 --num_workers 6 --seed 0 \
  --save_dir runs/eccv_mix_mae_s0
```

Suggested log files:
- `logs/pretrain/eccv_mixed/mix_nepa_s0.log`
- `logs/pretrain/eccv_mixed/mix_mae_s0.log`

### ScanObjectNN main-table launcher (local 2-GPU, legacy)

Run full + few-shot (K=0/1/5/10/20) for 4 methods (`scratch`, `mesh_nepa`, `mix_nepa`, `mix_mae`) and seeds `0,1,2`:

```bash
bash scripts/finetune/run_scanobjectnn_main_table_local.sh
```

Background launcher:

```bash
bash scripts/finetune/launch_scanobjectnn_main_table_local.sh
```

This launches 60 jobs total (2 in parallel, one per GPU), with resume/skip by `last.pt`:
- logs: `logs/finetune/scan_main_table/jobs/*.log`
- per-GPU runner logs: `logs/finetune/scan_main_table/jobs/runner_gpu0.log`, `logs/finetune/scan_main_table/jobs/runner_gpu1.log`
- outputs: `runs/scan_<method>_k<K>_s<seed>/`
- speed knobs (env): `MC_EVAL_K_VAL=1` (fast), `MC_EVAL_K_TEST=4` (final test only)
- local recommendation on this machine (RTX PRO 6000 x2): `BATCH=96` for ScanObjectNN finetune

Recommended monitoring:

```bash
tail -f logs/finetune/scan_main_table/jobs/runner_gpu0.log
tail -f logs/finetune/scan_main_table/jobs/runner_gpu1.log
nvidia-smi
```

### ScanObjectNN fine-tune from ShapeNet pretrain (local 2-GPU)

Run full + few-shot (K=0/1/5/10/20) for 3 methods (`scratch`, `shapenet_nepa`, `shapenet_mae`) and seeds `0,1,2`:

```bash
bash scripts/finetune/run_scanobjectnn_shapenet_table_local.sh
```

Background launcher:

```bash
bash scripts/finetune/launch_scanobjectnn_shapenet_table_local.sh
```

Chain launcher (auto-start mixed/main-table after shapenet-table completes successfully):

```bash
bash scripts/finetune/launch_chain_shapenet_to_main.sh
```

This launches 45 jobs total (2 in parallel, one per GPU), with resume/skip by `last.pt`:
- logs: `logs/finetune/scan_shapenet_table/jobs/*.log`
- per-GPU runner logs: `logs/finetune/scan_shapenet_table/jobs/runner_gpu0.log`, `logs/finetune/scan_shapenet_table/jobs/runner_gpu1.log`
- outputs: `runs/scan_<method>_k<K>_s<seed>/`

### M1 pretrain set (ShapeNet-based mixed)

Pretrain jobs for M1:
- `shapenet_mesh_udf_nepa_s0` (mesh+UDF)
- `shapenet_mix_nepa_s0` (mesh+UDF+ScanObjectNN no-ray)
- `shapenet_mix_mae_s0` (same mixed corpus, MAE objective)

Run:

```bash
bash scripts/pretrain/launch_shapenet_m1_pretrains_local.sh
```

Logs:
- `logs/pretrain/m1/pipeline.log`
- `logs/pretrain/m1/*.log`

### M1 ScanObjectNN table (local 2-GPU)

Methods:
- `scratch`
- `shapenet_nepa` (mesh-only pretrain)
- `shapenet_mesh_udf_nepa`
- `shapenet_mix_nepa`
- `shapenet_mix_mae`

Run:

```bash
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

Logs:
- `logs/finetune/scan_m1_table/pipeline.log`
- `logs/finetune/scan_m1_table/jobs/*.log`

### M1 end-to-end chain

Wait for current `scan_shapenet_table` completion, then launch:
1) M1 pretrains
2) M1 ScanObjectNN table

```bash
bash scripts/finetune/launch_m1_pipeline_after_shapenet_table.sh
```

### v0 cache

- preprocess params: `pc_points=1024`, `pt_pool=2000`, `ray_pool=1000`, `n_views=10`, `rays_per_view=200`
- cache stats:
  - train: `9809 / 9843`
  - test: `2468 / 2468`
- note: pointcloud used mesh-ray fallback in this stage.

### v1 cache (interface-strict)

- includes pointcloud-derived ray DDA pools.
- cache stats:
  - train: `9781 / 9843`
  - test: `2468 / 2468`
- handling policy: failed meshes are excluded from training set because datasets enumerate existing `*.npz` only.

## 5) Results

## 5.0 M1 ScanObjectNN table status (current snapshot)

As of this snapshot, M1 fine-tune is not fully complete yet.

- expected jobs: `5 methods x 5 K x 3 seeds = 75`
- completed (`runs/scan_*_k*_s*/last.pt` exists): `29 / 75`
- completion by method:
  - `scratch`: `10 / 15`
  - `shapenet_nepa`: `10 / 15`
  - `shapenet_mesh_udf_nepa`: `3 / 15`
  - `shapenet_mix_nepa`: `3 / 15`
  - `shapenet_mix_mae`: `3 / 15`

Resume command:

```bash
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

The launcher skips jobs with existing `last.pt`, so it resumes from remaining jobs.

### M1 partial results (test_acc from completed jobs)

`n(seed)` is the number of completed seeds currently available for that `(method, K)`.

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 2 | 0.8221 +- 0.0038 |
| `scratch` | 1 | 2 | 0.1414 +- 0.0211 |
| `scratch` | 5 | 2 | 0.1456 +- 0.0033 |
| `scratch` | 10 | 2 | 0.1115 +- 0.0000 |
| `scratch` | 20 | 2 | 0.1115 +- 0.0000 |
| `shapenet_nepa` | 0 | 2 | 0.8075 +- 0.0028 |
| `shapenet_nepa` | 1 | 2 | 0.1630 +- 0.0328 |
| `shapenet_nepa` | 5 | 2 | 0.2283 +- 0.0117 |
| `shapenet_nepa` | 10 | 2 | 0.2618 +- 0.0155 |
| `shapenet_nepa` | 20 | 2 | 0.3247 +- 0.0042 |
| `shapenet_mesh_udf_nepa` | 0 | 1 | 0.8178 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 1 | 1 | 0.1345 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 10 | 1 | 0.2827 +- 0.0000 |
| `shapenet_mix_nepa` | 0 | 1 | 0.8285 +- 0.0000 |
| `shapenet_mix_nepa` | 5 | 1 | 0.2546 +- 0.0000 |
| `shapenet_mix_nepa` | 20 | 1 | 0.3793 +- 0.0000 |
| `shapenet_mix_mae` | 0 | 1 | 0.7856 +- 0.0000 |
| `shapenet_mix_mae` | 1 | 1 | 0.1399 +- 0.0000 |
| `shapenet_mix_mae` | 10 | 1 | 0.2662 +- 0.0000 |

## 5.1 v0 single-run summary

| Setting | Pretrain Backend | Finetune Backend | Final Acc (ep=99) | Best Acc (epoch) |
|---|---|---|---:|---:|
| Base | mesh | mesh | 0.8643 | 0.8643 (99) |
| Transfer A | mesh | pointcloud | 0.8468 | 0.8663 (73) |
| Transfer B | pointcloud | mesh | 0.8205 | 0.8278 (98) |

## 5.2 v1 single-run summary (`EVAL_SEED=0`, `MC_EVAL_K=4`)

| Setting | Pretrain Backend | Finetune Backend | Final Acc (ep=99) | Best Acc (epoch) |
|---|---|---|---:|---:|
| Base | mesh | mesh | 0.8602 | 0.8756 (79) |
| Transfer A | mesh | pointcloud | 0.8679 | 0.8740 (89) |
| Transfer B | pointcloud | mesh | 0.8578 | 0.8728 (81) |
| Base-PC | pointcloud | pointcloud | 0.8562 | 0.8740 (98) |

## 5.3 v1 multi-seed (seed `0,1,2`)

### Pretrained initialization

| Setting | Final mean +- std | Best mean +- std |
|---|---:|---:|
| mesh -> mesh | 0.8594 +- 0.0057 | 0.8752 +- 0.0104 |
| mesh -> pointcloud | 0.8499 +- 0.0115 | 0.8749 +- 0.0074 |
| pointcloud -> mesh | 0.8635 +- 0.0114 | 0.8755 +- 0.0043 |
| pointcloud -> pointcloud | 0.8563 +- 0.0101 | 0.8718 +- 0.0033 |

### From-scratch baseline

| Setting | Final mean +- std | Best mean +- std |
|---|---:|---:|
| scratch mesh | 0.8498 +- 0.0025 | 0.8671 +- 0.0042 |
| scratch pointcloud | 0.8549 +- 0.0072 | 0.8705 +- 0.0040 |

## 6) What can be claimed now

- Backend transfer is consistently viable under a shared token interface.
- Pretraining improves over scratch on best-epoch metrics in both mesh and pointcloud settings.
- v1 removes the main v0 criticism (pointcloud ray fallback) by using pointcloud-derived ray answers.

## 7) Reproduction commands (PBS)

### Preprocess (v1)

```bash
qsub -v SPLIT=train,OUT_ROOT=data/modelnet40_cache_v1,PC_GRID=64,PC_DILATE=1 \
  scripts/preprocess/preprocess_modelnet40.sh
qsub -v SPLIT=test,OUT_ROOT=data/modelnet40_cache_v1,PC_GRID=64,PC_DILATE=1 \
  scripts/preprocess/preprocess_modelnet40.sh
```

### Single-run (v1)

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,SAVE_DIR=runs/querynepa3d_meshpre_v1 \
  scripts/pretrain/nepa3d_pretrain.sh

qsub -W depend=afterok:<PRE_JOB_ID> \
  -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,CKPT=runs/querynepa3d_meshpre_v1/ckpt_ep049.pt,EVAL_SEED=0,MC_EVAL_K=4 \
  scripts/finetune/nepa3d_finetune.sh

qsub -W depend=afterok:<PRE_JOB_ID> \
  -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=pointcloud,CKPT=runs/querynepa3d_meshpre_v1/ckpt_ep049.pt,EVAL_SEED=0,MC_EVAL_K=4 \
  scripts/finetune/nepa3d_finetune.sh
```

### From-scratch baseline

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,SEED=0,EVAL_SEED=0,MC_EVAL_K=4,SAVE_DIR=runs/querynepa3d_scratch_mesh_s0 \
  scripts/finetune/nepa3d_finetune_scratch.sh
```

## 8) Remaining limitations

- Training set still has failed meshes (62 missing in v1 cache).
- Reporting should include both final and best values (final-only can understate performance variance).
- Linear-probe accuracy is still low and unstable across seeds; probe protocol may need tuning.
- ScanObjectNN currently has a single successful run; multi-seed and scratch/pretrain controls are still pending.

## 9) New experiment mode: missing-ray + modality dropout

To support partial query availability safely, the code now includes:

- token types `TYPE_MISSING_RAY` and `TYPE_EOS`
- missing-ray masking in NEPA loss (missing-ray targets are excluded)
- explicit EOS at sequence end (classification still uses last token robustly)

### New runtime options

- pretrain (`nepa3d/train/pretrain.py`)
  - `--drop_ray_prob` in `[0,1]`
  - `--force_missing_ray`
  - `--add_eos` (`1` or `0`)
- finetune (`nepa3d/train/finetune_cls.py`)
  - `--drop_ray_prob_train` in `[0,1]`
  - `--force_missing_ray`
  - `--add_eos` (`1`, `0`, `-1` for infer-from-ckpt)
- backend
  - added `pointcloud_noray` in addition to `mesh`, `pointcloud`, `pointcloud_meshray`

### Minimal PBS recipes for this study

#### A) mesh pretrain without dropout

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,DROP_RAY_PROB=0.0,ADD_EOS=1,SAVE_DIR=runs/querynepa3d_meshpre_missray_p0 \
  scripts/pretrain/nepa3d_pretrain.sh
```

#### B) mesh pretrain with modality dropout (`p=0.3`)

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,DROP_RAY_PROB=0.3,ADD_EOS=1,SAVE_DIR=runs/querynepa3d_meshpre_missray_p03 \
  scripts/pretrain/nepa3d_pretrain.sh
```

#### C) pointcloud finetune with no rays (0-filled via missing-ray path)

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=pointcloud,FORCE_MISSING_RAY=1,ADD_EOS=1,CKPT=runs/querynepa3d_meshpre_missray_p03/ckpt_ep049.pt,EVAL_SEED=0,MC_EVAL_K=4 \
  scripts/finetune/nepa3d_finetune.sh
```

#### D) point-only baseline (`n_ray=0`)

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,N_RAY=0,ADD_EOS=1,SAVE_DIR=runs/querynepa3d_meshpre_pointonly \
  scripts/pretrain/nepa3d_pretrain.sh
```

### 9.1 Ablation results (single seed)

All runs below use `EVAL_SEED=0`, `MC_EVAL_K=4` and finetune backend `pointcloud` with missing-ray setting where specified.

| Setting | Final Acc (ep=99) | Best Acc (epoch) | Log |
|---|---:|---:|---|
| Baseline: mesh pretrain -> pointcloud finetune (ray available) | 0.8679 | 0.8740 | `logs/ft_pointcloud.out` |
| A: `drop_ray_prob=0.0` pretrain -> pointcloud finetune (`FORCE_MISSING_RAY=1`) | 0.8505 | 0.8671 (54) | `logs/ab_ft_p0_to_pc_missing.out` |
| B: `drop_ray_prob=0.3` pretrain -> pointcloud finetune (`FORCE_MISSING_RAY=1`) | 0.8537 | 0.8655 (74) | `logs/ab_ft_p03_to_pc_missing.out` |
| D: point-only (`n_ray=0`) pretrain -> pointcloud finetune (`n_ray=0`) | 0.8517 | 0.8671 (71) | `logs/ab_ft_pointonly_pc.out` |

Observations:
- Missing-ray finetune is consistently below the ray-available baseline in this single-seed run.
- `drop_ray_prob=0.3` is close to `drop_ray_prob=0.0` (no clear gain yet).
- Point-only baseline is in the same range as missing-ray runs.

## 10) SHOULD / COULD implementations added

### SHOULD-1: third representation backend (`voxel`)

- new backend: `nepa3d/backends/voxel_backend.py`
- use with:

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=voxel,SAVE_DIR=runs/querynepa3d_voxelpre_v1 \
  scripts/pretrain/nepa3d_pretrain.sh
```

### SHOULD-2: DDA quality metrics + grid/dilate sweep

- analysis tool: `nepa3d/analysis/dda_metrics.py`
- single evaluation script: `scripts/analysis/nepa3d_dda_metrics.sh`
- sweep script: `scripts/analysis/nepa3d_dda_sweep.sh`

Examples:

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,SPLIT=test \
  scripts/analysis/nepa3d_dda_metrics.sh

qsub -v MODELNET_ROOT=data/ModelNet40,OUT_BASE=data/modelnet40_cache_sweep,SPLIT=test,PC_GRIDS=\"32 64 128\",PC_DILATES=\"0 1 2\" \
  scripts/analysis/nepa3d_dda_sweep.sh
```

Current DDA metric run on `test` (`logs/dda_test.out`):
- `hit_acc=0.9583`, `precision=0.8609`, `recall=0.9464`, `f1=0.9016`
- `depth_abs_mean=0.1008`, `depth_abs_p90=0.2349`, `depth_abs_p99=0.9866`
- CSV output: `results/dda_metrics_test.csv`

### SHOULD-3: objective baseline under same token interface

- pretrain supports `--objective mae` (masked embedding reconstruction)

```bash
qsub -v CACHE_ROOT=data/modelnet40_cache_v1,BACKEND=mesh,OBJECTIVE=mae,MASK_RATIO=0.4,SAVE_DIR=runs/querynepa3d_meshpre_mae_v1 \
  scripts/pretrain/nepa3d_pretrain.sh
```

Current MAE baseline result (`logs/mae_ft_mesh.out`):
- `best_val=0.8855` (ep 70), `test_acc=0.8606`

### COULD-1: real point-cloud dataset path (ScanObjectNN)

- download helper: `nepa3d/data/download_scanobjectnn.sh`
- preprocessing script: `nepa3d/data/preprocess_scanobjectnn.py`
- PBS wrapper: `scripts/preprocess/preprocess_scanobjectnn.sh`
- PBS download wrapper: `scripts/preprocess/download_scanobjectnn.sh`
- finetune wrapper: `scripts/finetune/nepa3d_finetune_scanobjectnn.sh`

Examples:

```bash
qsub -v OUT_DIR=data/ScanObjectNN \
  scripts/preprocess/download_scanobjectnn.sh

qsub -v SCAN_ROOT=data/ScanObjectNN/h5_files,OUT_ROOT=data/scanobjectnn_cache_v1,SPLIT=all \
  scripts/preprocess/preprocess_scanobjectnn.sh

qsub -v CACHE_ROOT=data/scanobjectnn_cache_v1,BACKEND=pointcloud_noray,CKPT=runs/querynepa3d_ab_meshpre_p0/ckpt_ep049.pt \
  scripts/finetune/nepa3d_finetune_scanobjectnn.sh
```

Status:
- First run failed at preprocess due missing `h5py` (`logs/scan_pp.err`).
- `h5py` installed into `.venv`.
- Preprocess rerun completed:
  - train cache: `59542` files
  - test cache: `21889` files
- First finetune rerun failed due ckpt type mismatch (`n_types=3` ckpt with `pointcloud_noray` / missing-ray tokens).
- Final rerun with `n_types=5` ckpt completed successfully (`logs/scan_ft3.out`):
  - `train_backend=pointcloud_noray`, `eval_backend=pointcloud_noray`
  - `num_train=53594`, `num_val=5948`, `num_test=21889`
  - `best_val=0.8906` (ep 98), `test_acc=0.8036`

Compatibility note:
- For `BACKEND=pointcloud_noray`, use a checkpoint trained with missing-ray token support (`n_types=5`), e.g.:
  - `runs/querynepa3d_ab_meshpre_p0/ckpt_ep049.pt`
  - `runs/querynepa3d_ab_meshpre_p03/ckpt_ep049.pt`

## 11) MUST completion (val-selected 3x3 multi-seed + linear probe)

All MUST jobs were run with:
- stratified `val_ratio=0.1`, `val_seed=0`
- best checkpoint selected by `val_acc`
- final `test_acc` evaluated once from best-val checkpoint
- seeds: `0,1,2`

### 11.1 3x3 transfer matrix (pretrained fine-tune)

Metric below is `test_acc` mean/std over seeds (`logs/ft_*_s*.out`).

| Pretrain -> Eval | mean +- std |
|---|---:|
| mesh -> mesh | 0.8601 +- 0.0025 |
| mesh -> pointcloud | 0.8576 +- 0.0025 |
| mesh -> pointcloud_meshray | 0.8638 +- 0.0085 |
| pointcloud -> mesh | 0.8572 +- 0.0090 |
| pointcloud -> pointcloud | 0.8593 +- 0.0059 |
| pointcloud -> pointcloud_meshray | 0.8576 +- 0.0088 |
| pointcloud_meshray -> mesh | 0.8628 +- 0.0111 |
| pointcloud_meshray -> pointcloud | 0.8521 +- 0.0161 |
| pointcloud_meshray -> pointcloud_meshray | 0.8549 +- 0.0124 |

Aggregate over all 27 runs:
- mean `test_acc=0.8584`, std `0.0101`

### 11.2 3x3 linear-probe matrix (`--freeze_backbone`)

Metric below is `test_acc` mean/std over seeds (`logs/lp_*_s*.out`).

| Probe train backend -> Eval backend | mean +- std |
|---|---:|
| mesh -> mesh | 0.1251 +- 0.0812 |
| mesh -> pointcloud | 0.1303 +- 0.0862 |
| mesh -> pointcloud_meshray | 0.1282 +- 0.0819 |
| pointcloud -> mesh | 0.0669 +- 0.0134 |
| pointcloud -> pointcloud | 0.0683 +- 0.0122 |
| pointcloud -> pointcloud_meshray | 0.0667 +- 0.0143 |
| pointcloud_meshray -> mesh | 0.0693 +- 0.0029 |
| pointcloud_meshray -> pointcloud | 0.0703 +- 0.0036 |
| pointcloud_meshray -> pointcloud_meshray | 0.0705 +- 0.0038 |

Aggregate over all 27 probe runs:
- mean `test_acc=0.0884`, std `0.0561`

### 11.3 Additional SHOULD results

- Voxel backend transfer (`logs/vox_ft_mesh.out`):
  - `best_val=0.9043` (ep 92), `test_acc=0.8728`
- MAE objective baseline (`logs/mae_ft_mesh.out`):
  - `best_val=0.8855` (ep 70), `test_acc=0.8606`

## 12) UCPR / CPAC loop (cache-reuse path)

This section tracks the new ECCV-oriented loop:

- implement `unpaired primitive` utilities
- run quick experiments
- update this section
- get feedback and iterate

### 12.1 Implementation added (non-destructive to existing core)

New files:

- split builder from existing ShapeNet cache:
  - `nepa3d/data/shapenet_unpaired_split.py`
- unpaired cache materializer (symlink/hardlink/copy):
  - `nepa3d/data/preprocess_shapenet_unpaired.py`
- UCPR evaluation:
  - `nepa3d/analysis/retrieval_ucpr.py`
- CPAC-UDF probe evaluation:
  - `nepa3d/analysis/completion_cpac_udf.py`
- mix config (compatible with current `datasets:` schema):
  - `nepa3d/configs/shapenet_unpaired_mix.yaml`
- wrappers:
  - `scripts/preprocess/make_shapenet_unpaired_split.sh`
  - `scripts/preprocess/preprocess_shapenet_unpaired.sh`
  - `scripts/analysis/nepa3d_ucpr.sh`
  - `scripts/analysis/nepa3d_cpac_udf.sh`
- run-plan memo:
  - `docs/eccv_ucpr_cpac_tables.md`

Design note:
- Existing training code (`pretrain.py`, `finetune_cls.py`, `mixed_pretrain.py`) was kept unchanged.
- Unpaired split uses already-built cache (`data/shapenet_cache_v0`) to avoid expensive re-preprocess.

### 12.2 Unpaired split/cache created from existing ShapeNet cache

Executed:

```bash
python -m nepa3d.data.shapenet_unpaired_split \
  --cache_root data/shapenet_cache_v0 \
  --train_split train --eval_split test \
  --out_json data/shapenet_unpaired_splits_v1.json \
  --ratios 0.34 0.33 0.33 --seed 0

python -m nepa3d.data.preprocess_shapenet_unpaired \
  --src_cache_root data/shapenet_cache_v0 \
  --split_json data/shapenet_unpaired_splits_v1.json \
  --out_root data/shapenet_unpaired_cache_v1 \
  --link_mode symlink
```

Current split counts:

- `train_mesh=15875`
- `train_pc=15407`
- `train_udf=15406`
- `eval=5185`

Metadata:
- `data/shapenet_unpaired_splits_v1.json`
- `data/shapenet_unpaired_cache_v1/_meta/split_source.json`

### 12.3 Smoke runs (debug scale)

Purpose:
- validate end-to-end execution of new UCPR/CPAC scripts before full-scale jobs.

Small debug pretrain checkpoint:

```bash
python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 96 --mix_seed 0 \
  --objective nepa \
  --batch 8 --epochs 1 --num_workers 0 \
  --n_point 64 --n_ray 64 \
  --d_model 128 --layers 2 --heads 4 \
  --save_dir runs/debug_ucpr_nepa_s0 \
  --seed 0
```

Generated ckpt:
- `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt`

UCPR debug eval:

```bash
python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/debug_ucpr_nepa_s0/ckpt_ep000.pt \
  --query_backend mesh \
  --gallery_backend udfgrid \
  --max_files 200 \
  --out_json results/ucpr_debug_mesh2udf.json
```

Result (`results/ucpr_debug_mesh2udf.json`):
- `r@1=0.005`
- `r@5=0.025`
- `r@10=0.050`
- `mAP=0.0302`

CPAC-UDF debug eval:

```bash
python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/debug_ucpr_nepa_s0/ckpt_ep000.pt \
  --context_backend pointcloud_noray \
  --n_context 64 --n_query 64 \
  --max_shapes 120 --head_train_ratio 0.25 \
  --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_debug_pc2udf.json
```

Result (`results/cpac_debug_pc2udf.json`):
- `mae=0.0819`
- `rmse=0.1099`
- `iou@tau=0.4786`

Important:
- These are smoke/debug numbers (small model + tiny training budget), not final ECCV table values.

### 12.4 Next run set (for next feedback cycle)

1) Pretrain full checkpoints for:
- mesh-only NEPA
- mixed-unpaired NEPA
- mixed-unpaired MAE

2) Evaluate UCPR (Table 1 candidates):
- `mesh -> pointcloud_noray`
- `mesh -> udfgrid`
- `pointcloud_noray -> udfgrid`

3) Evaluate CPAC-UDF (Table 2 candidate):
- `context_backend=pointcloud_noray`

4) Integrate into ScanObjectNN few-shot comparisons (Table 3 candidate).

### 12.5 Fast pretrain config decision (2026-02-13)

Goal:
- finish `mixed-unpaired` pretraining as fast as possible on local 2-GPU.

Quick throughput benchmark (same samples/epoch, NEPA/MAE each tested with `mix_num_samples=19200`, `epochs=1`, `n_point=n_ray=256`, `num_workers=6`):

| batch | NEPA wall sec | MAE wall sec |
|---:|---:|---:|
| 96 | 19.08 | 21.91 |
| 256 | 19.97 | 22.54 |
| 512 | 21.32 | 23.86 |
| 768 | 22.51 | 24.74 |

Decision:
- Use `batch=96` (fastest in local measurement).
- Even though VRAM has headroom, larger batch was slower in wall-time for this workload.

Relaunched run (resume from epoch-1 checkpoints):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective nepa --drop_ray_prob 0.3 \
  --batch 96 --epochs 50 --num_workers 6 --seed 0 \
  --n_point 256 --n_ray 256 \
  --resume runs/eccv_upmix_nepa_s0/ckpt_ep001.pt \
  --save_dir runs/eccv_upmix_nepa_s0

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective mae --mask_ratio 0.4 \
  --batch 96 --epochs 50 --num_workers 6 --seed 0 \
  --n_point 256 --n_ray 256 \
  --resume runs/eccv_upmix_mae_s0/ckpt_ep001.pt \
  --save_dir runs/eccv_upmix_mae_s0
```

Current logs:
- `logs/pretrain/eccv_unpaired_mixed/upmix_nepa_s0_fast_bs96_resume_20260213_145746.log`
- `logs/pretrain/eccv_unpaired_mixed/upmix_mae_s0_fast_bs96_resume_20260213_145746.log`

Resume note:
- old checkpoints did not contain optimizer state, so resume starts from loaded model weights with fresh optimizer (`[resume] optimizer state missing ...` is expected).

## 13) Feedback Sync Memo (Feb 19, 2026)

This memo mirrors active docs so legacy readers do not misread recent completion results.

Confirmed observations:

- A-1 (`coarse_to_fine 16->32->64`) is effective versus `grid uniform`; it is close to tuned `near_surface` on grid IoU.
- C-2 improves pool completion strongly, but can degrade grid completion when used alone.
- B-2 helps recover grid degradation and improves B-2+C-2 balance.
- `256->512` scale quick run improved, while naive continuation to `1024` showed instability/regression.

Interpretation caution:

- `encdec_plusgut_bbox` should be treated as a topology-coordinate diagnostic unless explicitly retrained as an independent line.
- Do not interpret this as a finalized fair main-table model line by itself.

Fix priority aligned with feedback:

1. Stabilize `512->1024` transition (optimizer-state handling around `pos_emb` resize + transition diagnostics).
2. Re-check encdec training path (decoder causal behavior / leakage checks / `qa_layout=split` metadata consistency).
3. Keep A-1 reported as query-design gain vs uniform, not unconditional win over tuned near-surface.
4. Keep UCPR table naming as `MRR (= single-positive mAP)` for clarity.
