# NEPA-3D Active Track (ShapeNet/ScanObjectNN + UCPR/CPAC)

This file is the active-run hub.

- Active track:
  - ScanObjectNN few-shot/classification (paper-safe protocol variants)
  - UCPR/CPAC evaluation loop on ShapeNet-unpaired cache
  - K-plane/Tri-plane baseline track on ShapeNet-unpaired cache (UCPR/CPAC)
- As-of snapshot date: February 16, 2026

Legacy means early/pre-review snapshots (including old ModelNet40-era runs). Current active ModelNet40 protocol is tracked separately.

## Quick Links

- Results index: `nepa3d/docs/results_index.md`
- ScanObjectNN core3 active tables: `nepa3d/docs/results_scanobjectnn_core3_active.md`
- ScanObjectNN review tables (paper-safe core3, full-ft + linear-probe): `nepa3d/docs/results_scanobjectnn_review_active.md`
- ModelNet40 PointGPT-style protocol (full + few-shot LP): `nepa3d/docs/results_modelnet40_pointgpt_active.md`
- ScanObjectNN M1 legacy snapshot (`75/75`): `nepa3d/docs/results_scanobjectnn_m1_legacy.md`
- UCPR/CPAC active results (incl. QA cycle): `nepa3d/docs/results_ucpr_cpac_active.md`
- UCPR/CPAC planning doc: `nepa3d/docs/eccv_ucpr_cpac_tables.md`
- K-plane/Tri-plane results (pilot + full e50) are tracked in `nepa3d/docs/results_ucpr_cpac_active.md`

## 1) Current experiment definition

Pretrain corpora:

- `shapenet_nepa`: ShapeNet mesh only
- `shapenet_mesh_udf_nepa`: ShapeNet mesh + ShapeNet UDF
- `shapenet_mix_nepa`: ShapeNet mesh + ShapeNet UDF + ScanObjectNN pointcloud no-ray
- `shapenet_mix_mae`: same mixed data as above, objective=`mae`

Downstream task:

- ScanObjectNN classification
- full + few-shot: `K=0,1,5,10,20`
- seeds: `0,1,2`

## 2) Data preparation

### 2.1 ShapeNet cache (`data/shapenet_cache_v0`)

Input root:

- `data/ShapeNetCore.v2`

Profile used in active runs:

- `PC_POINTS=2048`
- `PT_POOL=20000`
- `RAY_POOL=8000`
- `N_VIEWS=20`
- `RAYS_PER_VIEW=400`
- `PC_GRID=64`, `PC_DILATE=1`
- `DF_GRID=64`, `DF_DILATE=1`
- `PT_SURFACE_RATIO=0.5`, `PT_SURFACE_SIGMA=0.02`
- `PT_DIST_MODE=kdtree`, `DIST_REF_POINTS=8192`
- `WORKERS=4`, `CHUNK_SIZE=1`, `MAX_TASKS_PER_CHILD=2`

Command:

```bash
SHAPENET_ROOT=data/ShapeNetCore.v2 \
OUT_ROOT=data/shapenet_cache_v0 \
SPLIT=all \
PC_POINTS=2048 PT_POOL=20000 RAY_POOL=8000 \
N_VIEWS=20 RAYS_PER_VIEW=400 \
PC_GRID=64 PC_DILATE=1 DF_GRID=64 DF_DILATE=1 \
PT_SURFACE_RATIO=0.5 PT_SURFACE_SIGMA=0.02 \
PT_DIST_MODE=kdtree DIST_REF_POINTS=8192 \
WORKERS=4 CHUNK_SIZE=1 MAX_TASKS_PER_CHILD=2 \
bash scripts/preprocess/preprocess_shapenet.sh
```

Output layout:

- `data/shapenet_cache_v0/train/<synset>/<model>.npz`
- `data/shapenet_cache_v0/test/<synset>/<model>.npz`
- split manifest: `data/shapenet_cache_v0/_splits/{train.txt,test.txt}`

### 2.2 ScanObjectNN cache (paper-safe setting)

Recommended input root (split-specific, required for paper tables):

- `data/ScanObjectNN/h5_files/main_split`

Recommended cache root:

- `data/scanobjectnn_main_split_v2`

Profile:

- `PT_POOL=4000`
- `RAY_POOL=256`
- `PT_SURFACE_RATIO=0.5`, `PT_SURFACE_SIGMA=0.02`

Local command:

```bash
.venv/bin/python -u -m nepa3d.data.preprocess_scanobjectnn \
  --scan_root data/ScanObjectNN/h5_files/main_split \
  --out_root data/scanobjectnn_main_split_v2 \
  --split all \
  --pt_pool 4000 \
  --ray_pool 256 \
  --pt_surface_ratio 0.5 \
  --pt_surface_sigma 0.02 \
  --seed 0
```

Safety behavior:

- preprocess fails fast when duplicate h5 basenames are detected across `scan_root` (`--allow_duplicate_stems` to override intentionally)
- provenance files:
  - `data/scanobjectnn_main_split_v2/_meta/scanobjectnn_train_source.txt`
  - `data/scanobjectnn_main_split_v2/_meta/scanobjectnn_test_source.txt`

Legacy note:

- `data/scanobjectnn_cache_v2` is kept for historical/internal runs only

### 2.3 ShapeNet unpaired cache for UCPR/CPAC (`data/shapenet_unpaired_cache_v1`)

1. Create split map JSON:

```bash
.venv/bin/python -u -m nepa3d.data.shapenet_unpaired_split \
  --cache_root data/shapenet_cache_v0 \
  --train_split train \
  --eval_split test \
  --out_json data/shapenet_unpaired_splits_v1.json \
  --seed 0 \
  --ratios 0.34 0.33 0.33
```

2. Materialize split view:

```bash
.venv/bin/python -u -m nepa3d.data.preprocess_shapenet_unpaired \
  --src_cache_root data/shapenet_cache_v0 \
  --split_json data/shapenet_unpaired_splits_v1.json \
  --out_root data/shapenet_unpaired_cache_v1 \
  --splits train_mesh train_pc train_udf eval \
  --link_mode symlink
```

3. Step0 migration (add `pt_dist_pc_pool` to source cache):

```bash
.venv/bin/python -u -m nepa3d.data.migrate_add_pt_dist_pc_pool \
  --cache_root data/shapenet_cache_v0 \
  --splits train,test \
  --workers 16
```

## 3) Train/eval launch commands

M1 pretrain:

```bash
bash scripts/pretrain/launch_shapenet_m1_pretrains_local.sh
```

M1 fine-tune table:

```bash
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

Paper-safe protocol variants (OBJ-BG / OBJ-ONLY / PB-T50-RS):

```bash
bash scripts/finetune/launch_scanobjectnn_variants_chain_local.sh
```

Review-response chain (core3 with mix methods + `N_RAY=0` + linear-probe):

```bash
bash scripts/finetune/launch_scanobjectnn_review_chain_local.sh
```

Review follow-up chain (no `n_point` scaling; K=1 seed expansion + dist ablation + QA+dualmask spot-check):

```bash
bash scripts/finetune/launch_scanobjectnn_review_followups_chain_local.sh
```

ModelNet40 protocol run (full + episodic few-shot):

```bash
bash scripts/finetune/launch_modelnet40_pointgpt_protocol_local.sh
```

Auto-chain (wait current review jobs, then start ModelNet40 protocol):

```bash
bash scripts/finetune/launch_after_review_modelnet_chain_local.sh
```

UCPR template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,EVAL_SEED=0,EVAL_SEED_GALLERY=999,POOLING=mean_a,OUT_JSON=results/ucpr_mesh2udf.json \
  scripts/analysis/nepa3d_ucpr.sh
```

CPAC-UDF template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,CONTEXT_BACKEND=pointcloud_noray,HEAD_TRAIN_SPLIT=train_udf,HEAD_TRAIN_BACKEND=udfgrid,DISJOINT_CONTEXT_QUERY=1,CONTEXT_MODE_TEST=normal,REP_SOURCE=h,QUERY_SOURCE=pool,BASELINE=nn_copy,OUT_JSON=results/cpac_pc2udf.json \
  scripts/analysis/nepa3d_cpac_udf.sh
```

K-plane/Tri-plane pretrain template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,MIX_CONFIG=nepa3d/configs/shapenet_unpaired_mix.yaml,PLANE_TYPE=kplane,FUSION=product,PLANE_RES=64,PLANE_CH=32,HIDDEN_DIM=128,EPOCHS=50,BATCH=96,SAVE_DIR=runs/eccv_kplane_product_s0 \
  scripts/pretrain/nepa3d_kplane_pretrain.sh
```

K-plane UCPR template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<kplane_ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,POOLING=mean_query,EVAL_SEED=0,EVAL_SEED_GALLERY=999,OUT_JSON=results/ucpr_kplane_mesh2udf.json \
  scripts/analysis/nepa3d_kplane_ucpr.sh
```

K-plane CPAC template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<kplane_ckpt>,CONTEXT_BACKEND=pointcloud_noray,HEAD_TRAIN_SPLIT=train_udf,HEAD_TRAIN_BACKEND=udfgrid,QUERY_SOURCE=pool,BASELINE=nn_copy,OUT_JSON=results/cpac_kplane_pc2udf.json \
  scripts/analysis/nepa3d_kplane_cpac.sh
```

K-plane full-chain (wait pretrain completion, then run UCPR/CPAC pack):

```bash
bash scripts/analysis/launch_kplane_full_chain_local.sh
```

K-plane fusion sweep chain (wait `kplane_sum` / `kplane_sum_large` e50 completion, then run tie-aware UCPR/CPAC pack):

```bash
bash scripts/analysis/launch_kplane_sum_chain_local.sh
```

Qualitative CPAC (grid query + marching cubes):

```bash
.venv/bin/python -u -m nepa3d.analysis.qualitative_cpac_marching_cubes \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt <ckpt> \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --grid_res 32 --mc_level 0.03 \
  --out_dir results/qual_mc
```

## 4) Resume behavior

Pretrain:

- `pretrain.py` supports `--resume` and `--auto_resume`
- launchers pass `--resume <save_dir>/last.pt`

Fine-tune:

- each job is skipped when `runs/.../last.pt` exists
- relaunch resumes remaining jobs only

## 5) Logs and outputs

- pretrain logs: `logs/pretrain/`
- fine-tune logs: `logs/finetune/`
- UCPR/CPAC JSON: `results/ucpr_*.json`, `results/cpac_*.json`

Helpers:

```bash
bash scripts/logs/show_pipeline_status.sh
bash scripts/logs/cleanup_stale_pids.sh
```

## 6) Current result snapshot

### 6.1 ScanObjectNN review tables (active, complete)

Status (as of February 16, 2026):

- full fine-tune: `225/225` complete
- linear probe (`freeze_backbone=1`): `225/225` complete
- variants: `obj_bg`, `obj_only`, `pb_t50_rs`
- methods: `scratch`, `shapenet_nepa`, `shapenet_mesh_udf_nepa`, `shapenet_mix_nepa`, `shapenet_mix_mae`

Best-by-K (full fine-tune, selected):

| Variant | K | best method | test_acc mean +- std |
|---|---:|---|---:|
| `obj_bg` | 0 | `shapenet_mix_nepa` | 0.6718 +- 0.0077 |
| `obj_bg` | 20 | `shapenet_mix_nepa` | 0.4997 +- 0.0131 |
| `obj_only` | 0 | `shapenet_mix_nepa` | 0.6690 +- 0.0029 |
| `obj_only` | 20 | `shapenet_mix_nepa` | 0.4945 +- 0.0045 |
| `pb_t50_rs` | 0 | `shapenet_mix_nepa` | 0.5501 +- 0.0023 |
| `pb_t50_rs` | 20 | `shapenet_mix_nepa` | 0.3038 +- 0.0068 |

Full tables:

- `nepa3d/docs/results_scanobjectnn_review_active.md`
- raw/summary CSV: `results/scan_variants_review_raw.csv`, `results/scan_variants_review_summary.csv`

### 6.2 ModelNet40 PointGPT-style protocol (active, complete)

Status (as of February 16, 2026):

- full fine-tune: `15/15` complete (5 methods x 3 seeds)
- few-shot linear-probe episodic: `200/200` complete (`N={5,10}`, `K={10,20}`, 10 trials)
- total: `215/215` complete

Best-by-setting:

- full: `shapenet_mesh_udf_nepa` (`0.8633 +- 0.0059`)
- few-shot LP `N=5, K=10`: `shapenet_nepa` (`0.5893 +- 0.1759`)
- few-shot LP `N=5, K=20`: `shapenet_nepa` (`0.6031 +- 0.1612`)
- few-shot LP `N=10, K=10`: `shapenet_nepa` (`0.4091 +- 0.0667`)
- few-shot LP `N=10, K=20`: `shapenet_nepa` (`0.5151 +- 0.0824`)

Full tables:

- `nepa3d/docs/results_modelnet40_pointgpt_active.md`
- raw/summary CSV: `results/modelnet40_pointgpt_protocol_raw.csv`, `results/modelnet40_pointgpt_protocol_summary.csv`

### 6.3 Paper-safe ScanObjectNN core3 baseline snapshot

- baseline snapshot (without mix methods): `nepa3d/docs/results_scanobjectnn_core3_active.md`

### 6.4 Legacy M1 (`75/75`) moved

- `nepa3d/docs/results_scanobjectnn_m1_legacy.md`

### 6.5 UCPR/CPAC active details moved

- `nepa3d/docs/results_ucpr_cpac_active.md`
- latest follow-up (`pooling/context controls`, Feb 15, 2026) is also tracked there
- latest MAE parity + eval-seed variance follow-up (Feb 15, 2026) is tracked there

### 6.6 K-plane/Tri-plane full run (active, e50)

- completed checkpoints:
  - `runs/eccv_kplane_product_s0/ckpt_ep049.pt`
  - `runs/eccv_triplane_sum_s0/ckpt_ep049.pt`
- quick readout (`mesh -> udfgrid`, `pooling=mean_query`, `max_files=1000`, tie-aware ranking):
  - k-plane(product): `R@1=0.002`, `mAP=0.01992`
  - tri-plane(sum): `R@1=0.050`, `mAP=0.11058`
- CPAC readout (`pointcloud_noray -> udf`, non-trans, `max_shapes=800`):
  - k-plane(product): `MAE=0.17604`, `RMSE=0.24831`, `IoU@0.03=0.61277`
  - tri-plane(sum): `MAE=0.09774`, `RMSE=0.15701`, `IoU@0.03=0.75702`
- retrieval evaluator fix:
  - tie-aware ranking + constant-embedding sanity check added to `retrieval_ucpr.py` and `retrieval_kplane.py`
  - details and before/after comparison are in:
    - `nepa3d/docs/results_ucpr_cpac_active.md` (`Tie-Aware UCPR Fix + Sanity`)
- full commands and pooling/ablation/control results are in:
  - `nepa3d/docs/results_ucpr_cpac_active.md`

### 6.7 K-plane fusion sweep (completed, e50)

- completed checkpoints:
  - `runs/eccv_kplane_sum_s0/ckpt_ep049.pt` (`plane_type=kplane`, `fusion=sum`, `plane_res=64`, `ch=32`, `hid=128`)
  - `runs/eccv_kplane_sum_large_s0/ckpt_ep049.pt` (`plane_type=kplane`, `fusion=sum`, `plane_res=128`, `ch=64`, `hid=256`)
- logs:
  - `logs/pretrain/eccv_kplane_baseline/kplane_sum_s0_bs96_e50.log`
  - `logs/pretrain/eccv_kplane_baseline/kplane_sum_large_s0_bs96_e50.log`
- quick readout (tie-aware UCPR):
  - `mesh -> udfgrid`
    - kplane(sum): `R@1=0.036`, `mAP=0.09429`
    - kplane(sum, large): `R@1=0.030`, `mAP=0.06659`
  - `mesh -> pointcloud_noray`
    - kplane(sum): `R@1=0.040`, `mAP=0.09963`
    - kplane(sum, large): `R@1=0.040`, `mAP=0.09308`
- quick readout (CPAC pool normal, non-trans):
  - kplane(sum): `MAE=0.09776`, `RMSE=0.15700`, `IoU@0.03=0.75630`
  - kplane(sum, large): `MAE=0.14229`, `RMSE=0.21260`, `IoU@0.03=0.52054`
  - `kplane(sum)` is the stronger variant in this sweep
- auto-eval chain:
  - launcher: `scripts/analysis/launch_kplane_sum_chain_local.sh`
  - worker: `scripts/analysis/run_kplane_sum_chain_local.sh`
  - pipeline log: `logs/analysis/kplane_sum_chain/pipeline.log`
  - final completion is tracked by `results/ucpr_kplane_sum*_e50*_tiefix.json` and `results/cpac_kplane_sum*_e50*.json`

## 7) Notes for paper

- For camera-ready classification tables, use split-specific cache (`main_split`) and protocol-variant tables.
- Current classification tables are aggregated from `last.pt`; in this codebase `last.pt` is saved after reloading best-val state.
- ScanObjectNN task here is query-token classification (`POINT xyz + dist`, optional ray), not raw point-set classification.
- In ScanObjectNN caches used for active classification tables, `pt_dist_pool` is pointcloud-derived (KDTree to observed scan points), so this downstream setup is valid as point-observation-based query-token classification.
- MC evaluation is used in current setup (`mc_eval_k_test=4`); report this explicitly against raw-point baselines.
- ScanObjectNN classification should be treated as downstream/supporting evidence; core unpaired capability evidence is UCPR/CPAC.
- In review classification tables, `shapenet_*_nepa` checkpoints are `objective=nepa` (non-MAE masking), while `shapenet_mix_mae` is `objective=mae` with token masking (`mask_ratio=0.4`); those tables use pre-QA/dual-mask checkpoints.
- Fine-tune launcher now defaults to `N_RAY=0` when `BACKEND=pointcloud_noray` and `N_RAY` is not explicitly set.
- ModelNet40 few-shot protocol support was added to `finetune_cls.py` via `--fewshot_n_way` and `--fewshot_way_seed` (episodic N-way M-shot trials).
- Current ModelNet40 protocol in this repo is `full fine-tune` + `few-shot linear probe`; LP-FT (two-stage linear-probe then full-unfreeze) is not included yet.
