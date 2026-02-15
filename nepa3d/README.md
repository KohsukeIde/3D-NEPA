# NEPA-3D Active Track (ShapeNet/ScanObjectNN + UCPR/CPAC)

This file is the active-run hub.

- Active track:
  - ScanObjectNN few-shot/classification (paper-safe protocol variants)
  - UCPR/CPAC evaluation loop on ShapeNet-unpaired cache
- As-of snapshot date: February 15, 2026

Legacy means ModelNet40-era experiments only. See `nepa3d/docs/results_modelnet40_legacy.md`.

## Quick Links

- Results index: `nepa3d/docs/results_index.md`
- ScanObjectNN core3 active tables: `nepa3d/docs/results_scanobjectnn_core3_active.md`
- ScanObjectNN M1 legacy snapshot (`75/75`): `nepa3d/docs/results_scanobjectnn_m1_legacy.md`
- UCPR/CPAC active results (incl. QA cycle): `nepa3d/docs/results_ucpr_cpac_active.md`
- UCPR/CPAC planning doc: `nepa3d/docs/eccv_ucpr_cpac_tables.md`

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

### 6.1 Paper-safe ScanObjectNN core3 (active)

Status (as of February 15, 2026):

- completed jobs: `135 / 135`
- variants: `obj_bg`, `obj_only`, `pb_t50_rs`
- methods: `scratch`, `shapenet_nepa`, `shapenet_mesh_udf_nepa`

Best-by-K summary:

| Variant | K | best method | test_acc mean +- std |
|---|---:|---|---:|
| `obj_bg` | 0 | `shapenet_mesh_udf_nepa` | 0.6575 +- 0.0092 |
| `obj_bg` | 20 | `shapenet_mesh_udf_nepa` | 0.4687 +- 0.0239 |
| `obj_only` | 0 | `shapenet_mesh_udf_nepa` | 0.6621 +- 0.0165 |
| `obj_only` | 20 | `shapenet_mesh_udf_nepa` | 0.4968 +- 0.0162 |
| `pb_t50_rs` | 0 | `shapenet_mesh_udf_nepa` | 0.5228 +- 0.0092 |
| `pb_t50_rs` | 20 | `shapenet_mesh_udf_nepa` | 0.2898 +- 0.0215 |

Full tables:

- `nepa3d/docs/results_scanobjectnn_core3_active.md`

### 6.2 Legacy M1 (`75/75`) moved

- `nepa3d/docs/results_scanobjectnn_m1_legacy.md`

### 6.3 UCPR/CPAC active details moved

- `nepa3d/docs/results_ucpr_cpac_active.md`
- latest follow-up (`pooling/context controls`, Feb 15, 2026) is also tracked there

## 7) Notes for paper

- For camera-ready classification tables, use split-specific cache (`main_split`) and protocol-variant tables.
- Current classification tables are aggregated from `last.pt`; in this codebase `last.pt` is saved after reloading best-val state.
- ScanObjectNN task here is query-token classification (`POINT xyz + dist`, optional ray), not raw point-set classification.
- MC evaluation is used in current setup (`mc_eval_k_test=4`); report this explicitly against raw-point baselines.
- ScanObjectNN classification should be treated as downstream/supporting evidence; core unpaired capability evidence is UCPR/CPAC.
- Fine-tune launcher now defaults to `N_RAY=0` when `BACKEND=pointcloud_noray` and `N_RAY` is not explicitly set.
- ModelNet40 few-shot protocol support was added to `finetune_cls.py` via `--fewshot_n_way` and `--fewshot_way_seed` (episodic N-way M-shot trials).
