# NEPA-3D Active Track (ShapeNet/ScanObjectNN + UCPR/CPAC)

This file tracks the current active experiments.

- Active track:
  - M1 few-shot table on ScanObjectNN
  - UCPR/CPAC evaluation loop on ShapeNet-unpaired cache
- As-of snapshot date: February 13, 2026

Legacy means ModelNet40-era experiments only. See `nepa3d/docs/results_modelnet40_legacy.md`.

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

### 2.2 ScanObjectNN v2 cache (`data/scanobjectnn_cache_v2`)

Input root:

- `data/ScanObjectNN/h5_files`

Profile used in active runs:

- `PT_POOL=4000`
- `RAY_POOL=256`
- `PT_SURFACE_RATIO=0.5`, `PT_SURFACE_SIGMA=0.02`

Local command (Python module; cluster wrapper is `scripts/preprocess/preprocess_scanobjectnn.sh`):

```bash
.venv/bin/python -u -m nepa3d.data.preprocess_scanobjectnn \
  --scan_root data/ScanObjectNN/h5_files \
  --out_root data/scanobjectnn_cache_v2 \
  --split all \
  --pt_pool 4000 \
  --ray_pool 256 \
  --pt_surface_ratio 0.5 \
  --pt_surface_sigma 0.02 \
  --seed 0
```

Output layout:

- `data/scanobjectnn_cache_v2/train/class_XXX/*.npz`
- `data/scanobjectnn_cache_v2/test/class_XXX/*.npz`

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

## 3) Train/eval launch commands

M1 pretrain (3 jobs on 2 GPUs):

```bash
bash scripts/pretrain/launch_shapenet_m1_pretrains_local.sh
```

M1 fine-tune table:

```bash
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

Optional chain (start fine-tune after pretrain success):

```bash
bash scripts/finetune/launch_scanobjectnn_m1_after_pretrain.sh
```

UCPR command template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,OUT_JSON=results/ucpr_mesh2udf.json \
  scripts/analysis/nepa3d_ucpr.sh
```

CPAC-UDF command template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,CONTEXT_BACKEND=pointcloud_noray,OUT_JSON=results/cpac_pc2udf.json \
  scripts/analysis/nepa3d_cpac_udf.sh
```

Detailed UCPR/CPAC table plan:

- `docs/eccv_ucpr_cpac_tables.md`

## 4) Resume behavior

Pretrain:

- `pretrain.py` supports `--resume` and `--auto_resume`.
- Launchers pass `--resume <save_dir>/last.pt`.
- Interrupted pretrain resumes from `last.pt` if present.

Fine-tune table:

- Each job is skipped when `runs/scan_<method>_k<K>_s<seed>/last.pt` exists.
- Relaunch command resumes remaining jobs only.

## 5) Logs and outputs

- pretrain logs: `logs/pretrain/m1/`
- fine-tune logs: `logs/finetune/scan_m1_table/`
- fine-tune job logs: `logs/finetune/scan_m1_table/jobs/*.log`
- UCPR/CPAC result JSON (recommended path): `results/ucpr_*.json`, `results/cpac_*.json`
- outputs: `runs/scan_<method>_k<K>_s<seed>/`

Helper scripts:

```bash
bash scripts/logs/show_pipeline_status.sh
bash scripts/logs/cleanup_stale_pids.sh
```

## 6) Current result snapshot (partial, M1 few-shot)

Status (as of February 13, 2026):

- completed jobs: `30 / 75`
- completion by method:
  - `scratch`: `10/15`
  - `shapenet_nepa`: `10/15`
  - `shapenet_mesh_udf_nepa`: `3/15`
  - `shapenet_mix_nepa`: `4/15`
  - `shapenet_mix_mae`: `3/15`

Table below is computed from completed `runs/*/last.pt` only.
`n(seed)` is the number of finished seeds for each `(method, K)`.

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
| `shapenet_mix_nepa` | 1 | 1 | 0.1293 +- 0.0000 |
| `shapenet_mix_nepa` | 5 | 1 | 0.2546 +- 0.0000 |
| `shapenet_mix_nepa` | 20 | 1 | 0.3793 +- 0.0000 |
| `shapenet_mix_mae` | 0 | 1 | 0.7856 +- 0.0000 |
| `shapenet_mix_mae` | 1 | 1 | 0.1399 +- 0.0000 |
| `shapenet_mix_mae` | 10 | 1 | 0.2662 +- 0.0000 |

## 7) Notes

- `scratch K=10/20` shows majority-class collapse behavior in current seeds.
- Do not finalize claims until all `75` jobs are complete.

## 8) UCPR/CPAC active results

Canonical artifact paths:

- UCPR JSON: `results/ucpr_*.json`
- CPAC JSON: `results/cpac_*.json`

Current recorded metrics:

### UCPR

| Tag | CKPT | Query -> Gallery | Split | max_files | R@1 | R@5 | R@10 | mAP | Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `debug_local` | `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt` | `mesh -> udfgrid` | `eval` | 200 | 0.0050 | 0.0250 | 0.0500 | 0.0302 | migrated from old README text |

### CPAC-UDF

| Tag | CKPT | Context -> Target | Split | max_shapes | MAE | RMSE | Note |
|---|---|---|---|---:|---:|---:|---|
| `pending_external_sync` | `-` | `pointcloud_noray -> udf` | `eval` | - | - | - | running on another machine |
