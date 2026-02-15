# NEPA-3D Active Track (ShapeNet/ScanObjectNN + UCPR/CPAC)

This file tracks the current active experiments.

- Active track:
  - M1 few-shot table on ScanObjectNN
  - UCPR/CPAC evaluation loop on ShapeNet-unpaired cache
- As-of snapshot date: February 15, 2026

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

### 2.2 ScanObjectNN cache (paper-safe setting)

Recommended input root (split-specific, required for paper tables):

- `data/ScanObjectNN/h5_files/main_split`

Recommended cache root:

- `data/scanobjectnn_main_split_v2`

Profile:

- `PT_POOL=4000`
- `RAY_POOL=256`
- `PT_SURFACE_RATIO=0.5`, `PT_SURFACE_SIGMA=0.02`

Local command (Python module):

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

Cluster wrapper (same defaults):

```bash
bash scripts/preprocess/preprocess_scanobjectnn.sh
```

Safety behavior:

- preprocess now fails fast when duplicate h5 basenames are detected across `scan_root` (`--allow_duplicate_stems` to override intentionally).
- provenance file is saved:
  - `data/scanobjectnn_main_split_v2/_meta/scanobjectnn_train_source.txt`
  - `data/scanobjectnn_main_split_v2/_meta/scanobjectnn_test_source.txt`

Output layout:

- `data/scanobjectnn_main_split_v2/train/class_XXX/*.npz`
- `data/scanobjectnn_main_split_v2/test/class_XXX/*.npz`

Legacy note:

- `data/scanobjectnn_cache_v2` is kept for historical/internal runs.
- For camera-ready tables, use a split-specific cache (`scanobjectnn_main_split_v2` or an explicitly named split cache).

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

3. Step0 migration (add `pt_dist_pc_pool` to source cache; required to avoid mesh-distance leakage in pointcloud backend):

```bash
.venv/bin/python -u -m nepa3d.data.migrate_add_pt_dist_pc_pool \
  --cache_root data/shapenet_cache_v0 \
  --splits train,test \
  --workers 16
```

Migration logs (example):

- `logs/preprocess_migrate_ptdistpc_test.log` (`ok=5185 fail=0`)
- `logs/preprocess_migrate_ptdistpc_train.log` (`ok=46688 fail=0`)

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

## 6) Current result snapshot (complete, M1 few-shot)

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

## 7) Notes

- `scratch K=10/20` shows majority-class collapse behavior in current seeds.
- This is not a K-shot sampling bug:
  - `stratified_kshot` produces exact per-class counts (`K x 15 classes`).
  - observed `test_acc=0.1115` matches test majority ratio (`2440/21889=0.11147`).
  - observed best-val around `0.1249` matches val majority ratio (`743/5948=0.12492`).
- ScanObjectNN task here is not raw point-set classification:
  - model input is query-token sequence with `POINT xyz + dist` (`pt_dist_pool`), and optional ray channels.
  - comparisons to raw-point baselines should explicitly note this representation difference.
- Main M1 few-shot run is complete (`75/75`).

## 8) UCPR/CPAC active results

Canonical artifact paths:

- UCPR JSON: `results/ucpr_*.json`
- CPAC JSON: `results/cpac_*.json`

### 8.1 External-PC run profile (synced)

This block is for the external machine run, recorded in the same style as M1.

Pretrain checkpoints used for evaluation:

- `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt`
- `runs/eccv_upmix_mae_s0/ckpt_ep049.pt`

Pretrain setting summary (both objectives):

- config: `nepa3d/configs/shapenet_unpaired_mix.yaml`
- `mix_num_samples=200000`, `mix_seed=0`
- `epochs=50`, `batch=96`, `n_point=256`, `n_ray=256`, `num_workers=6`
- resume from `ckpt_ep001.pt` to final `ckpt_ep049.pt`

Scripts/modules used:

- pretrain: `nepa3d.train.pretrain`
- retrieval eval: `nepa3d.analysis.retrieval_ucpr`
- completion eval: `nepa3d.analysis.completion_cpac_udf`
- migration: `nepa3d.data.migrate_add_pt_dist_pc_pool`
- wrappers:
  - `scripts/analysis/nepa3d_ucpr.sh`
  - `scripts/analysis/nepa3d_cpac_udf.sh`
  - `scripts/preprocess/migrate_add_pt_dist_pc_pool.sh`

Evaluation commands used for this synced snapshot:

```bash
# UCPR (mesh -> udfgrid), 1k subset
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_nepa_s0/ckpt_ep049.pt \
  --query_backend mesh \
  --gallery_backend udfgrid \
  --max_files 1000 \
  --out_json results/ucpr_nepa_ep049_mesh2udf_1k.json

.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --query_backend mesh \
  --gallery_backend udfgrid \
  --max_files 1000 \
  --out_json results/ucpr_mae_ep049_mesh2udf_1k.json

# CPAC-UDF (pointcloud_noray -> udf), 800-shape subset
.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_nepa_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --n_context 256 --n_query 256 \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_nepa_ep049_pc2udf_800.json

.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --n_context 256 --n_query 256 \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_mae_ep049_pc2udf_800.json

# UCPR (pointcloud_noray -> udfgrid), 1k subset, after Step0 migration
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_nepa_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray \
  --gallery_backend udfgrid \
  --max_files 1000 \
  --out_json results/ucpr_nepa_ep049_pc2udf_1k_postfix.json

.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray \
  --gallery_backend udfgrid \
  --max_files 1000 \
  --out_json results/ucpr_mae_ep049_pc2udf_1k_postfix.json

# CPAC-UDF non-transductive (head train split fixed to train_udf)
.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --head_train_split train_udf \
  --head_train_backend udfgrid \
  --context_backend pointcloud_noray \
  --ckpt runs/eccv_upmix_nepa_s0/ckpt_ep049.pt \
  --n_context 256 --n_query 256 \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_nepa_ep049_pc2udf_800_nontrans_postfix.json

.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 \
  --split eval \
  --head_train_split train_udf \
  --head_train_backend udfgrid \
  --context_backend pointcloud_noray \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --n_context 256 --n_query 256 \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_mae_ep049_pc2udf_800_nontrans_postfix.json
```

Synced logs/artifacts:

- pretrain logs: `logs/pretrain/eccv_unpaired_mixed/upmix_*_fast_bs96_resume_*.log`
- eval logs: `logs/analysis/ucpr_*_ep049_*.log`, `logs/analysis/cpac_*_ep049_*.log`
- result JSON:
  - `results/ucpr_nepa_ep049_mesh2udf_1k.json`
  - `results/ucpr_mae_ep049_mesh2udf_1k.json`
  - `results/cpac_nepa_ep049_pc2udf_800.json`
  - `results/cpac_mae_ep049_pc2udf_800.json`
  - `results/ucpr_nepa_ep049_pc2udf_1k_postfix.json`
  - `results/ucpr_mae_ep049_pc2udf_1k_postfix.json`
  - `results/cpac_nepa_ep049_pc2udf_800_nontrans_postfix.json`
  - `results/cpac_mae_ep049_pc2udf_800_nontrans_postfix.json`

### 8.2 UCPR

| Tag | CKPT | Query -> Gallery | Split | max_files | R@1 | R@5 | R@10 | mAP | Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `debug_local` | `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt` | `mesh -> udfgrid` | `eval` | 200 | 0.0050 | 0.0250 | 0.0500 | 0.0302 | smoke run |
| `external_ep049_nepa` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `mesh -> udfgrid` | `eval` | 1000 | 0.0070 | 0.0370 | 0.0510 | 0.0277 | synced from external-PC run |
| `external_ep049_mae` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `mesh -> udfgrid` | `eval` | 1000 | 0.1270 | 0.2930 | 0.3980 | 0.2196 | synced from external-PC run |
| `external_ep049_nepa_postfix` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udfgrid` | `eval` | 1000 | 0.9990 | 1.0000 | 1.0000 | 0.9995 | after Step0 migration (`pt_dist_pc_pool`) |
| `external_ep049_mae_postfix` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udfgrid` | `eval` | 1000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | after Step0 migration (`pt_dist_pc_pool`) |

### 8.3 CPAC-UDF

| Tag | CKPT | Context -> Target | Split | max_shapes | MAE | RMSE | IoU@0.03 | Note |
|---|---|---|---|---:|---:|---:|---:|---|
| `debug_local` | `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt` | `pointcloud_noray -> udf` | `eval` | 120 | 0.0819 | 0.1099 | 0.4786 | smoke run |
| `external_ep049_nepa` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1169 | 0.1510 | 0.4047 | synced from external-PC run |
| `external_ep049_mae` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.0917 | 0.1210 | 0.4204 | synced from external-PC run |
| `external_ep049_nepa_nontrans_postfix` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1252 | 0.1613 | 0.3174 | non-transductive (`head_train_split=train_udf`) after Step0 |
| `external_ep049_mae_nontrans_postfix` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1020 | 0.1341 | 0.3353 | non-transductive (`head_train_split=train_udf`) after Step0 |

### 8.4 Current readout (subset, single-seed)

- In this synced subset evaluation, MAE objective is stronger than NEPA on both UCPR and CPAC.
- After Step0/1 cleanup, MAE remains stronger than NEPA on CPAC non-transductive.
- `pointcloud_noray -> udfgrid` UCPR is near-saturated for both objectives in this setting; treat it as an easy pair and do not use it as the sole alignment evidence.
- This section is still a partial readout (`mesh->udf` only for UCPR, subset eval sizes), not the final ECCV main table.

### 8.5 Done checklist (this cycle, no multi-seed)

Executed in this cycle:

1. Step0 patch integration (`pt_dist_pc_pool` support + backend priority):
   - `nepa3d/backends/pointcloud_backend.py`
   - `nepa3d/data/preprocess_modelnet40.py`
   - `nepa3d/data/migrate_add_pt_dist_pc_pool.py`
   - `scripts/preprocess/migrate_add_pt_dist_pc_pool.sh`
2. Step1 patch integration (CPAC non-transductive defaults):
   - `nepa3d/analysis/completion_cpac_udf.py`
   - `scripts/analysis/nepa3d_cpac_udf.sh`
3. Cache migration completed:
   - `data/shapenet_cache_v0/test`: `ok=5185`, `fail=0`
   - `data/shapenet_cache_v0/train`: `ok=46688`, `fail=0`
4. Post-fix single-seed evaluations completed:
   - UCPR `pointcloud_noray -> udfgrid` (`max_files=1000`)
   - CPAC non-transductive (`head_train_split=train_udf`, `max_shapes=800`)

Single-seed result summary from this cycle:

- UCPR `pointcloud_noray -> udfgrid`:
  - NEPA: `R@1=0.9990`, `mAP=0.9995`
  - MAE: `R@1=1.0000`, `mAP=1.0000`
- CPAC non-transductive:
  - NEPA: `MAE=0.1252`, `RMSE=0.1613`, `IoU@0.03=0.3174`
  - MAE: `MAE=0.1020`, `RMSE=0.1341`, `IoU@0.03=0.3353`

Scope note:

- Multi-seed was intentionally skipped in this cycle.

### 8.6 QA-token + dual-mask integration (this cycle)

Integrated patch source:

- `/home/cvrt/Downloads/nepa_qa_dualmask_patch`

Patched files:

- `nepa3d/token/tokenizer.py`
- `nepa3d/models/query_nepa.py`
- `nepa3d/models/causal_transformer.py`
- `nepa3d/train/pretrain.py`
- `nepa3d/data/dataset.py`
- `nepa3d/data/mixed_pretrain.py`
- `nepa3d/train/finetune_cls.py`

Key behavior added:

- optional Q/A tokenization (`--qa_tokens 1`)
- answer-only NEPA loss on Q/A sequences
- dual masking in causal attention (`--dual_mask_near/far/window`)
- dual-mask warmup schedule (`--dual_mask_warmup_frac`)
- finetune-side `qa_tokens` inference/override (`--qa_tokens -1/0/1`)

Local fix applied during merge:

- `nepa3d/train/pretrain.py`: fixed `start_ep` typo in schedule init
  (`global_step` now initialized from resumed `step`)

Smoke run (passed):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --objective nepa \
  --qa_tokens 1 \
  --dual_mask_near 0.4 \
  --dual_mask_far 0.1 \
  --dual_mask_window 32 \
  --dual_mask_warmup_frac 0.05 \
  --epochs 1 --batch 8 --n_point 64 --n_ray 64 \
  --num_workers 2 \
  --save_every 1 --save_last 1 \
  --save_dir runs/_tmp_qa_dualmask_smoke \
  --seed 0
```

Smoke artifacts:

- log: `logs/pretrain/_tmp_qa_dualmask_smoke.log`
- ckpt: `runs/_tmp_qa_dualmask_smoke/ckpt_ep000.pt`
- last: `runs/_tmp_qa_dualmask_smoke/last.pt`

Note:

- UCPR/CPAC evaluators are QA-aware in this branch (`--qa_tokens -1` = infer from ckpt, `--qa_tokens 0/1` = manual override).
- Patched files:
  - `nepa3d/analysis/retrieval_ucpr.py`
  - `nepa3d/analysis/completion_cpac_udf.py`

### 8.7 QA pretrain runs (completed)

Purpose:

- test whether `Q/A + dual-mask` reduces Morton-local shortcut compared to `Q/A (no dual-mask)` under the same data/objective

Launched runs:

1. QA + dual-mask (`GPU0`)

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective nepa \
  --qa_tokens 1 \
  --dual_mask_near 0.4 \
  --dual_mask_far 0.1 \
  --dual_mask_window 32 \
  --dual_mask_warmup_frac 0.05 \
  --epochs 50 --batch 96 --n_point 256 --n_ray 256 \
  --num_workers 6 \
  --save_every 1 --save_last 1 \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_s0 \
  --seed 0
```

2. QA no-dual baseline (`GPU1`)

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --objective nepa \
  --qa_tokens 1 \
  --dual_mask_near 0.0 \
  --dual_mask_far 0.0 \
  --dual_mask_window 32 \
  --dual_mask_warmup_frac 0.05 \
  --epochs 50 --batch 96 --n_point 256 --n_ray 256 \
  --num_workers 6 \
  --save_every 1 --save_last 1 \
  --save_dir runs/eccv_upmix_nepa_qa_nodual_s0 \
  --seed 0
```

Run logs:

- `logs/pretrain/eccv_qa_dualmask/upmix_nepa_qa_dualmask_s0_bs96.log`
- `logs/pretrain/eccv_qa_dualmask/upmix_nepa_qa_nodual_s0_bs96.log`

Completion status:

- `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt` and `last.pt` exist
- `runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt` and `last.pt` exist

### 8.8 QA UCPR/CPAC eval commands (this cycle, single-seed)

Evaluated checkpoints:

- `runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt`
- `runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt`

UCPR hard-pair + easy-pair (independent sampling):

```bash
# dualmask
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_mesh2udf_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend udfgrid --gallery_backend mesh \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_udf2mesh_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_mesh2pc_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_pc2udf_1k_indep.json

# nodual
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_nodual_ep049_mesh2udf_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend udfgrid --gallery_backend mesh \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_nodual_ep049_udf2mesh_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_nodual_ep049_mesh2pc_1k_indep.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --out_json results/ucpr_nepa_qa_nodual_ep049_pc2udf_1k_indep.json
```

UCPR ablation (`pointcloud_noray -> udfgrid`):

```bash
# dualmask
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --ablate_point_xyz \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_pc2udf_1k_indep_ablate_xyz.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --ablate_point_dist \
  --out_json results/ucpr_nepa_qa_dualmask_ep049_pc2udf_1k_indep_ablate_dist.json

# nodual
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --ablate_point_xyz \
  --out_json results/ucpr_nepa_qa_nodual_ep049_pc2udf_1k_indep_ablate_xyz.json
.venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 --ablate_point_dist \
  --out_json results/ucpr_nepa_qa_nodual_ep049_pc2udf_1k_indep_ablate_dist.json
```

CPAC non-transductive:

```bash
.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_nepa_qa_dualmask_ep049_pc2udf_800_nontrans.json

.venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 \
  --out_json results/cpac_nepa_qa_nodual_ep049_pc2udf_800_nontrans.json
```

### 8.9 QA UCPR results (single-seed, complete)

Main UCPR (independent sampling, `eval_seed=0`, `eval_seed_gallery=999`, `max_files=1000`):

| Pair | NoDual R@1/R@5/R@10/mAP | DualMask R@1/R@5/R@10/mAP |
|---|---|---|
| `mesh -> udfgrid` | `0.001 / 0.015 / 0.029 / 0.0145` | `0.006 / 0.021 / 0.041 / 0.0226` |
| `udfgrid -> mesh` | `0.002 / 0.006 / 0.012 / 0.0099` | `0.001 / 0.011 / 0.017 / 0.0102` |
| `mesh -> pointcloud_noray` | `0.003 / 0.012 / 0.029 / 0.0161` | `0.004 / 0.021 / 0.043 / 0.0218` |
| `pointcloud_noray -> udfgrid` | `0.215 / 0.419 / 0.532 / 0.3175` | `0.093 / 0.229 / 0.297 / 0.1661` |

`pointcloud_noray -> udfgrid` ablation:

| Model | Base R@1/mAP | `ablate_point_xyz` R@1/mAP | `ablate_point_dist` R@1/mAP |
|---|---|---|---|
| NoDual | `0.215 / 0.3175` | `0.009 / 0.0341` | `0.126 / 0.2162` |
| DualMask | `0.093 / 0.1661` | `0.007 / 0.0342` | `0.092 / 0.1631` |

### 8.10 QA CPAC non-transductive results (single-seed, complete)

Protocol:

- `eval_split=eval`, `head_train_split=train_udf`, `head_train_backend=udfgrid`
- `n_shapes_head_train=15406`, `n_shapes_head_test=800`
- `context_backend=pointcloud_noray`, `n_context=256`, `n_query=256`

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| `qa_nodual_ep049` | 0.03072 | 0.04154 | 0.72231 |
| `qa_dualmask_ep049` | 0.02717 | 0.03623 | 0.72676 |

### 8.11 Readout (this QA cycle, no multi-seed)

- Both QA pretrains finished to `ep049`.
- Hard-pair UCPR (`mesh -> udfgrid`, `mesh -> pointcloud_noray`) improved with dual-mask over no-dual.
- Easy-pair UCPR (`pointcloud_noray -> udfgrid`) is still higher for no-dual.
- CPAC non-trans improved with dual-mask (`MAE/RMSE` lower, `IoU@0.03` higher).
- Multi-seed was intentionally skipped in this cycle.
