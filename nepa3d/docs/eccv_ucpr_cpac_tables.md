# ECCV Loop Plan: UCPR / CPAC / Few-shot

This document tracks the minimal run set for the unpaired primitive loop.

## Scope

- Active experiment logs and latest numbers: `nepa3d/README.md`
- This file: stable run matrix and command templates for Table 1/2/3

## Table 0: Unpaired split/cache view

1. Build split JSON from existing ShapeNet cache:

```bash
qsub -v CACHE_ROOT=data/shapenet_cache_v0,OUT_JSON=data/shapenet_unpaired_splits_v1.json,SEED=0 \
  scripts/preprocess/make_shapenet_unpaired_split.sh
```

2. Materialize unpaired cache view (default symlink):

```bash
qsub -v SRC_CACHE_ROOT=data/shapenet_cache_v0,SPLIT_JSON=data/shapenet_unpaired_splits_v1.json,OUT_ROOT=data/shapenet_unpaired_cache_v1,LINK_MODE=symlink \
  scripts/preprocess/preprocess_shapenet_unpaired.sh
```

3. Apply Step0 migration (`pt_dist_pc_pool`) to source cache before UCPR/CPAC:

```bash
qsub -v CACHE_ROOT=data/shapenet_cache_v0,SPLITS=train,test,WORKERS=16 \
  scripts/preprocess/migrate_add_pt_dist_pc_pool.sh
```

## Table 1: UCPR (Unpaired Cross-Primitive Retrieval)

### Rows (methods)

- `mesh-only NEPA`
- `mixed-unpaired NEPA`
- `mixed-unpaired MAE`

### Columns (retrieval pairs)

- `mesh -> pointcloud_noray` (`R@1/5/10`, `MRR`)
- `mesh -> udfgrid` (`R@1/5/10`, `MRR`)
- `pointcloud_noray -> udfgrid` (`R@1/5/10`, `MRR`) as easy/sanity pair

Command template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,OUT_JSON=results/ucpr_mesh2udf.json \
  scripts/analysis/nepa3d_ucpr.sh
```

MUST-1 diagnostics (recommended before paper table freeze):

- independent sampling:
  - set `EVAL_SEED=0`, `EVAL_SEED_GALLERY=999`
- feature ablations:
  - `ABLATE_POINT_XYZ=1`
  - `ABLATE_POINT_DIST=1`

UCPR metric naming note:

- Current evaluator is single-positive retrieval; reported `mAP` is equivalent to reciprocal-rank average.
- In tables/captions, use `MRR (= single-positive mAP)` to avoid ambiguity.

Example:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,EVAL_SEED=0,EVAL_SEED_GALLERY=999,OUT_JSON=results/ucpr_mesh2udf_indep.json \
  scripts/analysis/nepa3d_ucpr.sh
```

## Table 2: CPAC-UDF probe

Default setting:

- `context_backend=pointcloud_noray`
- `target=UDF distance`
- non-transductive head training (`HEAD_TRAIN_SPLIT=train_udf`, `HEAD_TRAIN_BACKEND=udfgrid`)

Command template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,CONTEXT_BACKEND=pointcloud_noray,HEAD_TRAIN_SPLIT=train_udf,HEAD_TRAIN_BACKEND=udfgrid,OUT_JSON=results/cpac_pc2udf_nontrans.json \
  scripts/analysis/nepa3d_cpac_udf.sh
```

## Table 3: ScanObjectNN few-shot

Use the local table launcher:

```bash
bash scripts/finetune/run_scanobjectnn_m1_table_local.sh
```

Methods should include scratch, ShapeNet-only, ShapeNet+UDF, mixed-unpaired (NEPA/MAE).

## Notes

- Keep this file as protocol-level documentation.
- Put active run status, logs, and latest metrics only in `nepa3d/README.md`.
