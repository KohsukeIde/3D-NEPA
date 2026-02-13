# ECCV Loop Plan: UCPR / CPAC / Few-shot

This document tracks the minimal run set for the unpaired primitive loop.

## Table 0: Unpaired split/cache view

1) Build split JSON from existing ShapeNet cache:

```bash
qsub -v CACHE_ROOT=data/shapenet_cache_v0,OUT_JSON=data/shapenet_unpaired_splits_v1.json,SEED=0 \
  scripts/preprocess/make_shapenet_unpaired_split.sh
```

2) Materialize unpaired cache view (default symlink):

```bash
qsub -v SRC_CACHE_ROOT=data/shapenet_cache_v0,SPLIT_JSON=data/shapenet_unpaired_splits_v1.json,OUT_ROOT=data/shapenet_unpaired_cache_v1,LINK_MODE=symlink \
  scripts/preprocess/preprocess_shapenet_unpaired.sh
```

## Table 1: UCPR (Unpaired Cross-Primitive Retrieval)

Compare checkpoints:

- mesh-only NEPA
- mixed-unpaired NEPA
- mixed-unpaired MAE

Evaluate retrieval pairs:

- mesh -> pointcloud_noray
- mesh -> udfgrid
- pointcloud_noray -> udfgrid

Command template:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,QUERY_BACKEND=mesh,GALLERY_BACKEND=udfgrid,OUT_JSON=results/ucpr_mesh2udf.json \
  scripts/analysis/nepa3d_ucpr.sh
```

## Table 2: CPAC-UDF probe

Default setting:

- context_backend = pointcloud_noray
- target = UDF distance

Command:

```bash
qsub -v CACHE_ROOT=data/shapenet_unpaired_cache_v1,SPLIT=eval,CKPT=<ckpt>,CONTEXT_BACKEND=pointcloud_noray,OUT_JSON=results/cpac_pc2udf.json \
  scripts/analysis/nepa3d_cpac_udf.sh
```

## Table 3: ScanObjectNN few-shot

Use existing launcher:

```bash
bash scripts/finetune/run_scanobjectnn_main_table_local.sh
```

with methods extended by the new pretrain checkpoints.
