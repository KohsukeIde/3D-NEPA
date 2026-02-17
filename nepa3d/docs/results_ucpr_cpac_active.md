# UCPR/CPAC Active Results

This file stores active UCPR/CPAC experiment results and command logs that were previously embedded in `nepa3d/README.md`.

- Primary artifact paths: `results/ucpr_*.json`, `results/cpac_*.json`

## UCPR/CPAC active results

Canonical artifact paths:

- UCPR JSON: `results/ucpr_*.json`
- CPAC JSON: `results/cpac_*.json`

### External-PC run profile (synced)

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

### UCPR

| Tag | CKPT | Query -> Gallery | Split | max_files | R@1 | R@5 | R@10 | mAP | Note |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `debug_local` | `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt` | `mesh -> udfgrid` | `eval` | 200 | 0.0050 | 0.0250 | 0.0500 | 0.0302 | smoke run |
| `external_ep049_nepa` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `mesh -> udfgrid` | `eval` | 1000 | 0.0070 | 0.0370 | 0.0510 | 0.0277 | synced from external-PC run |
| `external_ep049_mae` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `mesh -> udfgrid` | `eval` | 1000 | 0.1270 | 0.2930 | 0.3980 | 0.2196 | synced from external-PC run |
| `external_ep049_nepa_postfix` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udfgrid` | `eval` | 1000 | 0.9990 | 1.0000 | 1.0000 | 0.9995 | after Step0 migration (`pt_dist_pc_pool`) |
| `external_ep049_mae_postfix` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udfgrid` | `eval` | 1000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | after Step0 migration (`pt_dist_pc_pool`) |

### CPAC-UDF

| Tag | CKPT | Context -> Target | Split | max_shapes | MAE | RMSE | IoU@0.03 | Note |
|---|---|---|---|---:|---:|---:|---:|---|
| `debug_local` | `runs/debug_ucpr_nepa_s0/ckpt_ep000.pt` | `pointcloud_noray -> udf` | `eval` | 120 | 0.0819 | 0.1099 | 0.4786 | smoke run |
| `external_ep049_nepa` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1169 | 0.1510 | 0.4047 | synced from external-PC run |
| `external_ep049_mae` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.0917 | 0.1210 | 0.4204 | synced from external-PC run |
| `external_ep049_nepa_nontrans_postfix` | `runs/eccv_upmix_nepa_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1252 | 0.1613 | 0.3174 | non-transductive (`head_train_split=train_udf`) after Step0 |
| `external_ep049_mae_nontrans_postfix` | `runs/eccv_upmix_mae_s0/ckpt_ep049.pt` | `pointcloud_noray -> udf` | `eval` | 800 | 0.1020 | 0.1341 | 0.3353 | non-transductive (`head_train_split=train_udf`) after Step0 |

### Current readout (subset, single-seed)

- In this synced subset evaluation, MAE objective is stronger than NEPA on both UCPR and CPAC.
- After Step0/1 cleanup, MAE remains stronger than NEPA on CPAC non-transductive.
- `pointcloud_noray -> udfgrid` UCPR is near-saturated for both objectives in this setting; treat it as an easy pair and do not use it as the sole alignment evidence.
- This section is still a partial readout (`mesh->udf` only for UCPR, subset eval sizes), not the final ECCV main table.

### Done checklist (this cycle, no multi-seed)

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

### QA-token + dual-mask integration (this cycle)

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

### QA pretrain runs (completed)

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

### QA UCPR/CPAC eval commands (this cycle, single-seed)

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

### QA UCPR results (single-seed, complete)

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

### QA CPAC non-transductive results (single-seed, complete)

Protocol:

- `eval_split=eval`, `head_train_split=train_udf`, `head_train_backend=udfgrid`
- `n_shapes_head_train=15406`, `n_shapes_head_test=800`
- `context_backend=pointcloud_noray`, `n_context=256`, `n_query=256`

| Model | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| `qa_nodual_ep049` | 0.03072 | 0.04154 | 0.72231 |
| `qa_dualmask_ep049` | 0.02717 | 0.03623 | 0.72676 |

### Readout (this QA cycle, no multi-seed)

- Both QA pretrains finished to `ep049`.
- Hard-pair UCPR (`mesh -> udfgrid`, `mesh -> pointcloud_noray`) improved with dual-mask over no-dual.
- Easy-pair UCPR (`pointcloud_noray -> udfgrid`) is still higher for no-dual.
- CPAC non-trans improved with dual-mask (`MAE/RMSE` lower, `IoU@0.03` higher).
- Multi-seed was intentionally skipped in this cycle.

## QA Follow-Up (Pooling + Context Controls, Feb 15, 2026)

This block records the next-step validation after integrating:

- `retrieval_ucpr.py`: `--pooling {eos,mean_a,mean_zhat}`
- `completion_cpac_udf.py`: disjoint context/query, `context_mode_{train,test}`, `rep_source={h,zhat}`
- `causal_transformer.py` + `query_nepa.py` + `pretrain.py`: type-aware dual mask plumbing (`--dual_mask_type_aware`)
- wrapper updates:
  - `scripts/analysis/nepa3d_ucpr.sh`
  - `scripts/analysis/nepa3d_cpac_udf.sh`

### Commands used (this follow-up)

CPAC context controls (non-transductive, capped head-train set for quick cycle):

```bash
# baseline (normal context), dualmask
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_normal_h_htrain4k.json

# no-context / mismatch (test-time controls)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test none \
  --rep_source h \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_testnone_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test mismatch \
  --mismatch_shift 1 --rep_source h \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_testmismatch_h_htrain4k.json

# z_hat readout + nodual comparison
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source zhat \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_normal_zhat_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_nodual_s0_pc2udf_800_normal_h_htrain4k.json
```

UCPR pooling controls and ablation:

```bash
# hard-pair pooling sweep (dualmask)
for P in eos mean_a mean_zhat; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend udfgrid \
    --eval_seed 0 --eval_seed_gallery 999 \
    --max_files 1000 --pooling ${P} \
    --out_json results/ucpr_nepa_qa_dualmask_s0_mesh2udf_1k_indep_${P}.json
done

for P in eos mean_a mean_zhat; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --eval_seed 0 --eval_seed_gallery 999 \
    --max_files 1000 --pooling ${P} \
    --out_json results/ucpr_nepa_qa_dualmask_s0_mesh2pc_1k_indep_${P}.json
done

# nodual comparison at pooling=mean_a
for G in udfgrid pointcloud_noray; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_nodual_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend ${G} \
    --eval_seed 0 --eval_seed_gallery 999 \
    --max_files 1000 --pooling mean_a \
    --out_json results/ucpr_nepa_qa_nodual_s0_mesh2${G}_1k_indep_mean_a.json
done

# easy-pair diagnostic at pooling=mean_a (dualmask)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 \
  --max_files 1000 --pooling mean_a \
  --out_json results/ucpr_nepa_qa_dualmask_s0_pc2udf_1k_indep_mean_a.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 \
  --max_files 1000 --pooling mean_a --ablate_point_xyz \
  --out_json results/ucpr_nepa_qa_dualmask_s0_pc2udf_1k_indep_mean_a_ablate_xyz.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --query_backend pointcloud_noray --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 \
  --max_files 1000 --pooling mean_a --ablate_point_dist \
  --out_json results/ucpr_nepa_qa_dualmask_s0_pc2udf_1k_indep_mean_a_ablate_dist.json
```

### Results (this follow-up)

CPAC (`htrain4k`, non-transductive, `n_context=n_query=256`):

| Model / setting | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| dualmask, normal, `rep=h` | 0.02653 | 0.03531 | 0.72281 |
| dualmask, normal, `rep=zhat` | 0.02653 | 0.03530 | 0.72363 |
| dualmask, no-context test | 0.23832 | 0.30362 | 0.22710 |
| dualmask, mismatch-context test | 0.08508 | 0.12443 | 0.40283 |
| nodual, normal, `rep=h` | 0.02946 | 0.04008 | 0.72161 |

UCPR hard-pair pooling sweep (`eval_seed=0`, `eval_seed_gallery=999`, `max_files=1000`):

| Pair | Pooling | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| `mesh -> udfgrid` | `eos` | 0.006 | 0.021 | 0.041 | 0.02257 |
| `mesh -> udfgrid` | `mean_a` | 0.017 | 0.048 | 0.066 | 0.04143 |
| `mesh -> udfgrid` | `mean_zhat` | 0.003 | 0.019 | 0.031 | 0.01834 |
| `mesh -> pointcloud_noray` | `eos` | 0.004 | 0.021 | 0.043 | 0.02184 |
| `mesh -> pointcloud_noray` | `mean_a` | 0.022 | 0.047 | 0.070 | 0.04509 |
| `mesh -> pointcloud_noray` | `mean_zhat` | 0.002 | 0.019 | 0.031 | 0.01531 |

UCPR nodual comparison (`pooling=mean_a`):

| Pair | NoDual R@1/R@5/R@10/mAP | DualMask R@1/R@5/R@10/mAP |
|---|---|---|
| `mesh -> udfgrid` | `0.002 / 0.020 / 0.035 / 0.01749` | `0.017 / 0.048 / 0.066 / 0.04143` |
| `mesh -> pointcloud_noray` | `0.012 / 0.033 / 0.052 / 0.03070` | `0.022 / 0.047 / 0.070 / 0.04509` |

Easy-pair diagnostic (`pointcloud_noray -> udfgrid`, `pooling=mean_a`, dualmask):

| Setting | R@1 | R@5 | R@10 | mAP |
|---|---:|---:|---:|---:|
| base | 0.116 | 0.274 | 0.382 | 0.20344 |
| `ablate_point_xyz` | 0.011 | 0.044 | 0.076 | 0.03891 |
| `ablate_point_dist` | 0.129 | 0.299 | 0.423 | 0.22077 |

### Readout (follow-up)

- CPAC improvements with dual-mask remain after adding disjoint/no-context/mismatch controls.
- CPAC no-context and mismatch both degrade strongly vs normal, so context is being used.
- For hard-pair UCPR, `mean_a` pooling is clearly better than `eos` and `mean_zhat`.
- Under `mean_a` pooling, dual-mask outperforms no-dual on both hard pairs tested.
- Easy-pair (`pc->udf`) remains largely `xyz`-driven in this setup (`ablate_point_xyz` collapse, `ablate_point_dist` not harmful).

## CPAC NN-Copy + Grid Query (Feb 15, 2026)

This cycle integrates:

- `completion_cpac_udf.py`:
  - `--baseline {none,nn_copy}`
  - `--baseline_only 1`
  - `--query_source {pool,grid}`
- `scripts/analysis/nepa3d_cpac_udf.sh`:
  - `QUERY_SOURCE`, `BASELINE`, `BASELINE_ONLY`
- new qualitative script:
  - `nepa3d/analysis/qualitative_cpac_marching_cubes.py`

### Commands used

```bash
# CPAC (pool query) + NN-copy baseline in the same run
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_normal_h_htrain4k_with_nncopy.json

# CPAC (grid query) + NN-copy baseline
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_grid_h_htrain4k_with_nncopy.json

# Baseline-only mode smoke
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --n_context 256 --n_query 256 \
  --query_source grid \
  --baseline nn_copy --baseline_only 1 \
  --max_shapes 200 --eval_seed 0 \
  --out_json results/cpac_nepa_qa_dualmask_s0_grid_nncopy_baselineonly_200.json
```

Qualitative marching-cubes smoke:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.qualitative_cpac_marching_cubes \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 1000 \
  --n_context 256 --n_query_probe 256 \
  --grid_res 16 --mc_level 0.03 \
  --max_shapes 1 --shape_offset 0 \
  --out_dir results/qual_mc_smoke \
  --save_volumes 1 --save_png 0
```

### Results

CPAC vs NN-copy baseline:

| Query source | Probe MAE / RMSE / IoU@0.03 | NN-copy MAE / RMSE / IoU@0.03 |
|---|---|---|
| `pool` | `0.02653 / 0.03531 / 0.72281` | `0.06151 / 0.09474 / 0.70891` |
| `grid` | `0.03578 / 0.04501 / 0.29105` | `0.10402 / 0.13157 / 0.12963` |

Baseline-only smoke (`grid`, `max_shapes=200`):

- `MAE=0.10812`, `RMSE=0.13551`, `IoU@0.03=0.12926`

Qualitative MC smoke (`grid_res=16`, 1 shape):

- output dir: `results/qual_mc_smoke/000_1203825bf97bc3524722e1824a086fad`
- summary: `results/qual_mc_smoke/summary.json`
- sample grid metrics: `MAE=0.04685`, `RMSE=0.05718`, `IoU@level(0.03)=0.30526`

### Notes

- `scikit-image` is required for marching cubes (`pip install scikit-image`).
- `save_png=1` requires matplotlib; the script skips preview export if matplotlib is unavailable.

## MAE Parity + Eval-Seed Variance (Feb 15, 2026, follow-up)

This cycle focused on two items from feedback:

- run MAE under the same CPAC/UCPR settings used in the QA+dualmask analysis
- add small variance readout with `eval_seed={0,1,2}` (pretrain seed fixed)

### Commands used

CPAC (MAE, same controls as QA+dualmask):

```bash
# normal + NN-copy (pool query)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_mae_s0_pc2udf_800_normal_h_htrain4k_with_nncopy.json

# no-context / mismatch controls
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test none \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_mae_s0_pc2udf_800_testnone_h_htrain4k.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test mismatch \
  --mismatch_shift 1 \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_mae_s0_pc2udf_800_testmismatch_h_htrain4k.json

# grid query + NN-copy
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_mae_s0_pc2udf_800_grid_h_htrain4k_with_nncopy.json
```

Eval-seed variance (CPAC pool normal, both ckpts):

```bash
# seed=1
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 1 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_normal_h_htrain4k_seed1.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 1 \
  --out_json results/cpac_mae_s0_pc2udf_800_normal_h_htrain4k_seed1.json

# seed=2
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 2 \
  --out_json results/cpac_nepa_qa_dualmask_s0_pc2udf_800_normal_h_htrain4k_seed2.json

TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 2 \
  --out_json results/cpac_mae_s0_pc2udf_800_normal_h_htrain4k_seed2.json
```

UCPR hard-pair (`pooling=mean_a`, `eval_seed={0,1,2}`, `eval_seed_gallery=999`):

```bash
# mesh -> udfgrid
for S in 0 1 2; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend udfgrid \
    --eval_seed ${S} --eval_seed_gallery 999 --max_files 1000 \
    --pooling mean_a \
    --out_json results/ucpr_nepa_qa_dualmask_s0_mesh2udf_1k_indep_mean_a_seed${S}.json
done

for S in 0 1 2; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend udfgrid \
    --eval_seed ${S} --eval_seed_gallery 999 --max_files 1000 \
    --pooling mean_a \
    --out_json results/ucpr_mae_s0_mesh2udf_1k_indep_mean_a_seed${S}.json
done

# mesh -> pointcloud_noray
for S in 0 1 2; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --eval_seed ${S} --eval_seed_gallery 999 --max_files 1000 \
    --pooling mean_a \
    --out_json results/ucpr_nepa_qa_dualmask_s0_mesh2pc_1k_indep_mean_a_seed${S}.json
done

for S in 0 1 2; do
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_mae_s0/ckpt_ep049.pt \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --eval_seed ${S} --eval_seed_gallery 999 --max_files 1000 \
    --pooling mean_a \
    --out_json results/ucpr_mae_s0_mesh2pc_1k_indep_mean_a_seed${S}.json
done
```

### Results

CPAC MAE parity check (seed0 controls, non-transductive, `htrain4k`):

| Model / setting | MAE | RMSE | IoU@0.03 |
|---|---:|---:|---:|
| NEPA QA+dualmask, pool normal | 0.02653 | 0.03531 | 0.72281 |
| MAE, pool normal | 0.09226 | 0.12263 | 0.41660 |
| MAE, pool no-context | 0.89780 | 1.13171 | 0.41151 |
| MAE, pool mismatch-context | 0.10155 | 0.13629 | 0.40275 |
| NEPA QA+dualmask, grid normal | 0.03578 | 0.04501 | 0.29105 |
| MAE, grid normal | 0.10323 | 0.12999 | 0.05971 |

CPAC vs NN-copy baseline (seed0):

| Model / query source | Probe MAE / RMSE / IoU@0.03 | NN-copy MAE / RMSE / IoU@0.03 |
|---|---|---|
| NEPA QA+dualmask, pool | `0.02653 / 0.03531 / 0.72281` | `0.06151 / 0.09474 / 0.70891` |
| MAE, pool | `0.09226 / 0.12263 / 0.41660` | `0.06151 / 0.09474 / 0.70891` |
| NEPA QA+dualmask, grid | `0.03578 / 0.04501 / 0.29105` | `0.10402 / 0.13157 / 0.12963` |
| MAE, grid | `0.10323 / 0.12999 / 0.05971` | `0.10402 / 0.13157 / 0.12963` |

Eval-seed variance (`eval_seed=0,1,2`, pretrain seed fixed):

| Task | Model | mean +- std |
|---|---|---:|
| CPAC pool MAE | NEPA QA+dualmask | 0.02653 +- 0.00001 |
| CPAC pool MAE | MAE | 0.09289 +- 0.00044 |
| CPAC pool RMSE | NEPA QA+dualmask | 0.03528 +- 0.00009 |
| CPAC pool RMSE | MAE | 0.12331 +- 0.00049 |
| CPAC pool IoU@0.03 | NEPA QA+dualmask | 0.72218 +- 0.00151 |
| CPAC pool IoU@0.03 | MAE | 0.40953 +- 0.00504 |
| UCPR `mesh->udfgrid` R@1 (`mean_a`) | NEPA QA+dualmask | 0.01767 +- 0.00047 |
| UCPR `mesh->udfgrid` R@1 (`mean_a`) | MAE | 0.01900 +- 0.00082 |
| UCPR `mesh->udfgrid` mAP (`mean_a`) | NEPA QA+dualmask | 0.04112 +- 0.00027 |
| UCPR `mesh->udfgrid` mAP (`mean_a`) | MAE | 0.05010 +- 0.00072 |
| UCPR `mesh->pointcloud_noray` R@1 (`mean_a`) | NEPA QA+dualmask | 0.02233 +- 0.00047 |
| UCPR `mesh->pointcloud_noray` R@1 (`mean_a`) | MAE | 0.02067 +- 0.00125 |
| UCPR `mesh->pointcloud_noray` mAP (`mean_a`) | NEPA QA+dualmask | 0.04568 +- 0.00043 |
| UCPR `mesh->pointcloud_noray` mAP (`mean_a`) | MAE | 0.05446 +- 0.00089 |

### Readout (this follow-up)

- Same-setting MAE parity runs are now recorded (CPAC controls + UCPR hard-pair `mean_a`).
- CPAC remains strongly in favor of NEPA QA+dualmask under the current non-transductive probe setup.
- Hard-pair UCPR remains mixed: MAE is higher on `mesh->udfgrid`, while NEPA QA+dualmask is slightly higher on `mesh->pointcloud_noray` R@1.
- Eval-seed variance on these subsets is small relative to NEPA-vs-MAE CPAC gap.

## K-Plane / Tri-Plane Baseline Pilot (Feb 15, 2026)

This cycle adds a plane-factorized baseline track under the same ShapeNet-unpaired cache and UCPR/CPAC protocol.

Goal:

- compare `tri-plane(sum)` vs `k-plane(product)` under the same context/query sampling
- keep evaluation protocol aligned with existing UCPR/CPAC (`eval_seed=0`, `eval_seed_gallery=999`, `head_train_split=train_udf`, non-transductive)

New modules/scripts added in this cycle:

- model: `nepa3d/models/kplane.py`
- data: `nepa3d/data/kplane_dataset.py`
- pretrain: `nepa3d/train/pretrain_kplane.py`
- eval:
  - `nepa3d/analysis/retrieval_kplane.py`
  - `nepa3d/analysis/completion_cpac_kplane.py`
- wrappers:
  - `scripts/pretrain/nepa3d_kplane_pretrain.sh`
  - `scripts/analysis/nepa3d_kplane_ucpr.sh`
  - `scripts/analysis/nepa3d_kplane_cpac.sh`

### Commands used (fast pilot)

Pretrain (`5` epochs, `mix_num_samples=20000`, single seed):

```bash
# K-plane(product)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain_kplane \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 20000 --mix_seed 0 \
  --epochs 5 --batch 96 --num_workers 6 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --plane_type kplane --fusion product \
  --plane_resolutions 64 --plane_channels 32 --hidden_dim 128 \
  --save_every 1 --save_last 1 \
  --save_dir runs/eccv_kplane_product_s0_fast5 \
  --seed 0

# Tri-plane(sum)
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.train.pretrain_kplane \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 20000 --mix_seed 0 \
  --epochs 5 --batch 96 --num_workers 6 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --plane_type triplane --fusion sum \
  --plane_resolutions 64 --plane_channels 32 --hidden_dim 128 \
  --save_every 1 --save_last 1 \
  --save_dir runs/eccv_triplane_sum_s0_fast5 \
  --seed 0
```

UCPR hard pairs:

```bash
# mean_query pooling
for CKPT in \
  runs/eccv_kplane_product_s0_fast5/ckpt_ep004.pt \
  runs/eccv_triplane_sum_s0_fast5/ckpt_ep004.pt
do
  NAME=$(basename "$(dirname "${CKPT}")")
  .venv/bin/python -u -m nepa3d.analysis.retrieval_kplane \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT}" \
    --query_backend mesh --gallery_backend udfgrid \
    --eval_seed 0 --eval_seed_gallery 999 \
    --max_files 1000 --n_context 256 --n_query 256 \
    --pooling mean_query \
    --out_json "results/ucpr_${NAME}_mesh2udf_1k_mean_query.json"
done

# plane_gap pooling
for CKPT in \
  runs/eccv_kplane_product_s0_fast5/ckpt_ep004.pt \
  runs/eccv_triplane_sum_s0_fast5/ckpt_ep004.pt
do
  NAME=$(basename "$(dirname "${CKPT}")")
  .venv/bin/python -u -m nepa3d.analysis.retrieval_kplane \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT}" \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --eval_seed 0 --eval_seed_gallery 999 \
    --max_files 1000 --n_context 256 --n_query 256 \
    --pooling plane_gap \
    --out_json "results/ucpr_${NAME}_mesh2pc_1k_plane_gap.json"
done
```

CPAC (non-transductive, pool query, NN-copy baseline):

```bash
# K-plane(product)
.venv/bin/python -u -m nepa3d.analysis.completion_cpac_kplane \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_kplane_product_s0_fast5/ckpt_ep004.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 --query_source pool \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_kplane_product_fast5_pc2udf_800_pool_htrain4k.json

# Tri-plane(sum)
.venv/bin/python -u -m nepa3d.analysis.completion_cpac_kplane \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/eccv_triplane_sum_s0_fast5/ckpt_ep004.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid \
  --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 \
  --disjoint_context_query 1 --query_source pool \
  --baseline nn_copy \
  --max_shapes 800 --head_train_ratio 0.2 \
  --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  --out_json results/cpac_triplane_sum_fast5_pc2udf_800_pool_htrain4k.json
```

### Results (fast pilot, single-seed)

UCPR hard pairs:

| Pair | Pooling | K-plane(product) R@1/R@5/R@10/mAP | Tri-plane(sum) R@1/R@5/R@10/mAP |
|---|---|---|---|
| `mesh -> udfgrid` | `mean_query` | `0.006 / 0.019 / 0.041 / 0.02506` | `0.013 / 0.037 / 0.069 / 0.03652` |
| `mesh -> pointcloud_noray` | `mean_query` | `0.010 / 0.030 / 0.063 / 0.03224` | `0.012 / 0.048 / 0.086 / 0.04054` |
| `mesh -> udfgrid` | `plane_gap` | `0.012 / 0.031 / 0.054 / 0.03281` | `0.012 / 0.039 / 0.065 / 0.03622` |
| `mesh -> pointcloud_noray` | `plane_gap` | `0.016 / 0.057 / 0.088 / 0.04649` | `0.023 / 0.073 / 0.110 / 0.05906` |

CPAC-UDF (non-transductive, `query_source=pool`, with NN-copy baseline):

| Model | MAE | RMSE | IoU@0.03 | NN-copy MAE/RMSE/IoU@0.03 |
|---|---:|---:|---:|---|
| K-plane(product) | 0.17990 | 0.24980 | 0.56906 | `0.06151 / 0.09474 / 0.70891` |
| Tri-plane(sum) | 0.10499 | 0.16542 | 0.68938 | `0.06151 / 0.09474 / 0.70891` |

UCPR ablation (`mesh -> udfgrid`, `pooling=mean_query`):

| Model | Base R@1/mAP | `ablate_query_xyz` R@1/mAP | `ablate_context_dist` R@1/mAP |
|---|---|---|---|
| K-plane(product) | `0.006 / 0.02506` | `0.366 / 0.37147` | `0.012 / 0.03896` |
| Tri-plane(sum) | `0.013 / 0.03652` | `0.003 / 0.01537` | `0.012 / 0.04258` |

### Readout (fast pilot)

- Baseline stack (train/eval) is now integrated and runs on the same cache/protocol as NEPA UCPR/CPAC.
- Under this `fast5` budget, tri-plane(sum) is stronger than k-plane(product) on both hard-pair UCPR and CPAC.
- Both plane baselines remain below current NEPA QA+dualmask CPAC numbers in this repo.
- The `k-plane(product)` `ablate_query_xyz` spike on `mesh->udfgrid` is likely a shortcut artifact and should not be used as main evidence without additional controls.
- These are pilot numbers (`epochs=5`, single seed), not final table settings.

## K-Plane / Tri-Plane Full Run (e50, Feb 16, 2026)

This block records the full-budget run (`epochs=50`, `mix_num_samples=200000`, seed=`0`) and the auto-eval chain outputs.

Training checkpoints:

- `runs/eccv_kplane_product_s0/ckpt_ep049.pt`
- `runs/eccv_triplane_sum_s0/ckpt_ep049.pt`

Training logs:

- `logs/pretrain/eccv_kplane_baseline/kplane_product_s0_bs96_e50.log`
- `logs/pretrain/eccv_kplane_baseline/triplane_sum_s0_bs96_e50.log`

### Commands used

Pretrain (local, 2 GPUs):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain_kplane \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --batch 96 --epochs 50 --lr 3e-4 --weight_decay 0.05 \
  --num_workers 6 --n_context 256 --n_query 256 \
  --query_source pool --target_mode backend --disjoint_context_query 1 \
  --plane_type kplane --fusion product \
  --plane_resolutions 64 --plane_channels 32 --hidden_dim 128 \
  --voxel_grid 64 --voxel_dilate 1 --voxel_max_steps 0 \
  --save_dir runs/eccv_kplane_product_s0 \
  --save_every 1 --save_last 1 --auto_resume 1 --seed 0

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u -m nepa3d.train.pretrain_kplane \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 200000 --mix_seed 0 \
  --batch 96 --epochs 50 --lr 3e-4 --weight_decay 0.05 \
  --num_workers 6 --n_context 256 --n_query 256 \
  --query_source pool --target_mode backend --disjoint_context_query 1 \
  --plane_type triplane --fusion sum \
  --plane_resolutions 64 --plane_channels 32 --hidden_dim 128 \
  --voxel_grid 64 --voxel_dilate 1 --voxel_max_steps 0 \
  --save_dir runs/eccv_triplane_sum_s0 \
  --save_every 1 --save_last 1 --auto_resume 1 --seed 0
```

Auto evaluation chain (wait for `ckpt_ep049.pt`, then UCPR/CPAC):

```bash
bash scripts/analysis/launch_kplane_full_chain_local.sh
```

Chain scripts:

- `scripts/analysis/run_kplane_full_chain_local.sh`
- `scripts/analysis/launch_kplane_full_chain_local.sh`

### Results (e50, single-seed)

UCPR hard pairs:

| Pair | Pooling | K-plane(product) R@1/R@5/R@10/mAP | Tri-plane(sum) R@1/R@5/R@10/mAP |
|---|---|---|---|
| `mesh -> udfgrid` | `mean_query` | `0.003 / 0.019 / 0.030 / 0.02029` | `0.049 / 0.140 / 0.214 / 0.11009` |
| `mesh -> pointcloud_noray` | `mean_query` | `0.007 / 0.023 / 0.053 / 0.02678` | `0.051 / 0.169 / 0.249 / 0.11707` |
| `mesh -> udfgrid` | `plane_gap` | `0.012 / 0.032 / 0.055 / 0.03146` | `0.031 / 0.102 / 0.166 / 0.07865` |
| `mesh -> pointcloud_noray` | `plane_gap` | `0.020 / 0.046 / 0.079 / 0.04400` | `0.060 / 0.146 / 0.201 / 0.11397` |

UCPR ablation (`mesh -> udfgrid`, `pooling=mean_query`):

| Model | Base R@1/mAP | `ablate_query_xyz` R@1/mAP | `ablate_context_dist` R@1/mAP |
|---|---|---|---|
| K-plane(product) | `0.003 / 0.02029` | `0.364 / 0.36868` | `0.010 / 0.02969` |
| Tri-plane(sum) | `0.049 / 0.11009` | `0.007 / 0.02146` | `0.049 / 0.11889` |

CPAC-UDF (non-transductive controls, `head_train_split=train_udf`, `head_train_max_shapes=4000`):

| Model / setting | MAE | RMSE | IoU@0.03 | Baseline (NN-copy) |
|---|---:|---:|---:|---|
| K-plane(product), `pool normal` | 0.17604 | 0.24831 | 0.61277 | `0.06151 / 0.09474 / 0.70891` |
| Tri-plane(sum), `pool normal` | 0.09774 | 0.15701 | 0.75702 | `0.06151 / 0.09474 / 0.70891` |
| K-plane(product), `pool testnone` | 0.28491 | 0.31325 | 0.00000 | - |
| Tri-plane(sum), `pool testnone` | 0.37194 | 0.41228 | 0.00000 | - |
| K-plane(product), `pool testmismatch` | 0.22691 | 0.27991 | 0.30185 | - |
| Tri-plane(sum), `pool testmismatch` | 0.16289 | 0.22628 | 0.39990 | - |
| K-plane(product), `grid normal` | 0.20849 | 0.25611 | 0.17028 | `0.10402 / 0.13157 / 0.12963` |
| Tri-plane(sum), `grid normal` | 0.15981 | 0.20480 | 0.11956 | `0.10402 / 0.13157 / 0.12963` |

### Readout (e50)

- Full-budget run confirms tri-plane(sum) > k-plane(product) on hard-pair UCPR in this setup.
- For CPAC pool-normal, tri-plane(sum) is substantially better than k-plane(product), but both remain behind the current NN-copy baseline on MAE/RMSE.
- CPAC controls (`testnone`, `testmismatch`) degrade strongly for both models, so context dependence is present.
- The `k-plane(product)` `ablate_query_xyz` spike persists at e50 and should be treated as a likely shortcut artifact, not main evidence.

### Tie-Aware UCPR Fix + Sanity (Feb 16, 2026, follow-up)

This follow-up applies tie-aware rank handling in both retrieval evaluators:

- `nepa3d/analysis/retrieval_ucpr.py`
- `nepa3d/analysis/retrieval_kplane.py`

Changes:

- rank computation switched from `count(score > corr)` to tie-aware sorting
- deterministic tiny noise (`tie_break_eps`, default `1e-6`) is added before ranking
- sanity mode added: `--sanity_constant_embed`
- `retrieval_kplane.py` also adds context controls:
  - `--context_mode_query {normal,none,mismatch}`
  - `--context_mode_gallery {normal,none,mismatch}`
  - `--disjoint_context_query`, `--mismatch_shift_{query,gallery}`

Wrapper updates:

- `scripts/analysis/nepa3d_ucpr.sh`
- `scripts/analysis/nepa3d_kplane_ucpr.sh`

Sanity check (`max_files=200`, constant embeddings):

| Evaluator | Pair | R@1 | expected random R@1 |
|---|---|---:|---:|
| `retrieval_kplane.py` | `mesh -> udfgrid` | 0.000 | 0.005 |
| `retrieval_ucpr.py` | `mesh -> udfgrid` | 0.000 | 0.005 |

Key re-readout after tie-fix (`max_files=1000`, `eval_seed=0`, `eval_seed_gallery=999`):

| Pair / setting | K-plane(product) R@1/mAP | Tri-plane(sum) R@1/mAP |
|---|---|---|
| `mesh -> udfgrid`, `mean_query` | `0.002 / 0.01992` | `0.050 / 0.11058` |
| `mesh -> pointcloud_noray`, `mean_query` | `0.006 / 0.02666` | `0.051 / 0.11700` |
| `mesh -> udfgrid`, `ablate_query_xyz` | `0.000 / 0.00774` | `0.006 / 0.02066` |
| `mesh -> udfgrid`, `ablate_context_dist` | `0.008 / 0.02442` | `0.049 / 0.11888` |

Context controls on UCPR (`mesh -> udfgrid`, `mean_query`):

| Model | normal R@1/mAP | `context_mode=none` R@1/mAP | `query=mismatch, gallery=normal` R@1/mAP |
|---|---|---|---|
| K-plane(product) | `0.002 / 0.01992` | `0.002 / 0.00817` | `0.001 / 0.00803` |
| Tri-plane(sum) | `0.050 / 0.11058` | `0.002 / 0.00817` | `0.002 / 0.01412` |

Interpretation update:

- The previous product `ablate_query_xyz` spike (`R@1=0.364`) is removed by tie-aware ranking and was an evaluation artifact.
- Under corrected ranking, both models collapse to near-random under `context_mode=none` and drop strongly under query-side mismatch.
- Main ordering remains unchanged: tri-plane(sum) > k-plane(product) on hard-pair UCPR in this setup.

## K-plane Product Robustness Recheck + c64 Follow-Up (Feb 17, 2026)

Objective:

- close the remaining uncertainty around product-model UCPR diagnostics before the next training sweep
- run the minimum additional product pretrains suggested in feedback (`c64`, `target_mode=udf`)

Evaluator update (this cycle):

- `nepa3d/analysis/retrieval_kplane.py` adds:
  - `--shuffle_gallery {0,1}`
  - `--gallery_shuffle_seed`
- wrapper passthrough added in `scripts/analysis/nepa3d_kplane_ucpr.sh`:
  - `SHUFFLE_GALLERY`
  - `GALLERY_SHUFFLE_SEED`

### A) Constant-embed sanity (`max_files=1000`, `pooling=plane_gap`)

| Setting | R@1 | R@5 | R@10 | mAP | expected random R@1 |
|---|---:|---:|---:|---:|---:|
| `sanity_constant_embed=1` | 0.002 | 0.006 | 0.012 | 0.00821 | 0.001 |

Artifact:

- `results/_sanity_const_kplane_product_mesh2udf_1k_plane_gap.json`

### B/C) Gallery-seed + gallery-shuffle robustness (`max_files=1000`, `pooling=plane_gap`)

| `eval_seed_gallery` | `shuffle_gallery` | R@1 | mAP |
|---:|---:|---:|---:|
| 999 | 0 | 0.012 | 0.03045 |
| 999 | 1 | 0.009 | 0.03022 |
| 2024 | 0 | 0.006 | 0.02622 |
| 2024 | 1 | 0.004 | 0.02351 |
| 4242 | 0 | 0.007 | 0.02658 |
| 4242 | 1 | 0.011 | 0.02937 |

Artifacts:

- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed999.json`
- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed999_shuf.json`
- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed2024.json`
- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed2024_shuf.json`
- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed4242.json`
- `results/_diag_kplane_product_mesh2udf_1k_plane_gap_gseed4242_shuf.json`

### D) `ablate_query_xyz` recheck (`mesh->udfgrid`, seed0)

`max_files=5185` (full eval split):

| Pooling | Base R@1/mAP | `ablate_query_xyz` R@1/mAP |
|---|---|---|
| `mean_query` | `0.00077 / 0.00605` | `0.00019 / 0.00234` |
| `plane_gap` | `0.00193 / 0.00899` | `0.00193 / 0.00899` |

`max_files=1000` (historical setting recheck):

| Pooling | Base R@1/mAP | `ablate_query_xyz` R@1/mAP |
|---|---|---|
| `mean_query` | `0.002 / 0.01992` | `0.000 / 0.00774` |
| `plane_gap` | `0.012 / 0.03045` | `0.012 / 0.03046` |

Artifacts:

- `results/_diag_kplane_product_mesh2udf_eval5185_mean_query.json`
- `results/_diag_kplane_product_mesh2udf_eval5185_mean_query_ablate_qxyz.json`
- `results/_diag_kplane_product_mesh2udf_eval5185_plane_gap.json`
- `results/_diag_kplane_product_mesh2udf_eval5185_plane_gap_ablate_qxyz.json`
- `results/_diag_recheck_kplane_product_mesh2udf_1k_mean_query.json`
- `results/_diag_recheck_kplane_product_mesh2udf_1k_mean_query_ablate_qxyz.json`
- `results/_diag_recheck_kplane_product_mesh2udf_1k_plane_gap.json`
- `results/_diag_recheck_kplane_product_mesh2udf_1k_plane_gap_ablate_qxyz.json`

Readout:

- The old product `ablate_query_xyz` jump does not reappear in the current evaluator path.
- `mean_query` degrades under `ablate_query_xyz`, while `plane_gap` remains effectively unchanged.
- For product fusion, `plane_gap` remains the safer retrieval descriptor for diagnostics.

### K-plane Product c64 Follow-Up (completed, Feb 17, 2026)

Completed checkpoints:

- `runs/eccv_kplane_product_c64_s0/ckpt_ep049.pt` (`target_mode=backend`, `plane_channels=64`)
- `runs/eccv_kplane_product_udf_c64_s0/ckpt_ep049.pt` (`target_mode=udf`, `plane_channels=64`)

Logs:

- `logs/pretrain/eccv_kplane_baseline/kplane_product_c64_s0_bs96_e50.log`
- `logs/pretrain/eccv_kplane_baseline/kplane_product_udf_c64_s0_bs96_e50.log`

Auto-eval chain:

- runner: `scripts/analysis/run_kplane_sum_chain_local.sh` (retargeted via env)
- pipeline log: `logs/analysis/kplane_product_c64_chain/pipeline.log`
- completion line confirmed: `all evaluations finished`

UCPR hard pairs (`pooling=mean_query`, tie-aware, `max_files=1000`):

| Pair | product s0 R@1/R@5/R@10/mAP | product c64 R@1/R@5/R@10/mAP | product udf-c64 R@1/R@5/R@10/mAP |
|---|---|---|---|
| `mesh -> udfgrid` | `0.002 / 0.019 / 0.029 / 0.01992` | `0.007 / 0.022 / 0.040 / 0.02321` | `0.003 / 0.021 / 0.040 / 0.02127` |
| `mesh -> pointcloud_noray` | `0.006 / 0.025 / 0.050 / 0.02666` | `0.008 / 0.027 / 0.052 / 0.02766` | `0.010 / 0.024 / 0.043 / 0.02672` |

CPAC-UDF (`head_train_split=train_udf`, `head_train_max_shapes=4000`, `max_shapes=800`):

| Model / setting | MAE | RMSE | IoU@0.03 | Baseline (NN-copy) |
|---|---:|---:|---:|---|
| product s0, `pool normal` | 0.17604 | 0.24831 | 0.61277 | `0.06151 / 0.09474 / 0.70891` |
| product c64, `pool normal` | 0.17617 | 0.24830 | 0.61118 | `0.06151 / 0.09474 / 0.70891` |
| product udf-c64, `pool normal` | 0.17595 | 0.24832 | 0.61409 | `0.06151 / 0.09474 / 0.70891` |
| product s0, `grid normal` | 0.20849 | 0.25611 | 0.17028 | `0.10402 / 0.13157 / 0.12963` |
| product c64, `grid normal` | 0.20838 | 0.25607 | 0.15499 | `0.10402 / 0.13157 / 0.12963` |
| product udf-c64, `grid normal` | 0.20852 | 0.25612 | 0.18067 | `0.10402 / 0.13157 / 0.12963` |

CPAC controls:

- all three product variants show `pool testnone` collapse (`IoU@0.03 = 0.00000`)
- `pool testmismatch` remains degraded (around `IoU@0.03 ~ 0.297 to 0.302`)

Readout:

- `plane_channels=64` improves product-model UCPR slightly on hard pairs.
- CPAC `pool normal` remains effectively unchanged and still below NN-copy.
- CPAC `grid normal` shows a small gain only for `target_mode=udf` (`IoU@0.03: 0.17028 -> 0.18067`), but error metrics stay almost unchanged.

## K-plane Fusion Sweep (completed, Feb 16, 2026)

Objective:

- follow-up from feedback item `(2)` and compare `kplane + sum` variants against existing `kplane(product)` / `triplane(sum)` under the same protocol

Completed checkpoints:

- base: `runs/eccv_kplane_sum_s0` (`fusion=sum`, `plane_res=64`, `plane_channels=32`, `hidden_dim=128`)
- large: `runs/eccv_kplane_sum_large_s0` (`fusion=sum`, `plane_res=128`, `plane_channels=64`, `hidden_dim=256`)

Logs:

- `logs/pretrain/eccv_kplane_baseline/kplane_sum_s0_bs96_e50.log`
- `logs/pretrain/eccv_kplane_baseline/kplane_sum_large_s0_bs96_e50.log`

Auto evaluation chain (wait for `ckpt_ep049.pt`, then run tie-aware UCPR/CPAC pack):

```bash
bash scripts/analysis/launch_kplane_sum_chain_local.sh
```

Smoke validation (existing checkpoints, reduced `max_files/max_shapes`) was run to verify chain wiring before waiting for new checkpoints:

- `scripts/analysis/run_kplane_sum_chain_local.sh`
- both model branches completed and emitted `_tmp_chain_smoke_*` JSON successfully

Final eval execution:

- `bash scripts/analysis/run_kplane_sum_chain_local.sh`
- tie-aware UCPR + CPAC pack completed for both `kplane_sum_s0_e50` and `kplane_sum_large_s0_e50`

### Results (e50, single-seed)

UCPR hard pairs (`pooling=mean_query`, tie-aware):

| Pair | K-plane(sum, base) R@1/R@5/R@10/mAP | K-plane(sum, large) R@1/R@5/R@10/mAP |
|---|---|---|
| `mesh -> udfgrid` | `0.036 / 0.130 / 0.187 / 0.09429` | `0.030 / 0.079 / 0.128 / 0.06659` |
| `mesh -> pointcloud_noray` | `0.040 / 0.146 / 0.217 / 0.09963` | `0.040 / 0.125 / 0.203 / 0.09308` |

CPAC-UDF (`head_train_split=train_udf`, `head_train_max_shapes=4000`):

| Model / setting | MAE | RMSE | IoU@0.03 | Baseline (NN-copy) |
|---|---:|---:|---:|---|
| K-plane(sum, base), `pool normal` | 0.09776 | 0.15700 | 0.75630 | `0.06151 / 0.09474 / 0.70891` |
| K-plane(sum, large), `pool normal` | 0.14229 | 0.21260 | 0.52054 | `0.06151 / 0.09474 / 0.70891` |
| K-plane(sum, base), `pool testnone` | 0.37192 | 0.41225 | 0.00000 | - |
| K-plane(sum, large), `pool testnone` | 0.31261 | 0.34175 | 0.00000 | - |
| K-plane(sum, base), `pool testmismatch` | 0.16285 | 0.22623 | 0.39971 | - |
| K-plane(sum, large), `pool testmismatch` | 0.20377 | 0.26490 | 0.23201 | - |
| K-plane(sum, base), `grid normal` | 0.15984 | 0.20484 | 0.12538 | `0.10402 / 0.13157 / 0.12963` |
| K-plane(sum, large), `grid normal` | 0.19829 | 0.24462 | 0.06178 | `0.10402 / 0.13157 / 0.12963` |

### Readout (fusion sweep)

- `kplane + sum` clearly outperforms `kplane + product` on hard-pair UCPR in this setup.
- `kplane + sum (base)` is stronger than `kplane + sum (large)` across UCPR and CPAC.
- Against previous baselines:
  - UCPR: `kplane + sum (base)` is below `tri-plane(sum)` but much closer than `kplane(product)`.
  - CPAC pool-normal: `kplane + sum (base)` is effectively on par with `tri-plane(sum)` (`MAE/RMSE/IoU` nearly equal).
- CPAC controls remain valid: both `testnone` and `testmismatch` degrade strongly from `pool normal`, confirming context dependence.

Chain artifacts:

- pipeline log: `logs/analysis/kplane_sum_chain/pipeline.log`
- note: this log only captures launcher-side wait start; final completion was confirmed from `results/*kplane_sum*_e50*.json` timestamps and contents
- expected result prefixes:
  - `results/ucpr_kplane_sum_s0_e50_*_tiefix.json`
  - `results/ucpr_kplane_sum_large_s0_e50_*_tiefix.json`
  - `results/cpac_kplane_sum_s0_e50_*.json`
  - `results/cpac_kplane_sum_large_s0_e50_*.json`

## K-plane `rg_product` Integration (Feb 17, 2026)

Implemented minimal rank-grouped product fusion for tri/k-plane baseline:

- `nepa3d/models/kplane.py`
  - new fusion: `rg_product`
  - new config: `product_rank_groups`, `product_group_reduce`
  - rank-grouped product reduction in both query features and plane-global descriptor paths
  - backward-compatible ckpt load defaults for old checkpoints
- `nepa3d/train/pretrain_kplane.py`
  - CLI: `--fusion rg_product`, `--product_rank_groups`, `--product_group_reduce`
  - saved `kplane_cfg` extended with new fields
- `scripts/pretrain/nepa3d_kplane_pretrain.sh`
  - forwards `PRODUCT_RANK_GROUPS`, `PRODUCT_GROUP_REDUCE`

### Smoke commands

```bash
# smoke pretrain
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.train.pretrain_kplane \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples 256 --mix_seed 0 \
  --batch 8 --epochs 1 --lr 3e-4 --weight_decay 0.05 \
  --num_workers 2 --n_context 64 --n_query 64 \
  --query_source pool --target_mode backend --disjoint_context_query 1 \
  --plane_type kplane --fusion rg_product \
  --product_rank_groups 16 --product_group_reduce mean \
  --plane_resolutions 64 --plane_channels 64 --hidden_dim 128 \
  --voxel_grid 64 --voxel_dilate 1 --voxel_max_steps 0 \
  --save_dir runs/_tmp_kplane_rg_product_smoke \
  --save_every 1 --save_last 1 --auto_resume 0 --seed 0

# smoke retrieval
TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m nepa3d.analysis.retrieval_kplane \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt runs/_tmp_kplane_rg_product_smoke/ckpt_ep000.pt \
  --query_backend mesh --gallery_backend udfgrid \
  --eval_seed 0 --eval_seed_gallery 999 \
  --max_files 200 --n_context 64 --n_query 64 \
  --pooling mean_query \
  --out_json results/ucpr_tmp_kplane_rgprod_smoke_mesh2udf_200_mean_query.json
```

### Smoke result

| Model | Pair | R@1 | R@5 | R@10 | mAP |
|---|---|---:|---:|---:|---:|
| `rg_product` smoke (`ep000`) | `mesh -> udfgrid` | 0.005 | 0.025 | 0.070 | 0.03174 |

Readout:

- `rg_product` path compiles, trains, checkpoints, and evaluates end-to-end.
- Existing checkpoints remain loadable via defaulted `kplane_cfg` fields.
- Next step is full e50/eval-pack run with `rg_product` (e.g., `plane_channels=64`, `product_rank_groups=16`) for fair comparison against current `sum/product` lines.

### `rg_product` e50 eval pack (completed, Feb 17, 2026)

Evaluated checkpoint:

- `runs/eccv_kplane_rg_product_c64_r16_mean_s0/ckpt_ep049.pt`

Execution log:

- `logs/analysis/kplane_rg_product_chain/kplane_rg_product_c64_r16_mean_s0_e50.log`

Result JSON:

- `results/ucpr_kplane_rg_product_c64_r16_mean_s0_e50_mesh2udf_1k_mean_query_tiefix.json`
- `results/ucpr_kplane_rg_product_c64_r16_mean_s0_e50_mesh2pc_1k_mean_query_tiefix.json`
- `results/ucpr_kplane_rg_product_c64_r16_mean_s0_e50_mesh2udf_1k_plane_gap_tiefix.json`
- `results/ucpr_kplane_rg_product_c64_r16_mean_s0_e50_mesh2pc_1k_plane_gap_tiefix.json`
- `results/cpac_kplane_rg_product_c64_r16_mean_s0_e50_pc2udf_800_pool_htrain4k_with_nncopy.json`
- `results/cpac_kplane_rg_product_c64_r16_mean_s0_e50_pc2udf_800_testnone_h_htrain4k.json`
- `results/cpac_kplane_rg_product_c64_r16_mean_s0_e50_pc2udf_800_testmismatch_h_htrain4k.json`
- `results/cpac_kplane_rg_product_c64_r16_mean_s0_e50_pc2udf_800_grid_h_htrain4k_with_nncopy.json`

UCPR hard pairs (`max_files=1000`, tie-aware):

| Pair | Mean-query R@1/mAP | Plane-gap R@1/mAP |
|---|---|---|
| `mesh -> udfgrid` | `0.003 / 0.02077` | `0.012 / 0.03224` |
| `mesh -> pointcloud_noray` | `0.006 / 0.02556` | `0.006 / 0.03320` |

CPAC-UDF (`max_shapes=800`, `htrain4k`):

| Setting | MAE | RMSE | IoU@0.03 | NN-copy IoU@0.03 |
|---|---:|---:|---:|---:|
| `pool normal` | 0.17612 | 0.24839 | 0.60960 | 0.70891 |
| `pool testnone` | 0.28488 | 0.31323 | 0.00000 | - |
| `pool testmismatch` | 0.22709 | 0.27996 | 0.29611 | - |
| `grid normal` | 0.20857 | 0.25618 | 0.16240 | 0.12963 |

Readout:

- `rg_product` retrieval is better with `plane_gap` than with `mean_query` on both hard pairs.
- CPAC `pool normal` remains below NN-copy, similar to existing `kplane(product)` behavior.
- CPAC controls are valid (`testnone` collapse, `testmismatch` degraded), confirming context dependence.

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

| Pair | R@1 | R@5 | R@10 | mAP |
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

| Pair | R@1 | R@5 | R@10 | mAP |
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

| Model | Pair | R@1 | mAP |
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
