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
