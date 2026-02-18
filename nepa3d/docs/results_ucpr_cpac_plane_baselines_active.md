# UCPR/CPAC Plane Baselines Active

This file isolates plane-baseline tracks from the main UCPR/CPAC log.

- Scope: `tri-plane` / `k-plane` families and related robustness checks
- Main track (NEPA/MAE): `results_ucpr_cpac_active.md`
- Source archive: `results_ucpr_cpac_active_mixed_archive.md`

## Protocol guardrails (comparison safety)

- UCPR in this baseline track is also single-positive retrieval; historical `mAP` values are equivalent to MRR (`1 / (rank+1)`).
- Distinguish seed-coupled vs independent UCPR sampling: `eval_seed_gallery` omitted/`-1` means seed-coupled fallback; explicit values (e.g. `999`) are independent sampling.
- For UCPR rows, keep protocol signature visible: `pooling`, `eval_seed`, `eval_seed_gallery`, `max_files`, `tie_break_eps`.
- For CPAC rows, keep protocol signature visible: `head_train_split`, `head_train_backend`, `head_train_max_shapes`, `n_context/n_query`, `query_source`, `disjoint_context_query`.

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

| Pair | Pooling | K-plane(product) R@1/R@5/R@10/MRR | Tri-plane(sum) R@1/R@5/R@10/MRR |
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

| Model | Base R@1/MRR | `ablate_query_xyz` R@1/MRR | `ablate_context_dist` R@1/MRR |
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

| Pair | Pooling | K-plane(product) R@1/R@5/R@10/MRR | Tri-plane(sum) R@1/R@5/R@10/MRR |
|---|---|---|---|
| `mesh -> udfgrid` | `mean_query` | `0.003 / 0.019 / 0.030 / 0.02029` | `0.049 / 0.140 / 0.214 / 0.11009` |
| `mesh -> pointcloud_noray` | `mean_query` | `0.007 / 0.023 / 0.053 / 0.02678` | `0.051 / 0.169 / 0.249 / 0.11707` |
| `mesh -> udfgrid` | `plane_gap` | `0.012 / 0.032 / 0.055 / 0.03146` | `0.031 / 0.102 / 0.166 / 0.07865` |
| `mesh -> pointcloud_noray` | `plane_gap` | `0.020 / 0.046 / 0.079 / 0.04400` | `0.060 / 0.146 / 0.201 / 0.11397` |

UCPR ablation (`mesh -> udfgrid`, `pooling=mean_query`):

| Model | Base R@1/MRR | `ablate_query_xyz` R@1/MRR | `ablate_context_dist` R@1/MRR |
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

| Pair / setting | K-plane(product) R@1/MRR | Tri-plane(sum) R@1/MRR |
|---|---|---|
| `mesh -> udfgrid`, `mean_query` | `0.002 / 0.01992` | `0.050 / 0.11058` |
| `mesh -> pointcloud_noray`, `mean_query` | `0.006 / 0.02666` | `0.051 / 0.11700` |
| `mesh -> udfgrid`, `ablate_query_xyz` | `0.000 / 0.00774` | `0.006 / 0.02066` |
| `mesh -> udfgrid`, `ablate_context_dist` | `0.008 / 0.02442` | `0.049 / 0.11888` |

Context controls on UCPR (`mesh -> udfgrid`, `mean_query`):

| Model | normal R@1/MRR | `context_mode=none` R@1/MRR | `query=mismatch, gallery=normal` R@1/MRR |
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

| Setting | R@1 | R@5 | R@10 | MRR (=single-positive mAP) | expected random R@1 |
|---|---:|---:|---:|---:|---:|
| `sanity_constant_embed=1` | 0.002 | 0.006 | 0.012 | 0.00821 | 0.001 |

Artifact:

- `results/_sanity_const_kplane_product_mesh2udf_1k_plane_gap.json`

### B/C) Gallery-seed + gallery-shuffle robustness (`max_files=1000`, `pooling=plane_gap`)

| `eval_seed_gallery` | `shuffle_gallery` | R@1 | MRR (=single-positive mAP) |
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

| Pooling | Base R@1/MRR | `ablate_query_xyz` R@1/MRR |
|---|---|---|
| `mean_query` | `0.00077 / 0.00605` | `0.00019 / 0.00234` |
| `plane_gap` | `0.00193 / 0.00899` | `0.00193 / 0.00899` |

`max_files=1000` (historical setting recheck):

| Pooling | Base R@1/MRR | `ablate_query_xyz` R@1/MRR |
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

| Pair | product s0 R@1/R@5/R@10/MRR | product c64 R@1/R@5/R@10/MRR | product udf-c64 R@1/R@5/R@10/MRR |
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

| Pair | K-plane(sum, base) R@1/R@5/R@10/MRR | K-plane(sum, large) R@1/R@5/R@10/MRR |
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

| Model | Pair | R@1 | R@5 | R@10 | MRR (=single-positive mAP) |
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

| Pair | Mean-query R@1/MRR | Plane-gap R@1/MRR |
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
