# 1024-Point Pretrain A/B/C/D (Active Plan)

Last updated: 2026-02-23

## 1. Scope

This document defines the active pretraining plan for:

- 1024-point base setting
- A/B/C/D ablation matrix
- multi-node PBS launch
- coarse-to-fine verification (`point_order_mode=fps`)

## 2. Code Changes Included

The following are now wired end-to-end for pretrain:

- `pretrain.py` CLI now supports:
  - `--pt_xyz_key`, `--pt_dist_key`, `--ablate_point_dist`
  - `--pt_sample_mode_train`, `--pt_fps_key`, `--pt_rfps_m`
  - `--point_order_mode {morton,fps,random}`
- Mixed pretrain builder now propagates those args to each dataset.
- Dataset now forwards `point_order_mode` to tokenizer.
- Tokenizer now supports explicit point ordering after sampling:
  - `morton` (legacy)
  - `fps` (keep sampled order)
  - `random` (shuffle)
- Pretrain shell wrappers now pass these knobs:
  - `scripts/pretrain/nepa3d_pretrain.sh`
  - `scripts/pretrain/nepa3d_pretrain_pointcloud.sh`

## 3. Run Matrix (Fixed)

All runs use `n_point=1024`.

### Run A (Main / CoT full)

- `mix_config`: `nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml`
- `n_ray=1024`, `qa_tokens=1`, `max_len=4500`
- `pt_xyz_key=pt_xyz_pool`, `ablate_point_dist=0`
- `pt_sample_mode_train=fps`, `point_order_mode=fps`

### Run B (SOTA-fair XYZ-only)

- `mix_config`: `nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml`
- `n_ray=0`, `qa_tokens=0`, `max_len=2500`
- `pt_xyz_key=pc_xyz`, `ablate_point_dist=1`
- `pt_sample_mode_train=fps`, `point_order_mode=morton`

### Run C (Order ablation vs A)

- Same as Run A except:
- `point_order_mode=morton`

### Run D (Ray ablation vs A)

- Same as Run A except:
- `n_ray=0`, `max_len=2500`

## 4. Data Preconditions

### 4.1 `pc_fps_order`

- `shapenet_cache_v0` and `scanobjectnn_main_split_v2` include `pc_fps_order`.

### 4.2 `pt_fps_order` (recommended for A/C/D speed)

For `pt_xyz_key=pt_xyz_pool` + `pt_sample_mode_train=fps`, precompute `pt_fps_order`:

```bash
python -m nepa3d.data.migrate_add_pt_fps_order \
  --cache_root data/shapenet_cache_v0 \
  --splits train,test \
  --pt_key pt_xyz_pool \
  --out_key pt_fps_order \
  --fps_k 2048 \
  --workers 32 \
  --write_mode append

python -m nepa3d.data.migrate_add_pt_fps_order \
  --cache_root data/scanobjectnn_main_split_v2 \
  --splits train,test \
  --pt_key pt_xyz_pool \
  --out_key pt_fps_order \
  --fps_k 2048 \
  --workers 32 \
  --write_mode append
```

If missing, tokenizer falls back to on-the-fly FPS (correct but slower).

## 5. Multi-Node PBS Launcher

New launcher (pbsdsh fan-out):  
`scripts/pretrain/nepa3d_pretrain_multinode_pbsdsh.sh`

Run-time node logs are written under:

- `logs/ddp_pretrain/ddp_pretrain_<PBS_JOBID>_<RUN_TAG>/logs/*.pretrain.log`

Single-run example (Run A):

```bash
qsub -l rt_QF=8 -l walltime=72:00:00 \
  -v RUN_TAG=runA,MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_POINT=1024,N_RAY=1024,QA_TOKENS=1,MAX_LEN=4500,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,PT_SAMPLE_MODE_TRAIN=fps,PT_FPS_KEY=auto,POINT_ORDER_MODE=fps,BATCH=16,NUM_WORKERS=8,SAVE_DIR=runs/pretrain_abcd_1024_runA \
  scripts/pretrain/nepa3d_pretrain_multinode_pbsdsh.sh
```

The launcher computes:

- `NUM_PROCESSES = NNODES * NPROC_PER_NODE`
- `NUM_MACHINES = NNODES`
- `MACHINE_RANK` per host from `PBS_NODEFILE`

then calls `scripts/pretrain/nepa3d_pretrain.sh` on each node with matching rendezvous env.

## 6. Parallel A/B/C/D Submission

Helper script added:

- `scripts/pretrain/submit_pretrain_abcd_qf.sh`

Example:

```bash
NODES_PER_RUN=2 WALLTIME=24:00:00 BATCH=16 EPOCHS=100 \
  bash scripts/pretrain/submit_pretrain_abcd_qf.sh
```

This submits 4 jobs (`runA/runB/runC/runD`) in parallel with fixed settings above.

## 7. Notes

- Mixed corpus with `pointcloud_noray` keeps ray tokens as missing, which is intended.
- `max_len` must satisfy:
  - QA mode: `1 + 2*n_point + 2*n_ray + eos`
  - legacy mode: `1 + n_point + n_ray + eos`
- If changing `n_point/n_ray`, update `max_len` accordingly.

## 8. Post-Pretrain Evaluation Launch

Submitted as A/B/C/D one-job-per-run bundles:

- script: `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
- submit helper: `scripts/eval/submit_abcd_cls_cpac_qf.sh`

Each run executes:

- ScanObjectNN classification fine-tune/eval (`pointcloud_noray`, `n_point=1024`, `n_ray=0`, `pc_xyz`, xyz-only)
  - fine-tune is launched with `accelerate` multi-GPU on one node (`NPROC_PER_NODE`, default `4`)
  - current rerun default is `mean_q` (`--cls_pooling mean_q`); old failed logs used `mean_pts`
- CPAC-UDF eval (`pointcloud_noray -> udf`, non-transductive head train: `train_udf`)
- ModelNet40 classification step is auto-run only when `data/modelnet40_cache_v2` exists; otherwise it is skipped.

### CPAC rerun note (Python 3.9)

- On Python 3.9, `nepa3d/analysis/completion_cpac_udf.py` failed at import time when using `|` union type hints.
- Fixed by replacing `T | None` / `slice | np.ndarray` hints with `typing.Optional` / `typing.Union`.
- CPAC-only rerun scripts:
  - run script: `scripts/eval/nepa3d_cpac_only_qf.sh`
  - submit helper: `scripts/eval/submit_abcd_cpac_only_qf.sh`
- Result JSON path: `results/cpac_abcd_1024_run{A,B,C,D}.json`

### CPAC rerun status

- Initial eval bundle jobs (`93547-93550`) failed at CPAC import with Python 3.9 type-hint incompatibility.
- CPAC-only rerun jobs (`93581-93584`) finished successfully (`Exit_status=0`).
- CPAC summary (`max_shapes=800`, split=`eval`):
  - Run A: `mae=0.1604`, `rmse=0.2056`, `iou@tau=0.2684`
  - Run B: `mae=0.1621`, `rmse=0.2081`, `iou@tau=0.2665`
  - Run C: `mae=0.1897`, `rmse=0.2396`, `iou@tau=0.1721`
  - Run D: `mae=0.2043`, `rmse=0.2513`, `iou@tau=0.1144`

## 9. Classification Result Snapshot (Historical / Superseded by §14)

- Source logs:
  - `logs/eval/abcd_cls_cpac/runA_classification_scan.log`
  - `logs/eval/abcd_cls_cpac/runB_classification_scan.log`
  - `logs/eval/abcd_cls_cpac/runC_classification_scan.log`
  - `logs/eval/abcd_cls_cpac/runD_classification_scan.log`
- Final ScanObjectNN metrics:
  - Run A: `best_val=0.3492`, `test_acc=0.3157`
  - Run B: `best_val=0.2439`, `test_acc=0.2230`
  - Run C: `best_val=0.2560`, `test_acc=0.2383`
  - Run D: `best_val=0.2373`, `test_acc=0.2260`
- Eval script behavior for all A/B/C/D runs (confirmed in line 3 of each log):
  - `train_backend=pointcloud_noray`, `eval_backend=pointcloud_noray`
  - `cls_pooling=mean_pts`
  - `ablate_point_dist=True` (XYZ-only classification setting)
  - Fine-tune uses `n_ray=0` input.

## 10. Verified Mismatch (Important)

- Intended plan: `qa_tokens=1` for A/C/D, `qa_tokens=0` for B.
- Actual saved checkpoints show `qa_tokens=0` for **all** A/B/C/D:
  - `runs/pretrain_abcd_1024_runA/last.pt`
  - `runs/pretrain_abcd_1024_runB/last.pt`
  - `runs/pretrain_abcd_1024_runC/last.pt`
  - `runs/pretrain_abcd_1024_runD/last.pt`
- Root cause:
  - `scripts/pretrain/submit_pretrain_abcd_qf.sh` passed `QA_TOKENS=...`
  - but `scripts/pretrain/nepa3d_pretrain.sh` did not forward `--qa_tokens`/`--qa_layout` to `nepa3d.train.pretrain`.
  - this forced default `--qa_tokens 0` from `pretrain.py`.

### Fix applied

- `scripts/pretrain/nepa3d_pretrain.sh` now forwards:
  - `--qa_tokens "${QA_TOKENS}"`
  - `--qa_layout "${QA_LAYOUT}"`
- This fixes future submissions, but does **not** change already-trained checkpoints.

## 11. Additional Sanity Check (pt_fps_order path)

- In final successful pretrain logs (`93506-93509`), no `pt_fps_order` missing / on-the-fly FPS fallback warnings were found.
- So the latest A/B/C/D training speed issue from missing `pt_fps_order` is resolved in those runs.

## 12. Log Audit: Why "Scaling" likely hurt

From the successful pretrain logs (`93506-93509`):

- DDP scale was `num_processes=8` (`2 nodes x 4 GPU`), `batch=16` per process (`global batch=128`).
- Effective pretrain LR recorded in logs was:
  - `lr=1.875e-05` (`[lr-scale] enabled: base_lr=3.75e-05 total_batch=128 ref_batch=256`)
- This is substantially smaller than the nominal `LR=3e-4` usually assumed in command templates, and can cause under-training while increasing scale.

Classification logs (`run*_classification_scan.log`) also show:

- all runs used `pointcloud_noray`, `n_ray=0`, `ablate_point_dist=True`, `cls_pooling=mean_pts`.
- this means A/C/D were evaluated in XYZ-only mode (their ray/dist pretrain advantages are intentionally removed in this eval setting).

Interpretation:

- There are two independent factors that can make scaling appear worse:
  - pretrain LR scaling behavior yields very small LR at large global batch
  - eval protocol forces xyz-only classification for all runs, which is not a full-spec A/C/D readout

## 13. Patch Applied (LR scaling / mixed precision / cls pooling)

Applied from `nepa_pretrain_lrscale_mixedprecision_patch.zip` (with one safety addition):

- Updated pretrain launcher scripts:
  - `scripts/pretrain/nepa3d_pretrain.sh`
  - `scripts/pretrain/nepa3d_pretrain_pointcloud.sh`
  - `scripts/pretrain/nepa3d_kplane_pretrain.sh`
  - `scripts/pretrain/run_shapenet_mix_pretrains_mainsplit_local.sh`
  - `scripts/pretrain/run_shapenet_m1_pretrains_local.sh`
  - `scripts/pretrain/run_shapenet_simple_local.sh`
- Changes:
  - `LR_SCALE_ENABLE` default set to `0` (off by default).
  - fixed base-LR derivation formula:
    - old: `BASE_LR = LR * LR_BASE_TOTAL_BATCH / 256`
    - new: `BASE_LR = LR * 256 / LR_BASE_TOTAL_BATCH`
  - `--mixed_precision` passed to Python with resolved `LAUNCH_MIXED_PRECISION` (`no/fp16/bf16`).
  - note: in `nepa3d_kplane_pretrain.sh`, added explicit `LAUNCH_MIXED_PRECISION` resolve block to avoid unset variable risk.

Classification robustness patch:

- `nepa3d/train/finetune_cls.py`
  - added `cls_pooling=mean_q` (query-point mean: `TYPE_POINT` + `TYPE_Q_POINT`).
  - safety guard: when `qa_tokens=1` + `--ablate_point_dist` + `cls_pooling=mean_pts`, auto-switch to `mean_q` with warning.
- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
  - `CLS_POOLING` env (default `mean_q`)
  - `ABLATE_POINT_DIST` env (default `1`)
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - forwards `CLS_POOLING` and `ABLATE_POINT_DIST` to submitted jobs.

### Re-run log checklist (must-pass)

- Pretrain log head:
  - expected by default: `[lr-scale] disabled: lr=<configured>`
  - `qa_tokens` in checkpoint args must match run design (A/C/D=`1`, B=`0`).
- Eval log head:
  - `cls_pooling` should resolve to `mean_q` for QA+dist-ablation runs.
  - if `qa_tokens=1` and dist ablation is on, no `mean_pts` should remain without warning/override.

## 14. Final Outcome (2026-02-23)

Final eval jobs (new batch) used:

- A/C: `logs/eval/abcd_cls_cpac_fix20260222_200311/run{A,C}.out`
- B/D rerun: `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/run{B,D}.out`

Final scheduler status:

- pretrain: `94030/94031/94032/94033` all `Exit_status=0`
- eval: `94034=0`, `94036=0`, `94071=0`, `94072=1`

### 14.1 ScanObjectNN classification (final)

From:

- `logs/eval/abcd_cls_cpac_fix20260222_200311/runA_fix20260222_200311_classification_scan.log`
- `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/runB_fix20260222_200311_rerun1_classification_scan.log`
- `logs/eval/abcd_cls_cpac_fix20260222_200311/runC_fix20260222_200311_classification_scan.log`
- `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/runD_fix20260222_200311_rerun1_classification_scan.log`

| Run | best_val | best_ep | test_acc |
|---|---:|---:|---:|
| A | 0.8554 | 96 | 0.6279 |
| B (rerun1) | 0.9065 | 89 | 0.6506 |
| C | 0.8460 | 83 | 0.5906 |
| D (rerun1) | 0.8155 | 82 | 0.6129 |

### 14.2 CPAC-UDF (final)

From JSON:

- `results/cpac_abcd_1024_runA_fix20260222_200311.json`
- `results/cpac_abcd_1024_runB_fix20260222_200311_rerun1.json`
- `results/cpac_abcd_1024_runC_fix20260222_200311.json`

| Run | mae | rmse | iou@tau |
|---|---:|---:|---:|
| A | 0.0639 | 0.0863 | 0.5576 |
| B (rerun1) | 0.1174 | 0.1575 | 0.4193 |
| C | 0.1026 | 0.1329 | 0.3528 |

Run D CPAC in rerun1 failed (no final JSON):

- log: `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/runD.out`
- error: `RuntimeError: The size of tensor a (4098) must match the size of tensor b (2500) at non-singleton dimension 1`
- interpretation: CPAC eval context/query token length exceeded the checkpoint positional length (`max_len=2500`) for D.

### 14.3 ModelNet40 status

All runs skipped ModelNet40 because cache was missing:

- `[skip] ModelNet cache not found: data/modelnet40_cache_v2`

Seen in:

- `logs/eval/abcd_cls_cpac_fix20260222_200311/runA.out`
- `logs/eval/abcd_cls_cpac_fix20260222_200311/runC.out`
- `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/runB.out`
- `logs/eval/abcd_cls_cpac_fix20260222_200311_rerun1/runD.out`

## 15. Log Audit (Problems / Non-problems)

### 15.1 Confirmed OK

- Pretrain A/B/C/D completed successfully (`Exit_status=0`).
- No NaN/Traceback/on-the-fly-FPS fallback warnings found in final pretrain logs (`ddp_pretrain_94030..94033`).
- Classification stage completed for all A/B/C/D (including D inside rerun1 before CPAC step).

### 15.2 Confirmed issue still open

- CPAC for Run D (rerun1) failed due token length mismatch (`4098` vs checkpoint `2500`).
- Therefore, D has valid classification metrics but no valid CPAC metric for this new batch.

### 15.3 Recommended fix for D-CPAC rerun

Either:

- reduce CPAC token load so total length fits `<=2500` (e.g., smaller `n_context` and/or `n_query` for D), or
- re-train D with larger `max_len` to support current CPAC eval tokenization.
