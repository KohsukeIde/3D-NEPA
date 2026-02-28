# 1024-Point Pretrain A/B/C/D (Active Plan)

Last updated: 2026-02-26

> Legacy note (2026-02-26): this file is now treated as a historical execution ledger.
> Active protocol-correct re-evaluation planning is tracked in:
> `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
> Canonical benchmark/read path:
> `nepa3d/docs/README.md`
> Track classification:
> this file belongs to the **Query-NEPA (token-level)** line.
> It is not the active source for the new Patch-NEPA Stage-2 line.
> Patch-NEPA active source:
> `nepa3d/docs/patch_nepa/patch_nepa_stage2_active.md`
> Interpretation guard:
> this ledger intentionally contains failed/superseded/misaligned historical runs.
> Do not use this file alone for benchmark claims; use the canonical validity boundary in
> `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`.

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

### 15.4 Next retrain policy (2026-02-23)

- To avoid this class of failure in the next batch, unify A/B/C/D pretrain to `max_len=4500`.
- `scripts/pretrain/submit_pretrain_abcd_qf.sh` now exposes `MAX_LEN_ABCD` (default `4500`) and applies it to all runs.
- CPAC wrappers perform a checkpoint-length precheck before evaluation and fail fast when `n_context/n_query` exceed the effective max length.

## 16. Feedback Patch Intake (2026-02-24)

### 16.1 `point_order_mode` patch status

Applied from `nepa_point_ordermode_patch_20260223.zip`:

- `nepa3d/train/finetune_cls.py`
  - added CLI: `--point_order_mode {morton,fps,random}`
  - now forwarded to train/val/test `ModelNet40QueryDataset`
- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
  - added `POINT_ORDER_MODE` env (default `morton`)
  - forwarded to both ScanObjectNN and ModelNet40 classification commands
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - added `POINT_ORDER_MODE` qsub propagation

### 16.2 Fine-tune optimization recipe patch

`nepa3d/train/finetune_cls.py` now includes:

- LR scheduling:
  - `--lr_scheduler {none,cosine}` (default `cosine`)
  - `--warmup_epochs` (default `10`)
  - `--warmup_start_factor` (default `0.1`)
  - `--min_lr` (default `1e-6`)
- gradient accumulation:
  - `--grad_accum_steps` (default `1`)
  - wired through `Accelerator(gradient_accumulation_steps=...)`
- gradient clipping:
  - `--max_grad_norm` (default `0.0`, disabled when `<=0`)
- per-epoch logs now report:
  - `lr`, `updates_ep`, `updates_total`
  - scheduler and accumulation settings in startup config line

Related launchers now expose these knobs:

- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
- `scripts/finetune/nepa3d_finetune_scanobjectnn.sh`

### 16.3 ModelNet40 (PLY tree) preprocessing support

Current dataset at `data/ModelNet40` is `ply_format/.../*.ply` (not `*.off`), so preprocessing was required.

Implemented in `nepa3d/data/preprocess_modelnet40.py`:

- mesh auto-discovery now supports:
  - `<root>/*/<split>/*.off` (official)
  - `<root>/<split>/*/*.off`
  - `<root>/ply_format/<split>/*/*.ply`
  - `<root>/<split>/*/*.ply`
- added `--mesh_glob` override
- robust class-name inference across supported layouts
- startup prints discovered source pattern and mesh count
- pointcloud-only `*.ply` fallback:
  - handles `trimesh.PointCloud` / `Scene` inputs
  - computes `pt_dist_pool` from nearest-neighbor distances (no mesh closest-point dependency)
  - builds ray pools from occupancy-based DDA fallback when mesh rays are unavailable

Also updated `scripts/preprocess/preprocess_modelnet40.sh`:

- added optional `MESH_GLOB` env pass-through to `--mesh_glob`

Quick local verification (venv):

- train discovery: `data/ModelNet40/ply_format/train/*/*.ply` -> `9843` meshes
- test discovery: `data/ModelNet40/ply_format/test/*/*.ply` -> `601` meshes
- one-file smoke preprocess on pointcloud-only PLY completed successfully (temporary cache cleaned after verification).

### 16.4 Next execution order

1. Build `data/modelnet40_cache_v2` from `data/ModelNet40` (now supported directly).
2. Re-run A/B/C/D eval bundle with updated fine-tune recipe and explicit `POINT_ORDER_MODE`.
3. Keep SOTA-fair table (xyz-only) and NEPA-full table (dist/ray enabled) separated in reporting.

## 17. Execution Log (2026-02-24)

### 17.1 ModelNet40 cache v2 build

Submitted via:

- `scripts/preprocess/submit_modelnet40_cache_v2_qf.sh`

Initial job:

- Job: `94919.qjcm`
- Result: failed (`Exit_status=1`)
- Cause: one invalid file in source tree raised `ValueError: Not a ply file!`
  - path: `data/ModelNet40/ply_format/test/chair/chair_0900_[8].ply` (size `0`)
  - log: `logs/preprocess/modelnet40_cache_v2/preprocess_modelnet40_cache_v2_20260224_010540.err`

Fix applied:

- `nepa3d/data/preprocess_modelnet40.py`
  - `trimesh.load(...)` failure now returns `False` (skip sample)
  - worker wrapper catches unexpected exceptions and marks sample as failed instead of aborting the full run

Retry job:

- Job: `94925.qjcm`
- Result: success (`Exit_status=0`)
- Logs:
  - `logs/preprocess/modelnet40_cache_v2/preprocess_modelnet40_cache_v2_retry1_20260224_010816.out`
  - `logs/preprocess/modelnet40_cache_v2/preprocess_modelnet40_cache_v2_retry1_20260224_010816.err`
- Final cache counts:
  - `data/modelnet40_cache_v2/train`: `9843` files
  - `data/modelnet40_cache_v2/test`: `600` files
  - note: source had `601` test files; one zero-byte invalid file was skipped.

### 17.2 A/B/C/D eval re-launch (dependency after cache build)

Submitted via:

- `scripts/eval/submit_abcd_cls_cpac_qf.sh`

Jobs:

- `94926.qjcm` (runA)
- `94927.qjcm` (runB)
- `94928.qjcm` (runC)
- `94929.qjcm` (runD)

Submission profile:

- dependency: `afterok:94925.qjcm`
- `EPOCHS_CLS=300`, `LR_CLS=3e-4`
- `LR_SCHEDULER=cosine`, `WARMUP_EPOCHS=10`, `MIN_LR=1e-6`
- `GRAD_ACCUM_STEPS=1`, `MAX_GRAD_NORM=1.0`
- `POINT_ORDER_MODE=morton`
- run set tag: `fix20260224_sched300_retry1`

Current log roots:

- launcher logs: `logs/eval/abcd_cls_cpac_fix20260224_sched300_retry1/`
- per-stage logs (inside each job):
  - `run*_classification_scan.log`
  - `run*_classification_modelnet.log`
  - `run*_cpac.log`
- outputs:
  - `runs/eval_abcd_1024_fix20260224_sched300_retry1/`
  - `results/abcd_1024_fix20260224_sched300_retry1/`

Initial runtime checks (all runs):

- CPAC max-length precheck passed.
- Classification started with intended settings (`cls_pooling=mean_q`, `ablate_point_dist=1`, `point_order_mode=morton`, cosine scheduler settings shown in logs).
- Early training heartbeat confirmed (`ep=0` logged for all A/B/C/D in `run*_classification_scan.log`).

### 17.3 Retry1 failure diagnosis and Retry2 relaunch

Retry1 jobs (`94926`..`94929`) all ended with `Exit_status=1`.

#### 17.3.1 Retry1 partial metrics (recorded before crash)

Although retry1 failed at final ModelNet test evaluation, ScanObjectNN classification had already completed and wrote final test metrics.

ScanObjectNN (from `run*_classification_scan.log`):

| Run | best_val | test_acc |
|---|---:|---:|
| A | 0.4517 | 0.3946 |
| B | 0.3163 | 0.2889 |
| C | 0.3874 | 0.3641 |
| D | 0.3189 | 0.2813 |

ModelNet40 status in retry1:

- training/validation reached `ep=299` and best validation was recorded,
- but final test step crashed due corrupted NPZ, so `test_acc` was not produced.

ModelNet40 (validation-only, from `run*_classification_modelnet.log`):

| Run | best_val | best_ep | test_acc |
|---|---:|---:|---|
| A | 0.8760 | 255 | not available (crashed) |
| B | 0.8418 | 269 | not available (crashed) |
| C | 0.8799 | 258 | not available (crashed) |
| D | 0.8643 | 245 | not available (crashed) |

CPAC status in retry1:

- not executed for A/B/C/D (jobs terminated at ModelNet test crash before CPAC stage).

Root cause:

- all runs crashed during ModelNet40 test evaluation with:
  - `zipfile.BadZipFile: File is not a zip file`
- source was corrupted cache entries in `data/modelnet40_cache_v2`:
  - `data/modelnet40_cache_v2/test/car/car_0260_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0263_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0265_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0266_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0267_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0268_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0269_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0270_[7].npz`
  - `data/modelnet40_cache_v2/test/car/car_0271_[7].npz`

Fixes:

- `nepa3d/data/preprocess_modelnet40.py`
  - added `is_valid_npz()` and changed skip logic to only skip valid NPZ.
  - invalid existing NPZ are now regenerated automatically.
- regenerated affected cache entries from source PLY files.
- post-fix integrity check: `total=10443`, `bad=0`.

Relaunch:

- New eval batch submitted:
  - `94968.qjcm` (runA)
  - `94969.qjcm` (runB)
  - `94970.qjcm` (runC)
  - `94971.qjcm` (runD)
- run set: `fix20260224_sched300_retry2`
- log root: `logs/eval/abcd_cls_cpac_fix20260224_sched300_retry2/`

#### 17.3.2 Retry2 final metrics (A/B/C/D all completed)

> Note (2026-02-24): this Retry2 block used deprecated pretrain checkpoints (`runs/pretrain_abcd_1024_run*`, `qa_tokens=0`, low-LR) and is kept only as historical trace, not as the current primary result.

Retry2 jobs all finished successfully (`Exit_status=0`):

- `94968.qjcm` runA
- `94969.qjcm` runB
- `94970.qjcm` runC
- `94971.qjcm` runD

Final metrics:

| Run | ScanObjectNN `test_acc` | ModelNet40 `test_acc` | CPAC `iou@tau` | CPAC `mae` | CPAC `rmse` |
|---|---:|---:|---:|---:|---:|
| A | 0.4150 | 0.9111 | 0.2684 | 0.1604 | 0.2056 |
| B | 0.2913 | 0.9346 | 0.2665 | 0.1621 | 0.2081 |
| C | 0.3831 | 0.9160 | 0.1721 | 0.1897 | 0.2396 |
| D | 0.3031 | 0.9199 | 0.1144 | 0.2043 | 0.2513 |

Source logs/results:

- ScanObjectNN: `logs/eval/abcd_cls_cpac_fix20260224_sched300_retry2/run*_classification_scan.log`
- ModelNet40: `logs/eval/abcd_cls_cpac_fix20260224_sched300_retry2/run*_classification_modelnet.log`
- CPAC JSON: `results/abcd_1024_fix20260224_sched300_retry2/cpac_abcd_1024_run*.json`

## 18. Run A Scan Ablation (16GPU, fps/morton x aug none/scan)

Purpose:

- Isolate ScanObjectNN classification sensitivity to:
  - token order at fine-tune (`POINT_ORDER_MODE=fps` vs `morton`)
  - augmentation preset (`SCAN_AUG_PRESET=none` vs `scanobjectnn`)
- Keep pretrain checkpoint fixed to Run A:
  - `runs/pretrain_abcd_1024_fix20260222_200311_runA/last.pt`

Launch/runtime:

- submit helper: `scripts/eval/submit_scan_ablation_aug_order_qf.sh`
- launcher: `scripts/eval/nepa3d_eval_cls_cpac_multinode_pbsdsh.sh`
- cluster shape: `4 nodes x 4 GPU/node = 16 GPU` per run (`NUM_PROCESSES=16`)
- jobs:
  - `95239.qjcm` runA_fps_augnone (`Exit_status=0`, walltime `03:00:09`)
  - `95240.qjcm` runA_fps_augscan (`Exit_status=0`, walltime `02:37:33`)
  - `95241.qjcm` runA_morton_augnone (`Exit_status=0`, walltime `02:37:05`)
  - `95242.qjcm` runA_morton_augscan (`Exit_status=0`, walltime `02:37:14`)

Final ScanObjectNN metrics:

| Run | point_order_mode | aug_preset | best_val | best_ep | test_acc |
|---|---|---|---:|---:|---:|
| runA_fps_augnone | fps | none | 0.9152 | 191 | 0.6969 |
| runA_morton_augnone | morton | none | 0.9165 | 170 | 0.6976 |
| runA_fps_augscan | fps | scanobjectnn | 0.8631 | 180 | 0.6459 |
| runA_morton_augscan | morton | scanobjectnn | 0.8882 | 237 | 0.6519 |

Source logs:

- `logs/eval/scan_ablation_aug_order_fix20260224_scanablation_2x2_16gpu/runA_fps_augnone_classification_scan.log`
- `logs/eval/scan_ablation_aug_order_fix20260224_scanablation_2x2_16gpu/runA_fps_augscan_classification_scan.log`
- `logs/eval/scan_ablation_aug_order_fix20260224_scanablation_2x2_16gpu/runA_morton_augnone_classification_scan.log`
- `logs/eval/scan_ablation_aug_order_fix20260224_scanablation_2x2_16gpu/runA_morton_augscan_classification_scan.log`

Quick read:

- `aug_preset=none` outperformed `aug_preset=scanobjectnn` by about `+0.045` to `+0.051` in `test_acc`.
- `fps` vs `morton` difference was small in this setup; `morton` was slightly higher (`+0.0007` to `+0.0060`).
- All runs show large `best_val` vs `test_acc` gap and reached near-perfect train accuracy in late epochs, indicating persistent overfitting risk.

## 19. SOTA-fair vs NEPA-full (Pool/LS + Regularization Ablations, 2026-02-24)

Purpose:

- Verify fine-tune-side improvements with short schedule (`EPOCHS_CLS=120`) under two protocols:
  - SOTA-fair: `pt_xyz_key=pc_xyz`, `ablate_point_dist=1`, `point_order_mode=morton`
  - NEPA-full: `pt_xyz_key=pt_xyz_pool`, `ablate_point_dist=0`, `point_order_mode=fps`
- Evaluate:
  - pooling + label smoothing combinations
  - regularization components (`fc_norm`, no-decay split, label smoothing)

Launch/runtime:

- jobs: `95628.qjcm` .. `95643.qjcm` (16 jobs total)
- each job: `rt_QF=1`, `ngpus=4` (4 GPU/job), all finished `Exit_status=0`
- walltime per job: roughly `03:55` to `04:02`

### 19.1 SOTA-fair: Pooling + Label Smoothing

Source: `logs/eval/scan_pool_ls_scan_sotafair_poolls_20260224_175813/*_classification_scan.log`

| Run | cls_pooling | label_smoothing | best_val | best_ep | test_acc |
|---|---|---:|---:|---:|---:|
| run_q_ls00 | mean_q | 0.0 | 0.9559 | 89 | 0.7510 |
| run_q_ls01 | mean_q | 0.1 | 0.9541 | 73 | 0.7492 |
| run_a_ls00 | mean_a | 0.0 | 0.9491 | 113 | 0.7350 |
| run_a_ls01 | mean_a | 0.1 | 0.9453 | 59 | 0.7274 |

### 19.2 SOTA-fair: Regularization Ablation

Source: `logs/eval/scan_regab_scan_sotafair_regab_20260224_175813/*_classification_scan.log`

| Run | use_fc_norm | label_smoothing | weight_decay_norm | best_val | best_ep | test_acc |
|---|---:|---:|---:|---:|---:|---:|
| s0_base | 0 | 0.0 | 0.05 | 0.9417 | 73 | 0.7389 |
| s1_wdsplit | 0 | 0.0 | 0.00 | 0.9435 | 95 | 0.7326 |
| s2_fc_norm | 1 | 0.0 | 0.00 | 0.9545 | 65 | 0.7526 |
| s3_fc_norm_ls | 1 | 0.1 | 0.00 | 0.9535 | 89 | 0.7447 |

### 19.3 NEPA-full: Pooling + Label Smoothing

Source: `logs/eval/scan_pool_ls_scan_nepafull_poolls_20260224_175813/*_classification_scan.log`

| Run | cls_pooling | label_smoothing | best_val | best_ep | test_acc |
|---|---|---:|---:|---:|---:|
| run_q_ls01 | mean_q | 0.1 | 0.5345 | 27 | 0.4928 |
| run_q_ls00 | mean_q | 0.0 | 0.5423 | 21 | 0.4911 |
| run_a_ls01 | mean_a | 0.1 | 0.5278 | 22 | 0.4809 |
| run_a_ls00 | mean_a | 0.0 | 0.5140 | 96 | 0.4526 |

### 19.4 NEPA-full: Regularization Ablation

Source: `logs/eval/scan_regab_scan_nepafull_regab_20260224_175813/*_classification_scan.log`

| Run | use_fc_norm | label_smoothing | weight_decay_norm | best_val | best_ep | test_acc |
|---|---:|---:|---:|---:|---:|---:|
| s0_base | 0 | 0.0 | 0.05 | 0.5357 | 19 | 0.4842 |
| s1_wdsplit | 0 | 0.0 | 0.00 | 0.5272 | 20 | 0.4696 |
| s2_fc_norm | 1 | 0.0 | 0.00 | 0.5128 | 100 | 0.4498 |
| s3_fc_norm_ls | 1 | 0.1 | 0.00 | 0.5210 | 21 | 0.4734 |

Quick read:

- SOTA-fair:
  - `mean_q` clearly outperformed `mean_a` (about `+0.014` to `+0.022` in `test_acc`).
  - best was `s2_fc_norm` (`test_acc=0.7526`), indicating `fc_norm` helps in this protocol.
  - adding label smoothing on top (`s3_fc_norm_ls`) reduced `test_acc` vs `s2_fc_norm`.
- NEPA-full:
  - all settings were significantly lower (`test_acc ~0.45-0.49`) and did not show the same gains.
  - this indicates a protocol mismatch or data/feature usage issue that needs separate debugging before merging with SOTA-fair conclusions.
- No runtime exceptions were observed in these 16 runs (only terminal NCCL process-group shutdown warnings).

## 20. ScanObjectNN val-test gap audit (2026-02-24)

Question:

- Why is `best_val` very high (for example `~0.95`) while `test_acc` stays much lower (`~0.75`)?

### 20.1 Confirmed current split behavior

- `finetune_cls.py` builds `val` from `train` via random stratified file-level split:
  - `train_paths_full = list_npz(args.cache_root, "train")`
  - `train_paths, val_paths = stratified_train_val_split(...)`
  - refs: `nepa3d/train/finetune_cls.py:418`, `nepa3d/train/finetune_cls.py:439`
- split function is class-stratified but not group-aware:
  - groups by class only, shuffles file paths, takes first `n_val`.
  - ref: `nepa3d/data/modelnet40_index.py:22`

### 20.2 Cache composition confirms multi-variant setup

- `scanobjectnn_main_split_v2` train metadata includes 5 H5 sources:
  - `training_objectdataset.h5`
  - `training_objectdataset_augmented25_norot.h5`
  - `training_objectdataset_augmented25rot.h5`
  - `training_objectdataset_augmentedrot.h5`
  - `training_objectdataset_augmentedrot_scale75.h5`
  - ref: `data/scanobjectnn_main_split_v2/_meta/scanobjectnn_train_source.txt:8`
- log-confirmed counts in current runs:
  - `num_train=43298 num_val=4805 num_test=12137`
  - ref: `logs/eval/scan_pool_ls_scan_sotafair_poolls_20260224_175813/run_a_ls00_classification_scan.log:23`

### 20.3 Leakage check result (train/val)

Leak proxy:

- canonical group id = `class_id + base_object_id` (basename numeric suffix after removing augmentation tag).
- this approximates "same original object, different variant".

Measured on `data/scanobjectnn_main_split_v2/train`:

- total files: `48103`
- unique canonical groups: `14337`
- groups with multiple variants: `11909` (`83.06%`)
- group size histogram: `{4:10749, 1:2428, 2:921, 3:179, 5:60}`

Reproducing current split (`val_ratio=0.1`, `val_seed=0`):

- file split sizes: `train=43298`, `val=4805`
- val groups: `4188`
- overlapping groups between split-train and val: `3949` (`94.29%` of val groups)
- leaking val files (same canonical group exists in split-train): `4553/4805` (`94.76%`)

Interpretation:

- The current `val` is heavily contaminated by same-object variant overlap with train.
- This can inflate `val_acc` and create large `best_val` vs `test_acc` gaps.

### 20.4 Mitigation (recommended)

1. Replace file-level val split with group-aware stratified split for ScanObjectNN.
2. Group by canonical object id (`class + base id`), then assign full groups to train or val.
3. Keep test as official `test` split and report it separately as the benchmark metric.

Quick feasibility check:

- group-aware split with the same ratio gives roughly similar sizes (`train=43354`, `val=4749`) but zero train/val group overlap.

## 21. New knobs (LLRD / drop_path / val split) and pipeline (2026-02-24)

### 21.1 Fine-tune code updates

- `nepa3d/train/finetune_cls.py` now supports:
  - `--val_split_mode {file,group_auto,group_scanobjectnn}`
    - `group_auto` uses ScanObjectNN group-aware split (no same-group train/val leakage).
  - `--llrd` (layer-wise LR decay; `1.0` disables)
  - `--drop_path` (backbone stochastic depth rate)
- Group-aware split utility added in:
  - `nepa3d/data/modelnet40_index.py`
  - `scanobjectnn_group_key(...)` + grouped stratified split path

### 21.2 Backbone drop_path support

- Added block-level stochastic depth path to:
  - `nepa3d/models/causal_transformer.py`
  - `nepa3d/models/encdec_transformer.py`
- `QueryNepa` now accepts `drop_path` and forwards it:
  - `nepa3d/models/query_nepa.py`
- `pretrain.py` now has `--drop_path`; launcher scripts pass `DROP_PATH`.

### 21.3 Eval/submit script wiring

- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh` and `scripts/eval/submit_abcd_cls_cpac_qf.sh` now forward:
  - `LLRD`, `DROP_PATH`, `VAL_SPLIT_MODE`
  - `PT_SAMPLE_MODE_TRAIN_CLS`, `PT_SAMPLE_MODE_EVAL_CLS`, `PT_RFPS_M_CLS`
  - `AUG_EVAL` (vote-style TTA switch)
  - current default: `AUG_EVAL=1` (TTA on; set `AUG_EVAL=0` to disable)
- New ablation submit helper:
  - `scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh`
  - default ablation set: `base,llrd,dp,llrd_dp`
  - default `AUG_EVAL=1` (TTA on)

### 21.4 End-to-end pipeline submit helper

- New chained submit script:
  - `scripts/pipeline/submit_pretrain_then_sotafair_eval_qf.sh`
- Behavior:
  1. submit pretrain A/B/C/D
  2. wait via PBS dependency (`afterok:<all_pretrain_job_ids>`)
  3. auto-submit SOTA-fair eval ablation matrix with val split fix (`group_auto`)
  4. default `AUG_EVAL=1` (TTA on) for this eval flow

Default job count:

- pretrain: `4` jobs
- eval: `4 runs x 4 ablations = 16` jobs
- total: `20` jobs

## 22. TTA + Pipeline hotfixes (2026-02-25)

Purpose:

- Ensure the current evaluation flow actually runs `fps + TTA` and consumes the newly trained pretrain checkpoints (not legacy fixed checkpoints).

### 22.1 Script changes applied

- `scripts/pretrain/submit_pretrain_abcd_qf.sh`
  - added `DEFAULT_WORKDIR` / `WORKDIR` handling.
  - added `GROUP_LIST` and `qsub -W group_list=...`.
  - now passes `WORKDIR` via `-v WORKDIR=...` to each submitted pretrain job.
- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
  - default changed to `AUG_EVAL=1` (TTA on by default).
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - default changed to `AUG_EVAL=1`.
  - checkpoint defaults updated:
    - `runs/pretrain_abcd_1024_runA/last.pt`
    - `runs/pretrain_abcd_1024_runB/last.pt`
    - `runs/pretrain_abcd_1024_runC/last.pt`
    - `runs/pretrain_abcd_1024_runD/last.pt`
  - when `QSUB_DEPEND` is set and checkpoint file is not created yet, script now warns and continues submission instead of hard-failing.
- `scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh`
  - default changed to `AUG_EVAL=1`.
  - explicitly forwards `AUG_EVAL` to `submit_abcd_cls_cpac_qf.sh`.
- `scripts/pipeline/submit_pretrain_then_sotafair_eval_qf.sh`
  - now passes `CKPT_RUNA/B/C/D` explicitly to stage2 as:
    - `${WORKDIR}/runs/pretrain_abcd_1024_run{A,B,C,D}/last.pt`
  - forwards `AUG_EVAL=1` by default to stage2.

### 22.2 Behavior after fixes

- Classification eval defaults in this flow are now:
  - `PT_SAMPLE_MODE_EVAL_CLS=fps`
  - `AUG_EVAL=1`
  - `MC_EVAL_K_TEST=10`
- Note:
  - `AUG_EVAL=1` only becomes effective TTA when augmentation is non-trivial.
  - current run uses `SCAN_AUG_PRESET=scanobjectnn`, so test-time augmentation is active.

### 22.3 Submission record for this fix

- First attempt run set:
  - `fps_tta_20260225_012814`
- Issue observed:
  - pretrain jobs launched with incorrect working directory (`/var/spool/pbs`) and failed quickly.
  - stage2 was still targeting legacy fixed checkpoints.
- Second attempt run set (after hotfix):
  - `fps_tta_retry_20260225_013044`
- Submitted jobs:
  - pretrain: `95920` `95921` `95922` `95923`
  - eval (dependency on pretrain afterok): `95924` ... `95939`

### 22.4 Logs for this cycle

- Pipeline meta:
  - `logs/pipeline/pretrain_then_eval_fps_tta_retry_20260225_013044/pretrain_job_ids.txt`
- Pretrain logs:
  - `logs/ddp_pretrain/ddp_pretrain_95920.qjcm_runA/logs/*.pretrain.log`
  - `logs/ddp_pretrain/ddp_pretrain_95921.qjcm_runB/logs/*.pretrain.log`
  - `logs/ddp_pretrain/ddp_pretrain_95922.qjcm_runC/logs/*.pretrain.log`
  - `logs/ddp_pretrain/ddp_pretrain_95923.qjcm_runD/logs/*.pretrain.log`
- Eval logs (once dependency is released):
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_*/run*.out`
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_*/run*.err`

## 23. ScanObjectNN protocol-variant split hotfix (2026-02-25)

Purpose:

- Enforce ScanObjectNN reporting by protocol variant (`obj_bg`, `obj_only`, `pb_t50_rs`) instead of mixed `main_split` cache.

### 23.1 Issue found

- `scripts/eval/submit_abcd_cls_cpac_qf.sh` did not propagate several eval-control env vars to PBS jobs.
- As a result, caller-side overrides for:
  - `RUN_SCAN`, `RUN_MODELNET`, `RUN_CPAC`
  - `SCAN_CACHE_ROOT`, `MODELNET_CACHE_ROOT`, `UNPAIRED_CACHE_ROOT`
  - `SCAN_AUG_PRESET`, `MODELNET_AUG_PRESET`
  - `MC_EVAL_K_VAL`, `MC_EVAL_K_TEST`
  were ignored in actual launched jobs.

### 23.2 Fix applied

- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - now defines and forwards the env vars above in `qsub -v`.
- `scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh`
  - now explicitly passes through:
    - `SCAN_CACHE_ROOT`, `MODELNET_CACHE_ROOT`, `UNPAIRED_CACHE_ROOT`
    - existing run toggles (`RUN_SCAN`, `RUN_MODELNET`, `RUN_CPAC`) and `AUG_EVAL`.
- new helper:
  - `scripts/eval/submit_sotafair_variants_llrd_droppath_ablation_qf.sh`
  - loops over `VARIANTS` (default `obj_bg,obj_only,pb_t50_rs`) and maps each to:
    - `data/scanobjectnn_obj_bg_v2`
    - `data/scanobjectnn_obj_only_v2`
    - `data/scanobjectnn_pb_t50_rs_v2`
  - per variant, calls existing SOTA-fair LLRD/drop_path ablation submitter.
- `scripts/pipeline/submit_pretrain_then_sotafair_eval_qf.sh`
  - supports optional `SCAN_VARIANTS=...`.
  - when set, stage2 uses the new variant submit helper.
  - variant mode defaults to `RUN_MODELNET=0`, `RUN_CPAC=0` (scan classification only).

### 23.3 Usage

Variant-split eval only (no pretrain submit):

```bash
RUN_SET_BASE_PREFIX=fix20260225_scan3 \
VARIANTS=obj_bg,obj_only,pb_t50_rs \
RUN_MODELNET=0 RUN_CPAC=0 \
bash scripts/eval/submit_sotafair_variants_llrd_droppath_ablation_qf.sh
```

End-to-end pretrain -> variant-split eval:

```bash
RUN_TAG_BASE=fix20260225_scan3 \
SCAN_VARIANTS=obj_bg,obj_only,pb_t50_rs \
RUN_MODELNET=0 RUN_CPAC=0 \
bash scripts/pipeline/submit_pretrain_then_sotafair_eval_qf.sh
```

### 23.4 Job count (default ablations=`base,llrd,dp,llrd_dp`)

- variant-split eval only:
  - `3 variants x 4 runs x 4 ablations = 48 jobs`
- pretrain + variant-split eval chain:
  - `4 pretrain + 48 eval = 52 jobs`
- if adding one extra non-variant ModelNet/CPAC matrix (`4x4=16`) separately:
  - total becomes `68 jobs`.

## 24. Result snapshot: `fps_tta_retry_20260225_013044` (2026-02-25)

Purpose:

- Record finalized metrics from the completed SOTA-fair ablation batch with:
  - `PT_SAMPLE_MODE_EVAL_CLS=fps`
  - `AUG_EVAL=1` (TTA on)
  - `VAL_SPLIT_MODE=group_auto` (resolved to `group_scanobjectnn(auto)` in Scan logs)

Run-set roots:

- logs:
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_base`
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_llrd`
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_dp`
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_llrd_dp`
- results:
  - `results/abcd_1024_fps_tta_retry_20260225_013044_sotafair_base`
  - `results/abcd_1024_fps_tta_retry_20260225_013044_sotafair_llrd`
  - `results/abcd_1024_fps_tta_retry_20260225_013044_sotafair_dp`
  - `results/abcd_1024_fps_tta_retry_20260225_013044_sotafair_llrd_dp`

### 24.1 Completion / sanity

- All `4 ablations x 4 runs = 16` jobs finished without runtime errors.
- `run*.err` were empty across all four ablation roots.
- CPAC JSON save markers were found for all runs in all ablations (`16/16`).

### 24.2 ScanObjectNN classification (`test_acc`)

| Ablation | Run A | Run B | Run C | Run D | Avg(ABC) |
|---|---:|---:|---:|---:|---:|
| `base` | 0.6726 | 0.6654 | 0.6204 | 0.2686 | 0.6528 |
| `dp` | 0.6803 | 0.6576 | 0.6222 | 0.2158 | 0.6534 |
| `llrd` | 0.5155 | 0.4862 | 0.4633 | 0.1786 | 0.4883 |
| `llrd_dp` | 0.4508 | 0.4658 | 0.4287 | 0.0840 | 0.4484 |

Quick read:

- `base` and `dp` are nearly tied on `Avg(ABC)` (`dp` +0.0006).
- `llrd` and `llrd_dp` are clearly lower in this run set.
- `Run D` remains substantially weaker than A/B/C under all ablations.

### 24.3 ModelNet40 classification (`test_acc`)

| Ablation | Run A | Run B | Run C | Run D |
|---|---:|---:|---:|---:|
| `base` | 0.9453 | 0.9453 | 0.9395 | 0.5752 |
| `dp` | 0.9414 | 0.9482 | 0.9482 | 0.1260 |
| `llrd` | 0.9229 | 0.9180 | 0.9209 | 0.2246 |
| `llrd_dp` | 0.9121 | 0.9072 | 0.9092 | 0.1953 |

### 24.4 CPAC-UDF (from result JSON)

Note:

- CPAC metrics were identical across the four ablation roots for the same run id (A/B/C/D), because CPAC stage in this flow did not use the fine-tune ablation knobs.

| Run | mae | rmse | iou@tau |
|---|---:|---:|---:|
| A | 0.05808 | 0.08149 | 0.63259 |
| B | 0.09551 | 0.13030 | 0.49248 |
| C | 0.08713 | 0.11350 | 0.36150 |
| D | 0.12917 | 0.16613 | 0.30167 |

### 24.5 Caveat (protocol split)

- This run set used `SCAN_CACHE_ROOT=data/scanobjectnn_main_split_v2` (mixed main_split-derived cache), not protocol-separated caches (`obj_bg` / `obj_only` / `pb_t50_rs`).
- Therefore, these numbers are useful for internal ablation comparison but are not final protocol-split benchmark tables.

## 25. Sampling-mode audit and policy update (2026-02-25)

Purpose:

- Explicitly record pretrain sampling mode provenance for the current result sets.
- Prevent recurrence of ambiguous `fps` vs `rfps` provenance in reports.

### 25.1 Audit result: current reported metrics are from `fps` pretrain

Confirmed from A/B/C/D pretrain logs used by the current eval chains:

- Run A:
  - `logs/ddp_pretrain/ddp_pretrain_95920.qjcm_runA/logs/qh459.pretrain.log:20`
  - `pt_sample_mode_train=fps`
- Run B:
  - `logs/ddp_pretrain/ddp_pretrain_95921.qjcm_runB/logs/qh134.pretrain.log:20`
  - `pt_sample_mode_train=fps`
- Run C:
  - `logs/ddp_pretrain/ddp_pretrain_95922.qjcm_runC/logs/qh139.pretrain.log:20`
  - `pt_sample_mode_train=fps`
- Run D:
  - `logs/ddp_pretrain/ddp_pretrain_95923.qjcm_runD/logs/qh033.pretrain.log:20`
  - `pt_sample_mode_train=fps`

Eval logs are consistent with this (`pt_sample_mode_train=fps`, `pt_sample_mode_eval=fps`), e.g.:

- `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_base/runA.out:15`

Conclusion:

- Section 24 (`fps_tta_retry_20260225_013044`) metrics are all based on `fps`-trained pretrain checkpoints.
- Newly submitted `nepafull_tta_20260225_141831_*` jobs also reference the same `runs/pretrain_abcd_1024_run{A,B,C,D}/last.pt` and therefore are `fps`-pretrain based as well.

### 25.2 Policy update for future reports (mandatory metadata)

For every pretrain/eval result block, record all of:

1. pretrain checkpoint path and job IDs
2. `pt_sample_mode_train` and `pt_rfps_m`
3. `pt_fps_key` / `pt_xyz_key` / `pt_dist_key`
4. eval-side `pt_sample_mode_train` and `pt_sample_mode_eval`
5. protocol split (`main_split` mixed vs `obj_bg` / `obj_only` / `pb_t50_rs`)

### 25.3 `rfps` scope guideline (A-only or not)

Guideline:

- `rfps` should **not** be A-only when comparing A/B/C/D in one table.
- To avoid sampling-mode confound, use the same pretrain sampling mode across all compared runs (`A/B/C/D` all `rfps`, or all `fps`).

If maintaining a separate SOTA-fair protocol that must stay `fps`, report it as a separate table and do not mix with `rfps` A/C/D numbers in one ablation conclusion.

## 26. `rfps` pretrain relaunch and fine-tune mode clarification (2026-02-25)

Purpose:

- Start a new A/B/C/D pretrain batch with `pt_sample_mode_train=rfps`.
- Explicitly record that currently running fine-tune/eval batches are still `fps` on train/eval sampling.

### 26.1 `rfps` pretrain submission (new batch)

Submitted jobs:

- `96381.qjcm` (`runA_rfps_20260225_142727`)
- `96382.qjcm` (`runB_rfps_20260225_142727`)
- `96383.qjcm` (`runC_rfps_20260225_142727`)
- `96384.qjcm` (`runD_rfps_20260225_142727`)

Submission summary file:

- `logs/pipeline/rfps_pretrain_20260225_142727_jobs.txt`

Checkpoint output roots:

- `runs/pretrain_abcd_1024_rfps_20260225_142727_runA`
- `runs/pretrain_abcd_1024_rfps_20260225_142727_runB`
- `runs/pretrain_abcd_1024_rfps_20260225_142727_runC`
- `runs/pretrain_abcd_1024_rfps_20260225_142727_runD`

Key launch settings:

- `PT_SAMPLE_MODE_TRAIN=rfps`
- `PT_RFPS_M=4096`
- `MAX_LEN=4500`
- run matrix otherwise follows A/B/C/D design (B uses `pc_xyz`, others `pt_xyz_pool`; A/C/D keep dist on, B dist ablation).

### 26.2 Clarification: currently running fine-tune/eval mode

- Existing eval batches launched before `rfps` pretrain completion (including `fps_tta_retry_20260225_013044_*` and `nepafull_tta_20260225_141831_*`) are based on `fps` pretrain checkpoints and run fine-tune with:
  - `pt_sample_mode_train=fps`
  - `pt_sample_mode_eval=fps`
- Example evidence:
  - `logs/eval/abcd_cls_cpac_fps_tta_retry_20260225_013044_sotafair_base/runA.out:15`
  - `logs/eval/abcd_cls_cpac_nepafull_tta_20260225_141831_base/runA.out:15`

Implication:

- No result table from the above eval batches should be labeled as `rfps`-pretrain-based.
- `rfps` claims must wait for checkpoints from jobs `96381`..`96384` and eval reruns that explicitly point to those checkpoints.

## 27. Fine-tune augmentation/LR protocol update (2026-02-25)

Purpose:

- Align fine-tune-side regularization closer to common PointGPT-style point-cloud classification practice.
- Remove weak-augmentation defaults that were effectively "rotation-only" for ScanObjectNN.

### 27.1 Code/script changes

Applied updates:

- `nepa3d/train/finetune_cls.py`
  - `aug_preset=scanobjectnn` is now **strong preset**:
    - `rotate_z=True`
    - `scale_min=0.67`, `scale_max=1.5`
    - `translate=0.2`
    - `jitter_sigma=0.01`, `jitter_clip=0.05`
  - `aug_preset=modelnet40` now includes jitter:
    - `rotate_z=False`
    - `scale_min=0.8`, `scale_max=1.25`
    - `translate=0.1`
    - `jitter_sigma=0.01`, `jitter_clip=0.05`
  - legacy-compatible presets were added:
    - `scanobjectnn_rot_only` (old rotate-only behavior)
    - `modelnet40_legacy` (old no-jitter behavior)
  - added `--aug_recompute_dist` (default `1`):
    - when jitter is enabled and point-distance is used (`ablate_point_dist=0`), recompute `pt_dist` from augmented `pc_xyz` via nearest-neighbor for strict xyz/dist consistency (slower).
- `nepa3d/data/dataset.py`
  - augmentation path now also transforms cached `pc_xyz`/`pc_n` with rigid/scale transforms.
  - jitter-aware strict distance recompute path added (guarded by `aug_recompute_dist`).
- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
  - fine-tune LR default changed: `LR_CLS=5e-4` (was `1e-4`)
  - added `AUG_RECOMPUTE_DIST` env (default `1`) and forwards `--aug_recompute_dist`
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - fine-tune LR default changed: `LR_CLS=5e-4` (was `1e-4`)
  - forwards `AUG_RECOMPUTE_DIST` to qsub jobs
- `scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh`
  - `SCAN_AUG_PRESET` default changed to `scanobjectnn` (was `none`)
- `scripts/pipeline/submit_pretrain_then_sotafair_eval_qf.sh`
  - stage2 `SCAN_AUG_PRESET` default changed to `scanobjectnn` (was `none`)

### 27.2 Reproducibility note

Because `scanobjectnn` preset semantics changed, old logs using `scanobjectnn` are **not** directly comparable unless you pin legacy preset explicitly:

- ScanObjectNN legacy behavior:
  - `SCAN_AUG_PRESET=scanobjectnn_rot_only`
- ModelNet40 legacy behavior:
  - `MODELNET_AUG_PRESET=modelnet40_legacy`

### 27.3 Recommended run protocol (current default-aligned)

- Train sampling:
  - `PT_SAMPLE_MODE_TRAIN_CLS=fps`
- Eval sampling:
  - `PT_SAMPLE_MODE_EVAL_CLS=fps`
- Voting/TTA:
  - `MC_EVAL_K_TEST=10`
  - `AUG_EVAL=1`
- Val split:
  - `VAL_SPLIT_MODE=group_auto`

This preserves FPS-based train/eval sampling while adding stronger geometric/noise augmentation as regularization.

## 28. Run-scope reduction to A/B only (resource control, 2026-02-25)

Purpose:

- Reduce running job count while keeping representative checkpoints/protocols for rapid iteration.

Decision:

- Keep only `Run A` / `Run B` active for the current cycle.
- Stop `Run C` / `Run D` jobs in-flight and defer full A/B/C/D rerun to a later stage.

Termination applied:

- NEPA-full eval (C/D across 4 ablations) terminated by user:
  - `96366`, `96367`, `96370`, `96371`, `96374`, `96375`, `96378`, `96379`
- rfps pretrain (C/D) terminated by user:
  - `96383`, `96384`

Confirmed from scheduler records (`qstat -fx` comments):

- all above jobs: `job_state=F`
- comment includes: `terminated by qch10156fh@qes02`

Remaining active set:

- NEPA-full eval (A/B): `96364`, `96365`, `96368`, `96369`, `96372`, `96373`, `96376`, `96377`
- rfps pretrain (A/B): `96381`, `96382`

## 29. ModelNet40 recheck + cache rebuild + 8-job submit (2026-02-25)

### 29.1 Source dataset verification (`data/ModelNet40`)

Rechecked input tree after replacement:

- OFF (official layout) is present and complete:
  - train: `9843`
  - test: `2468`
  - classes: `40/40` (train/test symmetric)
- `ply_format` tree is partial and should not be used for official evaluation:
  - train: `9843` (40 classes)
  - test: `601` (9 classes)
  - contains one zero-byte file:
    - `data/ModelNet40/ply_format/test/chair/chair_0900_[8].ply`

Conclusion:

- Official OFF tree is valid and should be the source for cache rebuild.

### 29.2 Cache rebuild submission

Submitted full overwrite rebuild for `data/modelnet40_cache_v2`:

- script: `scripts/preprocess/submit_modelnet40_cache_v2_qf.sh`
- key args: `OVERWRITE=1`, `SPLIT=all`, `N_WORKERS=32`
- job: `96491.qjcm`

### 29.3 8 eval jobs submitted (A/B only)

Requested matrix:

- pretrain modes: `fps`, `rfps`
- runs: `A`, `B`
- protocols: `SOTA-fair`, `NEPA-full`
- total: `8` jobs

All jobs were submitted with dependency on ModelNet cache rebuild:

- common dependency: `afterok:96491.qjcm`

Per-job dependency also includes the corresponding pretrain job:

- fps-A depends on `95920.qjcm`
- fps-B depends on `95921.qjcm`
- rfps-A depends on `96381.qjcm`
- rfps-B depends on `96382.qjcm`

Submitted eval jobs:

- `96492.qjcm` `fps_runA_sotafair`
- `96493.qjcm` `fps_runA_nepafull`
- `96494.qjcm` `fps_runB_sotafair`
- `96495.qjcm` `fps_runB_nepafull`
- `96496.qjcm` `rfps_runA_sotafair`
- `96497.qjcm` `rfps_runA_nepafull`
- `96498.qjcm` `rfps_runB_sotafair`
- `96499.qjcm` `rfps_runB_nepafull`

Run/log roots:

- run set: `ab_fps_rfps_2proto_20260225_163448`
- logs: `logs/eval/ab_fps_rfps_2proto_20260225_163448`
- outputs: `runs/eval_ab_fps_rfps_2proto_20260225_163448`
- results: `results/ab_fps_rfps_2proto_20260225_163448`

Protocol settings used:

- SOTA-fair:
  - `PT_XYZ_KEY_CLS=pc_xyz`
  - `ABLATE_POINT_DIST=1`
  - `POINT_ORDER_MODE=morton`
- NEPA-full:
  - `PT_XYZ_KEY_CLS=pt_xyz_pool`
  - `ABLATE_POINT_DIST=0`
  - `POINT_ORDER_MODE=fps`
- Common:
  - `VAL_SPLIT_MODE=group_auto`
  - `PT_SAMPLE_MODE_TRAIN_CLS=fps`
  - `PT_SAMPLE_MODE_EVAL_CLS=fps`
  - `AUG_EVAL=1` (TTA on)
  - `RUN_SCAN=1`, `RUN_MODELNET=1`, `RUN_CPAC=0`

### 29.4 LR rollback (no-sweep policy) and re-submit

User policy update:

- For this batch, do **not** change fine-tune LR.
- Keep the same LR as previous fine-tuning runs; LR sweep is deferred.

Action taken:

1. Stopped the first 8 submitted jobs (they had `LR_CLS=5e-4`).
   - terminated: `96492`..`96499`
2. Restored script defaults:
   - `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`: `LR_CLS` default back to `1e-4`
   - `scripts/eval/submit_abcd_cls_cpac_qf.sh`: `LR_CLS` default back to `1e-4`
3. Re-submitted the same 8-job matrix with explicit `LR_CLS=1e-4`.

Re-submitted jobs:

- `96536.qjcm` `fps_runA_sotafair`
- `96537.qjcm` `fps_runA_nepafull`
- `96538.qjcm` `fps_runB_sotafair`
- `96539.qjcm` `fps_runB_nepafull`
- `96540.qjcm` `rfps_runA_sotafair`
- `96541.qjcm` `rfps_runA_nepafull`
- `96542.qjcm` `rfps_runB_sotafair`
- `96543.qjcm` `rfps_runB_nepafull`

New run/log roots:

- run set: `ab_fps_rfps_2proto_lr1e4_20260225_164957`
- logs: `logs/eval/ab_fps_rfps_2proto_lr1e4_20260225_164957`
- outputs: `runs/eval_ab_fps_rfps_2proto_lr1e4_20260225_164957`
- results: `results/ab_fps_rfps_2proto_lr1e4_20260225_164957`

### 29.5 Additional LR sensitivity check (fps A/B only, 5e-4)

For a controlled A/B-only comparison request, added a second batch with:

- pretrain mode: `fps` only
- runs: `A`, `B`
- protocols: `SOTA-fair`, `NEPA-full`
- LR override: `LR_CLS=5e-4`
- common settings kept same (`group_auto`, `fps/fps`, `AUG_EVAL=1`, `RUN_CPAC=0`)

Submitted jobs:

- `96546.qjcm` `fps5e4_runA_sotafair`
- `96547.qjcm` `fps5e4_runA_nepafull`
- `96548.qjcm` `fps5e4_runB_sotafair`
- `96549.qjcm` `fps5e4_runB_nepafull`

Dependency:

- `afterok:96491.qjcm` (ModelNet cache rebuild) + corresponding fps pretrain completion.

Run/log roots:

- run set: `ab_fps_lr5e4_abtest_20260225_165338`
- logs: `logs/eval/ab_fps_lr5e4_abtest_20260225_165338`
- outputs: `runs/eval_ab_fps_lr5e4_abtest_20260225_165338`
- results: `results/ab_fps_lr5e4_abtest_20260225_165338`

## 30. Pretrain RFPS + augmentation (A/B) intake (2026-02-25)

Purpose:

- Add RFPS + mild geometric augmentation support to **pretrain** path (not only fine-tune).
- Launch A/B pretrain runs with:
  - `pt_sample_mode_train=rfps`
  - augmentation (mild): rotate-z + scale + jitter

### 30.1 Code/script changes

Pretrain CLI and dataset wiring:

- `nepa3d/train/pretrain.py`
  - added args:
    - `--aug_rotate_z`
    - `--aug_scale_min`, `--aug_scale_max`
    - `--aug_translate`
    - `--aug_jitter_sigma`, `--aug_jitter_clip`
    - `--aug_recompute_dist`
  - aug config is now printed in startup token-config line.
  - forwarded aug args to both:
    - mixed pretrain builder path
    - single-dataset `ModelNet40QueryDataset` path
- `nepa3d/data/mixed_pretrain.py`
  - `build_mixed_pretrain(...)` now accepts and forwards pretrain aug args to each dataset.

Pretrain shell wrappers:

- `scripts/pretrain/nepa3d_pretrain.sh`
  - added env knobs and forwarding:
    - `AUG_ROTATE_Z`
    - `AUG_SCALE_MIN`, `AUG_SCALE_MAX`
    - `AUG_TRANSLATE`
    - `AUG_JITTER_SIGMA`, `AUG_JITTER_CLIP`
    - `AUG_RECOMPUTE_DIST`
- `scripts/pretrain/nepa3d_pretrain_pointcloud.sh`
  - same aug knob forwarding added for consistency.

New submit helper:

- `scripts/pretrain/submit_pretrain_ab_rfps_aug_qf.sh`
  - submits **A/B only**
  - default pretrain mode: `rfps`
  - default aug (mild):
    - `AUG_ROTATE_Z=1`
    - `AUG_SCALE_MIN=0.8`
    - `AUG_SCALE_MAX=1.25`
    - `AUG_TRANSLATE=0.0`
    - `AUG_JITTER_SIGMA=0.01`
    - `AUG_JITTER_CLIP=0.05`
    - `AUG_RECOMPUTE_DIST=0` (off by default; slower when on)

### 30.2 Submitted pretrain jobs (A/B, rfps+aug)

Submitted via new helper:

- `96560.qjcm` `runA_rfps_aug_rfps_aug_ab_20260225_171018`
- `96561.qjcm` `runB_rfps_aug_rfps_aug_ab_20260225_171018`

Job-id record:

- `logs/pretrain/rfps_aug_ab_20260225_171018_job_ids.txt`

Checkpoint roots (expected):

- `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA`
- `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB`

## 31. LLRD mode update (linear) + A/B-only eval submission (2026-02-25)

### 31.1 Code changes

Implemented configurable LLRD schedule mode for fine-tuning:

- `nepa3d/train/finetune_cls.py`
  - added CLI:
    - `--llrd_mode {exp,linear}` (default: `exp` for backward compatibility)
  - `exp` mode (legacy): `lr_scale = llrd^(max_layer_idx - layer_idx)`
  - `linear` mode:
    - shallowest (`layer_idx=0`) -> `lr_scale=llrd`
    - deepest (`layer_idx=max_layer_idx`) -> `lr_scale=1.0`
    - linear interpolation in between
  - startup log and `[llrd]` summary now print `llrd_mode`.

Eval script wiring:

- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`
  - added env: `LLRD_MODE` (default `exp`)
  - forwards `--llrd_mode` to both ScanObjectNN and ModelNet40 fine-tune commands
  - startup config log now prints `llrd_mode`
- `scripts/eval/submit_abcd_cls_cpac_qf.sh`
  - forwards `LLRD_MODE` through qsub variables

### 31.2 A/B-only eval jobs submitted (linear LLRD)

Requested scope:

- A/B only
- linear LLRD validation run

Submitted jobs:

- `97033.qjcm` (`runA_llrdlin`)
- `97034.qjcm` (`runB_llrdlin`)

Note:

- Initial submission `97031/97032` was replaced immediately to enforce `VAL_SPLIT_MODE=group_auto`.
- Active run set is the re-submitted one below.

Submission settings:

- `LLRD=0.35`
- `LLRD_MODE=linear`
- `VAL_SPLIT_MODE=group_auto`
- other knobs kept at script defaults (no LR sweep change; default `LR_CLS=1e-4`)

Run roots:

- run set: `ab_llrd_linear035_groupauto_20260225_213924`
- logs: `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924`
- outputs: `runs/eval_ab_ab_llrd_linear035_groupauto_20260225_213924`
- results: `results/ab_ab_llrd_linear035_groupauto_20260225_213924`

### 31.3 Final metrics (A/B, linear LLRD)

Source logs/results:

- `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924/runA_llrdlin_classification_scan.log`
- `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924/runB_llrdlin_classification_scan.log`
- `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924/runA_llrdlin_classification_modelnet.log`
- `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924/runB_llrdlin_classification_modelnet.log`
- `results/ab_ab_llrd_linear035_groupauto_20260225_213924/cpac_abcd_1024_runA_llrdlin.json`
- `results/ab_ab_llrd_linear035_groupauto_20260225_213924/cpac_abcd_1024_runB_llrdlin.json`

| Run | ScanObjectNN `test_acc` | ModelNet40 `test_acc` | CPAC `mae` | CPAC `rmse` | CPAC `iou@tau` |
|---|---:|---:|---:|---:|---:|
| `runA_llrdlin` | 0.6300 | 0.8639 | 0.0581 | 0.0815 | 0.6326 |
| `runB_llrdlin` | 0.6207 | 0.8659 | 0.0955 | 0.1303 | 0.4925 |

## 32. Augmentation fine-tune results (A/B) + no-augmentation comparison (2026-02-25)

### 32.1 Status summary

- Early batch `ab_fps_rfps_2proto_20260225_163448` (`96492`..`96499`) is treated as superseded/incomplete.
  - `classification_scan.log` in this root stops near early epochs for several runs.
- Completed batches used for reporting:
  - `ab_fps_rfps_2proto_lr1e4_20260225_164957`
  - `ab_fps_lr5e4_abtest_20260225_165338`
- In these batches:
  - `RUN_CPAC=0` (classification-only, CPAC skipped)
  - `aug_preset=scanobjectnn` for ScanObjectNN stage
  - `AUG_EVAL=1` (TTA on), `PT_SAMPLE_MODE_{TRAIN,EVAL}=fps`

### 32.2 Final metrics (`ScanObjectNN` / `ModelNet40`)

Source roots:

- `logs/eval/ab_fps_rfps_2proto_lr1e4_20260225_164957`
- `logs/eval/ab_fps_lr5e4_abtest_20260225_165338`

`LR_CLS=1e-4` batch:

| setting | ScanObjectNN `test_acc` | ModelNet40 `test_acc` |
|---|---:|---:|
| `fps_runA_sotafair` | 0.6606 | 0.8620 |
| `fps_runA_nepafull` | 0.3341 | 0.8327 |
| `fps_runB_sotafair` | 0.6654 | 0.8695 |
| `fps_runB_nepafull` | 0.5389 | 0.8682 |
| `rfps_runA_sotafair` | 0.6571 | 0.8734 |
| `rfps_runA_nepafull` | 0.3298 | 0.8291 |
| `rfps_runB_sotafair` | 0.6654 | 0.8717 |
| `rfps_runB_nepafull` | 0.5660 | 0.8620 |

`LR_CLS=5e-4` batch (fps A/B only):

| setting | ScanObjectNN `test_acc` | ModelNet40 `test_acc` |
|---|---:|---:|
| `fps5e4_runA_sotafair` | 0.6875 | 0.8825 |
| `fps5e4_runA_nepafull` | 0.3074 | 0.8649 |
| `fps5e4_runB_sotafair` | 0.6762 | 0.8786 |
| `fps5e4_runB_nepafull` | 0.6042 | 0.8874 |

### 32.3 Comparison against augmentation-none baseline

Explicit augmentation-none baseline (historical, Run A only) from §18:

| baseline run | point order | aug preset | ScanObjectNN `test_acc` |
|---|---|---|---:|
| `runA_fps_augnone` | `fps` | `none` | 0.6969 |
| `runA_morton_augnone` | `morton` | `none` | 0.6976 |

Comparison (Run A, SOTA-fair) using morton no-aug baseline `0.6976`:

| setting | ScanObjectNN `test_acc` | delta vs no-aug (`0.6976`) |
|---|---:|---:|
| strong-aug + `LR_CLS=1e-4` (`fps_runA_sotafair`) | 0.6606 | -0.0370 |
| strong-aug + `LR_CLS=5e-4` (`fps5e4_runA_sotafair`) | 0.6875 | -0.0101 |

Notes:

- No explicit augmentation-none counterpart was run for `Run B` in this A/B batch, so B-side no-aug delta is unavailable.
- The no-aug baseline above is from a different historical batch/checkpoint context; use as directional reference, not strict apples-to-apples proof.

## 33. RFPS dual-mask status + 256 ON/OFF comparison launch (2026-02-26)

### 33.1 Current RFPS pretrain dual-mask status (confirmed OFF)

Checked current RFPS+aug checkpoints:

- `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt`
- `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt`

Checkpoint args confirm:

- `pt_sample_mode_train=rfps`
- `dual_mask_near=0.0`
- `dual_mask_far=0.0`

So current RFPS pretrain runs are dual-mask OFF in effect.

### 33.2 Script wiring update (dual-mask env pass-through)

Updated wrappers so dual-mask can be controlled from qsub env:

- `scripts/pretrain/nepa3d_pretrain.sh`
- `scripts/pretrain/nepa3d_pretrain_pointcloud.sh`
- `scripts/pretrain/submit_pretrain_ab_rfps_aug_qf.sh`

Added/forwarded env knobs:

- `DUAL_MASK_NEAR`, `DUAL_MASK_FAR`
- `DUAL_MASK_WINDOW`, `DUAL_MASK_WARMUP_FRAC`
- `DUAL_MASK_TYPE_AWARE`
- `DUAL_MASK_WINDOW_SCALE`, `DUAL_MASK_WINDOW_REF_TOTAL`

### 33.3 New compact comparison launcher (A/B, 256)

Added:

- `scripts/pretrain/submit_pretrain_ab_rfps_aug_dualmask256_qf.sh`

Purpose:

- A/B only, `PT_SAMPLE_MODE_TRAIN=rfps`, with augmentation
- compact token budget (`n_point=256`; A uses `n_ray=256`, B uses `n_ray=0`)
- direct dual-mask ON/OFF comparison

### 33.4 Submitted jobs (A/B x dual-mask OFF/ON)

Submitted run set:

- `rfps_aug_dm256_20260226_001557`

Jobs:

- `97137.qjcm` `runA_rfps256_dmoff_rfps_aug_dm256_20260226_001557`
- `97138.qjcm` `runA_rfps256_dmon_rfps_aug_dm256_20260226_001557`
- `97139.qjcm` `runB_rfps256_dmoff_rfps_aug_dm256_20260226_001557`
- `97140.qjcm` `runB_rfps256_dmon_rfps_aug_dm256_20260226_001557`

Job-id record:

- `logs/pretrain/rfps_aug_dualmask256_job_ids.txt`

## 34. 256-aligned fine-tune/eval launch for dual-mask comparison (2026-02-26)

Purpose:

- Evaluate 256 pretrain checkpoints with matched 256 fine-tune/eval settings.
- Keep protocol split style from prior A/B 2-proto runs.

### 34.1 New submit helper

Added:

- `scripts/eval/submit_ab_dualmask256_2proto_qf.sh`

Behavior:

- matrix: `A/B` x `dual-mask {off,on}` x `{SOTA-fair, NEPA-full}` = 8 jobs
- each eval job depends on the corresponding pretrain job:
  - A-dmoff -> `afterok:97137.qjcm`
  - A-dmon -> `afterok:97138.qjcm`
  - B-dmoff -> `afterok:97139.qjcm`
  - B-dmon -> `afterok:97140.qjcm`

### 34.2 256 eval settings (common)

- `N_POINT_CLS=256`
- `N_RAY_CLS=0`
- `PT_SAMPLE_MODE_TRAIN_CLS=fps`
- `PT_SAMPLE_MODE_EVAL_CLS=fps`
- `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
- `VAL_SPLIT_MODE=group_auto`
- `RUN_SCAN=1`, `RUN_MODELNET=1`, `RUN_CPAC=0`
- `LR_CLS=1e-4` (no LR sweep policy)

Protocol-specific mapping:

- SOTA-fair:
  - `PT_XYZ_KEY_CLS=pc_xyz`
  - `ABLATE_POINT_DIST=1`
  - `POINT_ORDER_MODE=morton`
- NEPA-full:
  - `PT_XYZ_KEY_CLS=pt_xyz_pool`
  - `ABLATE_POINT_DIST=0`
  - `POINT_ORDER_MODE=fps`

### 34.3 Submitted jobs

Run set:

- `dualmask256_eval_20260226_002435`

Submitted:

- `97142.qjcm` `runA_dmoff_sotafair`
- `97143.qjcm` `runA_dmoff_nepafull`
- `97144.qjcm` `runA_dmon_sotafair`
- `97145.qjcm` `runA_dmon_nepafull`
- `97146.qjcm` `runB_dmoff_sotafair`
- `97147.qjcm` `runB_dmoff_nepafull`
- `97148.qjcm` `runB_dmon_sotafair`
- `97149.qjcm` `runB_dmon_nepafull`

Job-id record:

- `logs/eval/ab_dualmask256_eval_job_ids.txt`

Run roots:

- logs: `logs/eval/ab_dualmask256_2proto_dualmask256_eval_20260226_002435`
- outputs: `runs/eval_ab_dualmask256_2proto_dualmask256_eval_20260226_002435`
- results: `results/ab_dualmask256_2proto_dualmask256_eval_20260226_002435`

### 34.4 Intermediate results already finalized in this batch

Completed eval jobs so far (`job_state=F`, `Exit_status=0`):

- `97142.qjcm` `runA_dmoff_sotafair`
- `97143.qjcm` `runA_dmoff_nepafull`
- `97144.qjcm` `runA_dmon_sotafair`
- `97145.qjcm` `runA_dmon_nepafull`
- `97146.qjcm` `runB_dmoff_sotafair`
- `97147.qjcm` `runB_dmoff_nepafull`
- `97148.qjcm` `runB_dmon_sotafair`
- `97149.qjcm` `runB_dmon_nepafull`

Final metrics:

| run | protocol | ScanObjectNN `test_acc` | ModelNet40 `test_acc` |
|---|---|---:|---:|
| `runA_dmoff_sotafair` | SOTA-fair | 0.6103 | 0.8665 |
| `runA_dmon_sotafair` | SOTA-fair | 0.6130 | 0.8714 |
| `runA_dmoff_nepafull` | NEPA-full | 0.2955 | 0.5677 |
| `runA_dmon_nepafull` | NEPA-full | 0.4478 | 0.8418 |
| `runB_dmoff_sotafair` | SOTA-fair | 0.7159 | 0.8880 |
| `runB_dmon_sotafair` | SOTA-fair | 0.7154 | 0.8929 |
| `runB_dmoff_nepafull` | NEPA-full | 0.5246 | 0.8551 |
| `runB_dmon_nepafull` | NEPA-full | 0.5089 | 0.8597 |

Notes:

- A/B all 8 jobs in this 256 dual-mask eval matrix are complete.
- This run set used `RUN_CPAC=0` (classification-only); CPAC metrics are not included here.

## 35. Live execution tracker (running/held/completed snapshot, 2026-02-26)

Purpose:

- Make it explicit which results are already finalized vs still in-flight.
- Avoid "missing results" ambiguity by tying each unreported item to job state.

### 35.1 Newly completed since last update

Pretrain (256 dual-mask comparison, B side):

- `97139.qjcm` `runB_rfps256_dmoff_rfps_aug_dm256_20260226_001557`:
  - `job_state=F`, `Exit_status=0`
  - checkpoint: `runs/pretrain_ab_256_rfps_aug_dualmask_rfps_aug_dm256_20260226_001557_runB_dmoff/last.pt`
- `97140.qjcm` `runB_rfps256_dmon_rfps_aug_dm256_20260226_001557`:
  - `job_state=F`, `Exit_status=0`
  - checkpoint: `runs/pretrain_ab_256_rfps_aug_dualmask_rfps_aug_dm256_20260226_001557_runB_dmon/last.pt`

Eval (linear-LLRD A/B batch, B side):

- `97034.qjcm` `runB_llrdlin`:
  - `job_state=F`, `Exit_status=0`
  - ScanObjectNN: `best_val=0.7023`, `best_ep=99`, `test_acc=0.6207`
  - ModelNet40: `best_val=0.8677`, `best_ep=64`, `test_acc=0.8659`
  - CPAC json: `results/ab_ab_llrd_linear035_groupauto_20260225_213924/cpac_abcd_1024_runB_llrdlin.json`

### 35.2 Running jobs (what each is)

As of this snapshot (`qselect -u $USER` / `qstat -x`):

- `96560.qjcm`:
  - `runA_rfps_aug_rfps_aug_ab_20260225_171018` pretrain (A, 1024, rfps+aug)
  - logs: `logs/ddp_pretrain/ddp_pretrain_96560.qjcm_runA_rfps_aug_rfps_aug_ab_20260225_171018`
- `97137.qjcm`:
  - `runA_rfps256_dmoff_rfps_aug_dm256_20260226_001557` pretrain (A, 256, dual-mask OFF)
  - logs: `logs/ddp_pretrain/ddp_pretrain_97137.qjcm_runA_rfps256_dmoff_rfps_aug_dm256_20260226_001557`
- `97138.qjcm`:
  - `runA_rfps256_dmon_rfps_aug_dm256_20260226_001557` pretrain (A, 256, dual-mask ON)
  - logs: `logs/ddp_pretrain/ddp_pretrain_97138.qjcm_runA_rfps256_dmon_rfps_aug_dm256_20260226_001557`
- `97146.qjcm`, `97147.qjcm`, `97148.qjcm`, `97149.qjcm`:
  - B-side 256 eval (`dmoff/dmon` x `sotafair/nepafull`)
  - logs: `logs/eval/ab_dualmask256_2proto_dualmask256_eval_20260226_002435`
- `97033.qjcm`:
  - `runA_llrdlin` eval (linear LLRD, A side)
  - logs: `logs/eval/ab_cls_cpac_ab_llrd_linear035_groupauto_20260225_213924`
- `96540.qjcm`, `96541.qjcm`:
  - legacy rfps-A eval (`sotafair` / `nepafull`, 1024 setting)
  - logs: `logs/eval/ab_fps_rfps_2proto_lr1e4_20260225_164957`

### 35.3 Held jobs (waiting for dependencies)

- `97142.qjcm` `runA_dmoff_sotafair` (`afterok:97137`)
- `97143.qjcm` `runA_dmoff_nepafull` (`afterok:97137`)
- `97144.qjcm` `runA_dmon_sotafair` (`afterok:97138`)
- `97145.qjcm` `runA_dmon_nepafull` (`afterok:97138`)

Meaning:

- These are not missing; they are intentionally queued behind A-side 256 pretrain completion.

### 35.4 "Unrecorded result" interpretation rule

If a result table is not yet written in this doc, classify it first:

1. `job_state=R`: still running (no final metric yet).
2. `job_state=H`: dependency wait (not started yet).
3. `job_state=F` with `Exit_status=0`: should be harvested and appended.

At this snapshot, unrecorded items are explained by (1) or (2), except `97034` which is now harvested above.

### 35.5 Next write-back plan

- When `97137/97138` finish:
  - start confirmation of `97142`..`97145` transition from `H -> R`
- When `97146`..`97149` and `97142`..`97145` finish:
  - append final 8-way table (`A/B`, `dmoff/dmon`, `sotafair/nepafull`) for:
    - ScanObjectNN `best_val/best_ep/test_acc`
    - ModelNet40 `best_val/best_ep/test_acc`

### 35.6 Refresh snapshot (newly finished + state transitions)

Additional completed jobs confirmed after the previous snapshot:

- `96560.qjcm`:
  - `runA_rfps_aug_rfps_aug_ab_20260225_171018` pretrain (A, 1024 rfps+aug)
  - `job_state=F`, `Exit_status=0`
  - checkpoint: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt`
- `97137.qjcm`:
  - `runA_rfps256_dmoff_rfps_aug_dm256_20260226_001557` pretrain
  - `job_state=F`, `Exit_status=0`
  - checkpoint: `runs/pretrain_ab_256_rfps_aug_dualmask_rfps_aug_dm256_20260226_001557_runA_dmoff/last.pt`
  - checkpoint args: `dual_mask_near=0.0`, `dual_mask_far=0.0`
- `97138.qjcm`:
  - `runA_rfps256_dmon_rfps_aug_dm256_20260226_001557` pretrain
  - `job_state=F`, `Exit_status=0`
  - checkpoint: `runs/pretrain_ab_256_rfps_aug_dualmask_rfps_aug_dm256_20260226_001557_runA_dmon/last.pt`
  - checkpoint args: `dual_mask_near=0.4`, `dual_mask_far=0.1`
- `97146.qjcm` / `97148.qjcm`:
  - B-side 256 eval (`dmoff_sotafair` / `dmon_sotafair`)
  - both `job_state=F`, `Exit_status=0`
  - metrics are summarized in §34.4

State transition confirmed:

- `97142`..`97145` moved from `H` to `R` (dependency satisfied).

Current non-finished jobs at this refresh:

- running:
  - `96540`, `96541`, `97033`, `97142`, `97143`, `97144`, `97145`, `97147`, `97149`
- held:
  - none

### 35.7 Refresh snapshot (latest check after 35.6)

Additional completion confirmed:

- `96561.qjcm`:
  - `runB_rfps_aug_rfps_aug_ab_20260225_171018` pretrain (B, 1024 rfps+aug)
  - `job_state=F`, `Exit_status=0`
  - `obittime=2026-02-25 21:50:32`
  - checkpoint: `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt`

No additional newly finished eval jobs were found in this refresh (relative to §34.4 / §35.6).

Current non-finished jobs:

- running:
  - `96540` `eval_rfps_runA_sotafair`
  - `96541` `eval_rfps_runA_nepafull`
  - `97033` `eval_runA_llrdlin`
  - `97142` `eval_runA_dmoff_sotafair`
  - `97143` `eval_runA_dmoff_nepafull`
  - `97144` `eval_runA_dmon_sotafair`
  - `97145` `eval_runA_dmon_nepafull`
- held:
  - none

### 35.8 Refresh snapshot (B-side 256 eval all complete)

Additional eval completions after §35.7:

- `97147.qjcm` `runB_dmoff_nepafull`: `job_state=F`, `Exit_status=0`
- `97149.qjcm` `runB_dmon_nepafull`: `job_state=F`, `Exit_status=0`

Their final metrics are now reflected in §34.4.

Current non-finished jobs:

- running:
  - `96540` `eval_rfps_runA_sotafair`
  - `96541` `eval_rfps_runA_nepafull`
  - `97033` `eval_runA_llrdlin`
  - `97142` `eval_runA_dmoff_sotafair`
  - `97143` `eval_runA_dmoff_nepafull`
  - `97144` `eval_runA_dmon_sotafair`
  - `97145` `eval_runA_dmon_nepafull`
- held:
  - none

### 35.9 Refresh snapshot (A-side completions + current queue)

Additional completions confirmed:

- `97033.qjcm` `runA_llrdlin`: `job_state=F`, `Exit_status=0`
  - ScanObjectNN: `best_val=0.7332`, `best_ep=97`, `test_acc=0.6300`
  - ModelNet40: `best_val=0.8701`, `best_ep=67`, `test_acc=0.8639`
  - CPAC summary (from `runA_llrdlin.out`):
    - `mae=0.0581`, `rmse=0.0815`, `iou@tau=0.6326`
  - CPAC json: `results/ab_ab_llrd_linear035_groupauto_20260225_213924/cpac_abcd_1024_runA_llrdlin.json`
- `97142.qjcm`..`97145.qjcm`:
  - A-side 256 dual-mask eval (`runA_dmoff/dmon` x `sotafair/nepafull`) all `job_state=F`, `Exit_status=0`
  - metrics are reflected in §34.4.

Current non-finished jobs at this snapshot:

- running:
  - `96540` `eval_rfps_runA_sotafair`
  - `96541` `eval_rfps_runA_nepafull`
  - `97182` `pt_b00_interleave_theta`
  - `97183` `pt_b01_split_theta`
  - `97184` `pt_b02_split_theta_typepos`
  - `97185` `pt_b03_split_viewraster_typepos`
  - `97186` `pt_b04_split_xanchor_morton_typepos`
  - `97187` `pt_b05_split_xanchor_fps_typepos`
  - `97188` `pt_b06_split_dirfps_typepos`
  - `97189` `pt_b07_event_xanchor_typepos`
  - `97190` `pt_b08_event_dirfps_typepos`
- held:
  - `97191`..`97208` (all eval jobs in `a256_queryrethink` set; waiting `afterok` on the corresponding pretrain jobs)

## 36. Query-rethink merge (split+SEP / ordering / type-pos) (2026-02-26)

Purpose:

- Merge `rethink_query` implementation into mainline while keeping existing training pipeline behavior by default.
- Add the extra Ray ordering requested in this review: **direction-space FPS**.

### 36.1 Core code merged from `rethink_query`

Merged into `nepa3d`:

- `nepa3d/token/ordering.py`
- `nepa3d/token/tokenizer.py`
- `nepa3d/data/dataset.py`
- `nepa3d/data/mixed_pretrain.py`
- `nepa3d/train/pretrain.py`
- `nepa3d/models/query_nepa.py`
- `nepa3d/utils/ckpt_utils.py`
- `nepa3d/analysis/retrieval_ucpr.py`
- `nepa3d/analysis/completion_cpac_udf.py`
- `nepa3d/analysis/qualitative_cpac_marching_cubes.py`

Manually merged (to preserve current LLRD behavior):

- `nepa3d/train/finetune_cls.py`
  - kept existing `--llrd_mode {exp,linear}` logic
  - added ckpt compatibility for resized `type_emb` / `type_pos_emb`
  - added `type_specific_pos` restoration from checkpoint args

### 36.2 New tokenizer / pretrain capabilities

Added:

- `qa_layout=split_sep`:
  - sequence becomes `[BOS] + Q... + [SEP] + A... (+[EOS])`
  - new token type: `TYPE_SEP=9`
- `sequence_mode={block,event}` (`bundle` alias -> `event`)
- `event_order_mode={morton,fps,random}`
- `ray_order_mode={theta_phi,view_raster,x_anchor_morton,x_anchor_fps,random,none}`
- `type_specific_pos` (type-local positional embedding)
- ckpt load compatibility for type-vocab/shape changes:
  - `maybe_resize_type_emb_in_state_dict`
  - `maybe_resize_type_pos_emb_in_state_dict`

### 36.3 Additional ordering implemented in this pass

Requested addition implemented:

- **Ray direction FPS ordering**:
  - new function: `sort_rays_by_direction_fps(ray_d)` in `nepa3d/token/ordering.py`
  - new `ray_order_mode` aliases in tokenizer:
    - `dir_fps`, `direction_fps`, `ray_fps`, `s2_fps`
  - wired into `pretrain.py` CLI choices (`--ray_order_mode dir_fps`)

### 36.4 Pretrain launcher wiring update

Updated wrappers:

- `scripts/pretrain/nepa3d_pretrain.sh`
- `scripts/pretrain/nepa3d_pretrain_pointcloud.sh`
- `scripts/pretrain/submit_pretrain_ab_rfps_aug_qf.sh`
- `scripts/pretrain/submit_pretrain_ab_rfps_aug_dualmask256_qf.sh`

New env passthroughs:

- `SEQUENCE_MODE`, `EVENT_ORDER_MODE`
- `RAY_ORDER_MODE`, `RAY_ANCHOR_MISS_T`, `RAY_VIEW_TOL`
- `TYPE_SPECIFIC_POS`
- `QA_LAYOUT` exposed in RFPS submit helpers for split/split_sep switching

Compatibility note:

- Defaults are set to prior behavior (`QA_LAYOUT=interleave`, `SEQUENCE_MODE=block`, `RAY_ORDER_MODE=theta_phi`, `TYPE_SPECIFIC_POS=0`), so existing runs are not implicitly changed.

### 36.5 Sanity check

Post-merge smoke check passed:

- `python -m py_compile` on all changed Python files: **passed**
- `build_sequence(..., qa_layout='split_sep', sequence_mode='event', ray_order_mode='dir_fps')` minimal runtime check: **passed**

## 37. A-only 256 query-rethink ablation submission (CPAC+chamfer enabled) (2026-02-26)

Purpose:

- Small-scale (`256`) ablation for newly merged query serialization methods.
- A-only pretrain (`n_point=256`, `n_ray=256`, `qa_tokens=1`) with 9 ordering/layout variants.
- Per-checkpoint eval on both protocols (SOTA-fair / NEPA-full), with CPAC enabled.
- CPAC mesh reconstruction/chamfer metrics enabled in the same eval pipeline.

### 37.1 Eval pipeline update for CPAC mesh metrics

Updated:

- `scripts/eval/nepa3d_eval_cls_cpac_qf.sh`

Added CPAC mesh env passthroughs to `completion_cpac_udf.py`:

- `CPAC_MESH_EVAL`
- `CPAC_MESH_EVAL_MAX_SHAPES`
- `CPAC_MESH_GRID_RES`
- `CPAC_MESH_CHUNK_N_QUERY`
- `CPAC_MESH_MC_LEVEL`
- `CPAC_MESH_NUM_SAMPLES`
- `CPAC_MESH_FSCORE_TAU`
- `CPAC_MESH_SAVE_DIR`
- `CPAC_MESH_STORE_PER_SHAPE`

### 37.2 New submit helper

Added:

- `scripts/pipeline/submit_a256_queryrethink_ablation_qf.sh`

Behavior:

- submit 9 pretrain jobs (A-only, 256)
- submit 18 eval jobs with dependency (`afterok:<corresponding_pretrain_job>`)
  - each pretrain checkpoint -> `{sotafair, nepafull}` eval
- eval uses:
  - `RUN_SCAN=1`, `RUN_MODELNET=1`, `RUN_CPAC=1`
  - `CPAC_MESH_EVAL=1` (Chamfer/F-score path enabled)

### 37.3 Submitted run set

Run set:

- `a256_queryrethink_20260226_024537`

Pretrain jobs (9):

- `97182.qjcm` `pt_b00_interleave_theta`
- `97183.qjcm` `pt_b01_split_theta`
- `97184.qjcm` `pt_b02_split_theta_typepos`
- `97185.qjcm` `pt_b03_split_viewraster_typepos`
- `97186.qjcm` `pt_b04_split_xanchor_morton_typepos`
- `97187.qjcm` `pt_b05_split_xanchor_fps_typepos`
- `97188.qjcm` `pt_b06_split_dirfps_typepos`
- `97189.qjcm` `pt_b07_event_xanchor_typepos`
- `97190.qjcm` `pt_b08_event_dirfps_typepos`

Eval jobs (18, dependency-held at submit time):

- `97191.qjcm` `ev_b00_interleave_theta_sotafair`
- `97192.qjcm` `ev_b00_interleave_theta_nepafull`
- `97193.qjcm` `ev_b01_split_theta_sotafair`
- `97194.qjcm` `ev_b01_split_theta_nepafull`
- `97195.qjcm` `ev_b02_split_theta_typepos_sotafair`
- `97196.qjcm` `ev_b02_split_theta_typepos_nepafull`
- `97197.qjcm` `ev_b03_split_viewraster_typepos_sotafair`
- `97198.qjcm` `ev_b03_split_viewraster_typepos_nepafull`
- `97199.qjcm` `ev_b04_split_xanchor_morton_typepos_sotafair`
- `97200.qjcm` `ev_b04_split_xanchor_morton_typepos_nepafull`
- `97201.qjcm` `ev_b05_split_xanchor_fps_typepos_sotafair`
- `97202.qjcm` `ev_b05_split_xanchor_fps_typepos_nepafull`
- `97203.qjcm` `ev_b06_split_dirfps_typepos_sotafair`
- `97204.qjcm` `ev_b06_split_dirfps_typepos_nepafull`
- `97205.qjcm` `ev_b07_event_xanchor_typepos_sotafair`
- `97206.qjcm` `ev_b07_event_xanchor_typepos_nepafull`
- `97207.qjcm` `ev_b08_event_dirfps_typepos_sotafair`
- `97208.qjcm` `ev_b08_event_dirfps_typepos_nepafull`

Submitted job list:

- `logs/eval/a256_queryrethink_a256_queryrethink_20260226_024537/submitted_jobs_a256_queryrethink_20260226_024537.txt`

### 37.4 Snapshot right after submission

Job states:

- pretrain `97182`..`97190`: `R`
- eval `97191`..`97208`: `H` (waiting for pretrain dependencies)

Confirmed in eval job variables (`97191.qjcm`):

- `RUN_CPAC=1`
- `CPAC_MESH_EVAL=1`
- `CPAC_MESH_EVAL_MAX_SHAPES=800`
- `CPAC_N_CONTEXT=256`, `CPAC_N_QUERY=256`, `CPAC_MAX_LEN=1300`

### 37.5 Final status check (all jobs finished) (2026-02-26)

Queue state:

- `qselect -u $USER` returned empty (no running/held jobs).

`rfps + augmentation (1024)` completion:

- `96560.qjcm` (`runA_rfps_aug_rfps_aug_ab_20260225_171018`): `F`, `Exit_status=0`
- `96561.qjcm` (`runB_rfps_aug_rfps_aug_ab_20260225_171018`): `F`, `Exit_status=0`

`a256_queryrethink` ablation completion:

- pretrain `97182`..`97190`: all `F`, `Exit_status=0`
- eval `97191`..`97208`: all `F`, `Exit_status=1`

Common eval failure reason (`97191`..`97208`):

- CPAC mesh-eval stage failed with max-length mismatch:
  - `ValueError: mesh_eval requires sequence length 1538, but model max_len=1300`
  - source logs: `logs/eval/a256_queryrethink_a256_queryrethink_20260226_024537/*.out`

Interpretation:

- classification stages (ScanObjectNN / ModelNet40) reached final `test_acc` lines in the eval logs,
- but job exit became non-zero at CPAC mesh-eval, so this batch is not a clean end-to-end success.

### 37.6 Classification results extracted from failed eval bundle

Even though all `97191`..`97208` jobs failed at CPAC mesh-eval, classification logs contain final metrics.

Source root:

- `logs/eval/a256_queryrethink_a256_queryrethink_20260226_024537`

ScanObjectNN (`*_classification_scan.log`):

| variant | protocol | best_val | best_ep | test_acc |
|---|---|---:|---:|---:|
| `b00_interleave_theta` | `sotafair` | 0.7756 | 98 | 0.5548 |
| `b00_interleave_theta` | `nepafull` | 0.3181 | 49 | 0.3045 |
| `b01_split_theta` | `sotafair` | 0.7656 | 92 | 0.5210 |
| `b01_split_theta` | `nepafull` | 0.3261 | 24 | 0.3161 |
| `b02_split_theta_typepos` | `sotafair` | 0.8141 | 99 | 0.6003 |
| `b02_split_theta_typepos` | `nepafull` | 0.3007 | 28 | 0.2930 |
| `b03_split_viewraster_typepos` | `sotafair` | 0.7841 | 92 | 0.5324 |
| `b03_split_viewraster_typepos` | `nepafull` | 0.3365 | 38 | 0.3067 |
| `b04_split_xanchor_morton_typepos` | `sotafair` | 0.8638 | 89 | 0.6335 |
| `b04_split_xanchor_morton_typepos` | `nepafull` | 0.3444 | 33 | 0.3354 |
| `b05_split_xanchor_fps_typepos` | `sotafair` | 0.8482 | 87 | 0.6021 |
| `b05_split_xanchor_fps_typepos` | `nepafull` | 0.3233 | 45 | 0.3049 |
| `b06_split_dirfps_typepos` | `sotafair` | 0.8349 | 90 | 0.6015 |
| `b06_split_dirfps_typepos` | `nepafull` | 0.3269 | 35 | 0.3069 |
| `b07_event_xanchor_typepos` | `sotafair` | 0.7334 | 98 | 0.5670 |
| `b07_event_xanchor_typepos` | `nepafull` | 0.3017 | 39 | 0.2840 |
| `b08_event_dirfps_typepos` | `sotafair` | 0.8073 | 99 | 0.5889 |
| `b08_event_dirfps_typepos` | `nepafull` | 0.3065 | 28 | 0.2957 |

ModelNet40 (`*_classification_modelnet.log`):

| variant | protocol | best_val | best_ep | test_acc |
|---|---|---:|---:|---:|
| `b00_interleave_theta` | `sotafair` | 0.8677 | 45 | 0.8509 |
| `b00_interleave_theta` | `nepafull` | 0.6274 | 99 | 0.6152 |
| `b01_split_theta` | `sotafair` | 0.8711 | 88 | 0.8610 |
| `b01_split_theta` | `nepafull` | 0.6724 | 80 | 0.6901 |
| `b02_split_theta_typepos` | `sotafair` | 0.8623 | 96 | 0.8636 |
| `b02_split_theta_typepos` | `nepafull` | 0.6177 | 90 | 0.5967 |
| `b03_split_viewraster_typepos` | `sotafair` | 0.8701 | 85 | 0.8630 |
| `b03_split_viewraster_typepos` | `nepafull` | 0.6982 | 86 | 0.7272 |
| `b04_split_xanchor_morton_typepos` | `sotafair` | 0.8657 | 74 | 0.8626 |
| `b04_split_xanchor_morton_typepos` | `nepafull` | 0.7520 | 99 | 0.7728 |
| `b05_split_xanchor_fps_typepos` | `sotafair` | 0.8745 | 57 | 0.8581 |
| `b05_split_xanchor_fps_typepos` | `nepafull` | 0.7212 | 90 | 0.7432 |
| `b06_split_dirfps_typepos` | `sotafair` | 0.8687 | 59 | 0.8649 |
| `b06_split_dirfps_typepos` | `nepafull` | 0.7349 | 89 | 0.7510 |
| `b07_event_xanchor_typepos` | `sotafair` | 0.8643 | 95 | 0.8525 |
| `b07_event_xanchor_typepos` | `nepafull` | 0.5845 | 97 | 0.5833 |
| `b08_event_dirfps_typepos` | `sotafair` | 0.8667 | 80 | 0.8568 |
| `b08_event_dirfps_typepos` | `nepafull` | 0.5889 | 89 | 0.5719 |

CPAC/chamfer status in this run set:

- all variants stopped at mesh precheck with `ValueError: mesh_eval requires sequence length 1538, but model max_len=1300`
- therefore no valid CPAC/chamfer summary is available from this bundle.

### 37.7 CPAC/chamfer retry submission (mesh precheck fix) (2026-02-26)

Purpose:

- Re-run only the failed CPAC part (including mesh/chamfer) for the same `a256_queryrethink` checkpoints.
- Keep classification results from §37.6 as-is.

Fix applied for retry:

- Keep:
  - `CPAC_N_CONTEXT=256`
  - `CPAC_N_QUERY=256`
  - `CPAC_MAX_LEN=1300`
- Change:
  - `CPAC_MESH_CHUNK_N_QUERY: 512 -> 256`
- rationale:
  - mesh precheck length uses `1 + 2*(n_context + mesh_chunk_n_query) + eos`
  - old `256+512` required `1538` (`>1300`)
  - new `256+256` requires `1026` (`<=1300`)

Submission:

- helper script: `scripts/eval/submit_a256_queryrethink_cpac_retry_qf.sh`
- source checkpoint set: `a256_queryrethink_20260226_024537`
- retry run set: `a256_queryrethink_cpac_retry_20260226_125022`
- log root: `logs/eval/a256_queryrethink_cpac_retry_20260226_125022`
- result root: `results/a256_queryrethink_cpac_retry_20260226_125022`
- submitted jobs list:
  - `logs/eval/a256_queryrethink_cpac_retry_20260226_125022/submitted_jobs_a256_queryrethink_cpac_retry_20260226_125022.txt`

Submitted 18 CPAC-only jobs:

- `97569.qjcm` `b00_interleave_theta_sotafair_cpacfix`
- `97570.qjcm` `b00_interleave_theta_nepafull_cpacfix`
- `97571.qjcm` `b01_split_theta_sotafair_cpacfix`
- `97572.qjcm` `b01_split_theta_nepafull_cpacfix`
- `97573.qjcm` `b02_split_theta_typepos_sotafair_cpacfix`
- `97574.qjcm` `b02_split_theta_typepos_nepafull_cpacfix`
- `97575.qjcm` `b03_split_viewraster_typepos_sotafair_cpacfix`
- `97576.qjcm` `b03_split_viewraster_typepos_nepafull_cpacfix`
- `97577.qjcm` `b04_split_xanchor_morton_typepos_sotafair_cpacfix`
- `97578.qjcm` `b04_split_xanchor_morton_typepos_nepafull_cpacfix`
- `97579.qjcm` `b05_split_xanchor_fps_typepos_sotafair_cpacfix`
- `97580.qjcm` `b05_split_xanchor_fps_typepos_nepafull_cpacfix`
- `97581.qjcm` `b06_split_dirfps_typepos_sotafair_cpacfix`
- `97582.qjcm` `b06_split_dirfps_typepos_nepafull_cpacfix`
- `97583.qjcm` `b07_event_xanchor_typepos_sotafair_cpacfix`
- `97584.qjcm` `b07_event_xanchor_typepos_nepafull_cpacfix`
- `97585.qjcm` `b08_event_dirfps_typepos_sotafair_cpacfix`
- `97586.qjcm` `b08_event_dirfps_typepos_nepafull_cpacfix`

State at submission check:

- all 18 jobs are `R` (`qstat -x`, immediate post-submit check).

### 37.8 Unreported-status clarification (2026-02-26)

Definition used in this document:

- `unreported` means there is no finalized metric table yet for that checkpoint/runset.

Current unreported items asked in review:

- Pretrain augmentation train/test-domain-gap track (§30):
  - pretrain checkpoints are completed (`96560`, `96561`),
  - but no downstream eval logs/results were found that reference:
    - `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt`
    - `runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt`
  - therefore this track is still unreported.
- Protocol-variant split benchmark (`obj_bg` / `obj_only` / `pb_t50_rs`, §23):
  - submit path is implemented,
  - but no executed runset/result table is recorded yet; still unreported.
- Train-variant thinning / overfit-suppression setting (for example, original-only train):
  - no dedicated implementation+run table is recorded yet; still unreported.
- ECCV-style reporting policy (test-only headline):
  - policy intent is noted, but existing tables still include `best_val`;
  - final protocol tables should be emitted as `test_acc`-first when runs complete.

Note on CPAC/chamfer retry:

- `97569`..`97586` are currently running (`R`) in `a256_queryrethink_cpac_retry_20260226_125022`;
- CPAC/chamfer numbers remain pending until these jobs finish successfully.

## 38. Protocol-Correct Re-evaluation Policy (2026-02-26)

For external/fair comparison tables, ScanObjectNN must use protocol-variant split caches
instead of mixed `main_split` cache.

Canonical variant protocol (one train h5 per variant):

- `obj_bg`:
  - train: `main_split/training_objectdataset.h5`
  - test: `main_split/test_objectdataset.h5`
- `obj_only`:
  - train: `main_split_nobg/training_objectdataset.h5`
  - test: `main_split_nobg/test_objectdataset.h5`
- `pb_t50_rs`:
  - train: `main_split/training_objectdataset_augmentedrot_scale75.h5`
  - test: `main_split/test_objectdataset_augmentedrot_scale75.h5`

Operational rule:

- Any ScanObjectNN result from `data/scanobjectnn_main_split_v2` is treated as
  internal/provisional and not used as final fair benchmark evidence.
- Final benchmark tables should headline `test_acc` only.
- `best_val` / `best_ep` are kept as internal diagnostics, not as headline metrics.

## 39. Re-evaluation Scope (Must-check Ablations)

Not every historical run is re-executed. The following ablations are considered mandatory
for protocol-correct re-evaluation.

### 39.1 Phase-1 (must-run for fair benchmark)

1. Variant cache build:
   - build `data/scanobjectnn_obj_bg_v2`
   - build `data/scanobjectnn_obj_only_v2`
   - build `data/scanobjectnn_pb_t50_rs_v2`
2. 1024 core pretrain comparison:
   - `Run A` vs `Run B` (at minimum; add `C/D` if resources permit)
3. Fine-tune regularization ablation on each variant:
   - `base`, `llrd`, `dp`, `llrd_dp`
4. Pretrain sampling-mode check on each variant:
   - `fps` pretrain checkpoint vs `rfps` pretrain checkpoint

### 39.2 Phase-2 (high-value follow-ups after Phase-1)

1. point-order sensitivity:
   - `point_order_mode=morton` vs `fps` (variant-split protocol)
2. augmentation sensitivity:
   - `scanobjectnn` vs `none` (variant-split protocol)
3. dual-mask ON/OFF:
   - run when comparable 1024 checkpoints are ready for both states.

### 39.3 Reporting policy for this re-evaluation cycle

- Publish one table per variant (`obj_bg`, `obj_only`, `pb_t50_rs`).
- Keep SOTA-fair and NEPA-full as separate tables.
- For each row, record metadata:
  - checkpoint path/job id
  - pretrain `pt_sample_mode_train` / `pt_rfps_m`
  - eval `pt_sample_mode_train` / `pt_sample_mode_eval`
  - ScanObjectNN cache root (must be one of variant caches above)

## 40. Change-Driven Verification Matrix (2026-02-26)

This section narrows what must be re-tested after switching to protocol-variant
reporting as the default.

### 40.1 Should 256 be re-run?

Yes. Re-run `256` as a low-cost gate before expensive `1024` reruns.

Role of `256`:

- validate protocol wiring (`SCAN_CACHE_ROOT` is variant cache, not main_split)
- validate trend direction quickly
- de-risk job scripts and dependency chain

Limitation:

- `256` is not the final headline benchmark; final claims should be from `1024`.

### 40.2 Must-test hypotheses after this policy change

1. Variant split can change ranking:
   - re-check core checkpoints on `obj_bg` / `obj_only` / `pb_t50_rs`.
2. In-domain Scan pretrain contamination risk:
   - compare pretrain corpus:
     - `mesh+udf+scan` (`pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml`)
     - `mesh+udf only` (`pretrain_mixed_shapenet_mesh_udf.yaml`)
3. Sampling robustness:
   - `fps` pretrain vs `rfps` pretrain under the same variant protocol.
4. Fine-tune ablation stability:
   - `base` vs `llrd` vs `dp` vs `llrd_dp` after (1)-(3) are confirmed.

### 40.3 Minimal execution order (recommended)

Phase A (protocol sanity):

1. build three variant caches (`obj_bg`, `obj_only`, `pb_t50_rs`)
2. check `_meta/scanobjectnn_{train,test}_source.txt` for each cache:
   - expected `h5_count=1` per split

Phase B (256 gate):

1. run small comparison for corpus effect:
   - `mesh+udf+scan` vs `mesh+udf only`
2. evaluate per variant, at least with `base`
3. if ranking/trend is unstable, do not scale to 1024 yet

Phase C (1024 core rerun):

1. re-evaluate key checkpoints on all three variants:
   - at minimum `A/B`; include `C/D` if resources allow
2. include `fps` vs `rfps` check for the same checkpoint family

Phase D (secondary ablations):

1. run `llrd/dp/llrd_dp` on top checkpoints from Phase C
2. optional follow-up:
   - point order (`morton` vs `fps`)
   - augmentation preset (`scanobjectnn` vs `none`)

### 40.4 What does not need full rerun immediately

- CPAC/chamfer is not directly affected by ScanObjectNN variant split itself.
- So CPAC/chamfer rerun is required only for newly trained checkpoints used for
  final tables, not for every intermediate Scan-only protocol test.

## 41. Historical Comparison Backfill Policy (2026-02-26)

Clarification for this cycle:

- yes, augmentation comparisons are included.
- For ScanObjectNN, the target is to re-check all previously claimed comparison axes
  under variant-split protocol (`obj_bg` / `obj_only` / `pb_t50_rs`).
- execution can be staged, but omission from final report is not allowed for axes
  that were previously used for conclusions.

### 41.1 Backfill target axes (ScanObjectNN)

1. Pretrain corpus axis:
   - `mesh+udf+scan` vs `mesh+udf only` vs `scan-pointcloud`
2. Pretrain sampling axis:
   - `fps` vs `rfps`
3. Pretrain architecture/tokenization axis:
   - A/B/C/D core settings
   - query serialization/rethink variants (use `256` gate first, then selected `1024`)
4. Fine-tune optimization axis:
   - `base`, `llrd`, `dp`, `llrd_dp`
   - `llrd_mode` (`exp`, `linear`) when relevant to prior claims
5. Fine-tune representation axis:
   - `point_order_mode` (`morton`, `fps`)
   - `cls_pooling` (`mean_q`, `mean_a`) where previously compared
6. Fine-tune regularization axis:
   - label smoothing sweep (`0.0`, `0.1`)
   - `use_fc_norm` and norm no-decay split ablation
7. Fine-tune augmentation axis:
   - `scanobjectnn` vs `none`
   - include legacy preset only when reproducing legacy claims
8. Dual-mask axis:
   - ON/OFF comparison for comparable checkpoint families

### 41.2 Practical execution rule

- If a past section contains a comparative claim on ScanObjectNN, that axis enters
  this backfill scope.
- `256` runs are allowed as fast gate/screening, but final claim updates should be
  confirmed on `1024`.
- final public-facing tables must be variant-split and `test_acc`-headline.

### 41.3 Recommended bundling to control queue size

1. Bundle A (protocol + core):
   - variant cache build
   - A/B core on 3 variants (base only), `256` then `1024`
2. Bundle B (optimization/regularization):
   - `llrd/dp/llrd_dp`, pooling, label smoothing, fc_norm/wd-split
3. Bundle C (representation/augmentation):
   - point order, augmentation preset, dual-mask
4. Bundle D (corpus/sampling/tokenization follow-up):
   - mesh+udf-only comparisons
   - `fps` vs `rfps`
   - selected query-rethink variants promoted from `256` to `1024`
