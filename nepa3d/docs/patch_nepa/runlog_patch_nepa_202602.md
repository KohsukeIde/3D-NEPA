# Patch-NEPA Runlog (2026-02)

Last updated: 2026-02-28

Track note:

- This file is dedicated to Patch-NEPA Stage-2 runs only.
- Historical Query-NEPA/pre-Stage-2 run history stays in:
  - `nepa3d/docs/query_nepa/runlog_202602.md`

## 1. Bootstrap (2026-02-28)

Stage-2 code integration completed:

- raw-return path in dataset/mixed pretrain:
  - `nepa3d/data/dataset.py`
  - `nepa3d/data/mixed_pretrain.py`
- new patch pretrain model/entry:
  - `nepa3d/models/patch_nepa.py`
  - `nepa3d/train/pretrain_patch_nepa.py`
- new launch scripts:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`

Notes:

- Stage bundle had incompatibilities; merged version is adapted to current repo APIs.
- `dataset.py` stage bug (`label` undefined in raw return path) was fixed during integration.

## 2. Point-Only Pretrain Submission

Submitted:

- job: `99591.qjcm`
- queue name: `patchnepa_ptonly`
- state at submission check: `R`
- run_set: `patchnepa_pointonly_20260228_143750`

Paths:

- save dir: `runs/patchnepa_pointonly/patchnepa_pointonly_20260228_143750`
- log root: `logs/patch_nepa_pretrain/patchnepa_pointonly_20260228_143750`
- main log:
  - `logs/patch_nepa_pretrain/patchnepa_pointonly_20260228_143750/run_pointonly_patchnepa_pointonly_20260228_143750.log`

## 3. DDP Migration + Relaunch (2026-02-28)

Findings:

- `99602.qjcm` used 4-GPU allocation but non-DDP launch path (single process), so effective multi-GPU utilization was not guaranteed.
- launcher gap was fixed by:
  - enabling `accelerate launch` path in:
    - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - adding multi-node pbsdsh wrapper:
    - `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`

Actions:

- `99602.qjcm` stopped.
- new 8-GPU DDP job submitted:
  - `99613.qjcm`
  - resources: 2 nodes x 4 GPUs = 8 GPUs total
  - run_set: `patchnepa_pointonly_ddp8_20260228_152058`

Logs:

- pbs:
  - `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_20260228_152058.pbs.log`
- per-node:
  - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99613.qjcm_run_pointonly_patchnepa_pointonly_ddp8_20260228_152058/logs/qh454.patchnepa.log`
  - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99613.qjcm_run_pointonly_patchnepa_pointonly_ddp8_20260228_152058/logs/qh455.patchnepa.log`

Config notes in node logs:

- `batch=16` is per-process.
- global batch is logged as `128` (`16 x 8 x 1`).
- patch config: `fps_knn`, `group_size=32`, `num_groups=64`.

## 4. Parity Fix Relaunch (2026-02-28)

Reason:

- `99613.qjcm` was stopped after launcher parity updates (resume/scheduler/aug/defaults cleanup).

Submitted:

- `99634.qjcm` (`patchnepa_ptonly_ddp8`)
- resources: 2 nodes x 4 GPUs = 8 GPUs total
- run_set: `patchnepa_pointonly_ddp8_fix_20260228_153151`

Logs:

- pbs:
  - `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_fix_20260228_153151.pbs.log`
- per-node:
  - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99634.qjcm_run_pointonly_patchnepa_pointonly_ddp8_fix_20260228_153151/logs/qh138.patchnepa.log`
  - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99634.qjcm_run_pointonly_patchnepa_pointonly_ddp8_fix_20260228_153151/logs/qh143.patchnepa.log`

Config notes in node logs:

- `num_processes=8`, `num_machines=2`.
- `batch=16` is per-process, global batch = `128`.
- point-only baseline:
  - `patch_embed=fps_knn`, `group_size=32`, `num_groups=64`
  - `pt_sample_mode=random` (this run), `pt_xyz_key=pc_xyz`, `ablate_point_dist=1`
  - `lr_scheduler=none`, `auto_resume=1`.

Post-run policy update:

- launcher defaults were switched to `pt_sample_mode=rfps` (with `pt_rfps_m=4096`) for subsequent Patch-NEPA runs.

## 5. RFPS-Default DDP8 Relaunch (2026-02-28)

After default-policy change to `rfps`, relaunch sequence:

- `99634.qjcm` stopped (was still `pt_sample_mode=random`).
- `99641.qjcm` submitted with `rfps` but wrong launcher path (non-multinode wrapper), then stopped.
- `99642.qjcm` multinode wrapper submit with malformed empty env vars, then stopped.
- final corrected run:
  - `99643.qjcm` (`patchnepa_ptonly_ddp8`)
  - run_set: `patchnepa_pointonly_ddp8_rfpsfix_20260228_154649`
  - pbs log:
    - `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_rfpsfix_20260228_154649.pbs.log`
  - node logs:
    - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh138.patchnepa.log`
    - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh142.patchnepa.log`

Topology/settings check in logs:

- `NNODES=2`, `NPROC_PER_NODE=4`, `NUM_PROCESSES=8`
- `PT_SAMPLE_MODE=rfps`
- global batch `= 16 * 8 * 1 = 128`

## 6. Next Append Policy

For each Stage-2 job, append:

- job id / queue state / exit status
- config key overrides (`N_POINT`, `N_RAY`, `PATCH_EMBED`, grouping, `USE_RAY_PATCH`)
- checkpoint path
- downstream finetune/eval dependency job ids
