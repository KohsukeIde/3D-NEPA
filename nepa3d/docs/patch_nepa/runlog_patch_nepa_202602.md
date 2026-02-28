# Patch-NEPA Runlog (2026-02)

Last updated: 2026-02-28

Track note:

- This file is dedicated to Patch-NEPA Stage-2 runs only.
- Historical Query-NEPA/pre-Stage-2 run history stays in:
  - `nepa3d/docs/query_nepa/runlog_202602.md`

## 0. Execution Rule Update (2026-02-28, mandatory)

- Stage-2 pretrain mainline is fixed to 16 GPUs:
  - `4 nodes x 4 GPU/node` (`rt_QF=4`, `NPROC_PER_NODE=4`)
  - launch path:
    - `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - submit path:
    - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
- Invalid pattern (do not report as mainline):
  - single-node launch with `ddp: num_processes=1` while expecting multi-GPU
- Incident note:
  - jobs `99999/100000` were submitted from single-node submit path and ran with `num_processes=1`; stopped and relaunched on true 16-GPU topology (`100010/100011`).

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

- launcher defaults were switched to `pt_sample_mode=rfps_cached` (with `pt_rfps_m=4096`) for subsequent Patch-NEPA runs.
- `pt_rfps_key` is required to resolve to RFPS bank key (`auto` allowed only when key exists).

## 5. RFPS-Default DDP8 Relaunch (2026-02-28)

After default-policy change to `rfps_cached`, relaunch sequence:

- `99634.qjcm` stopped (was still `pt_sample_mode=random`).
- `99641.qjcm` submitted during RFPS-policy transition but wrong launcher path (non-multinode wrapper), then stopped.
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
- `PT_SAMPLE_MODE=rfps_cached`
- `PT_RFPS_KEY=auto` (resolved to existing RFPS bank key)
- global batch `= 16 * 8 * 1 = 128`

## 6. Next Append Policy

For each Stage-2 job, append:

- job id / queue state / exit status
- config key overrides (`N_POINT`, `N_RAY`, `PATCH_EMBED`, grouping, `USE_RAY_PATCH`)
- checkpoint path
- downstream finetune/eval dependency job ids

## 7. Query-NEPA Shared Failure Modes (must be logged)

The following are known failures already seen in Query-NEPA and can reappear in Patch-NEPA.
Each Stage-2 run should explicitly confirm these are clean.

1. Multi-GPU allocation but non-DDP launch
- Symptom: job gets multiple GPUs but training is effectively single-process.
- Patch-NEPA evidence: `99602.qjcm` (stopped; replaced by DDP wrapper path).
- Impact: wrong throughput and unfair comparability.
- Required log check: `num_processes`, `num_machines`, global batch line.

2. Launcher env/workdir propagation mismatch
- Symptom: malformed/empty env vars or wrong root path in wrapper launch.
- Patch-NEPA evidence: `99641.qjcm`/`99642.qjcm` retries before final corrected submit.
- Impact: run either fails fast or runs with unintended settings.
- Required log check: run header (`workdir`, `run_tag`, `mix_config`, key args).

3. Sampling-order key mismatch -> expensive fallback path
- Symptom: requested sampled mode cannot use cached order key and falls back to on-the-fly computation.
- Query-NEPA precedent: repeated FPS fallback warnings and severe slowdown.
- Patch-NEPA policy: use `rfps_cached` with bank key availability prechecked.
- Fallback to on-the-fly RFPS/FPS is not allowed in Stage-2 mainline runs.
- Required log check: no fallback warnings; explicit `pt_sample_mode`, `pt_rfps_key`.

4. Data cardinality mismatch vs baseline protocol
- Symptom: mixed pretrain sees inflated effective samples vs Query baseline.
- Patch-NEPA fix: one-pass mix config (`replacement: false`) for parity.
- Required log check: mix config path and one-pass policy in run header.

5. Silent config drift across retries
- Symptom: relaunch changes recipe unintentionally (scheduler/resume/sample mode).
- Query-NEPA precedent: historically caused invalid cross-run comparisons.
- Patch-NEPA policy: run header must print full training recipe each job.
- Required log check: `lr_scheduler`, `auto_resume`, `resume_optimizer`, sampling mode.

## 8. PatchNEPA ptonly -> PatchCls Finetune (2026-02-28)

Run set:

- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007`
- job ids:
  - `100002.qjcm` (`obj_bg_nepa2d`) -> finished
  - `100003.qjcm` (`obj_only_nepa2d`) -> finished
  - `100004.qjcm` (`pb_t50_rs_nepa2d`) -> finished

Recipe snapshot (from logs):

- `val_split_mode=file`
- eval-time voting enabled:
  - `aug_eval=True`
  - `mc_test=10`
- source ckpt: point-only PatchNEPA one-pass pretrain
  - `runs/patchnepa_pointonly/patchnepa_pointonly_20260228_150907/ckpt_latest.pt`

Current results:

| job | variant | status | metric |
|---|---|---|---:|
| `100002` | `obj_bg` | done | `TEST acc=0.8003` |
| `100003` | `obj_only` | done | `TEST acc=0.7986` |
| `100004` | `pb_t50_rs` | done | `TEST acc=0.7047` |

Source logs:

- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/pb_t50_rs_nepa2d.out`

Scratch comparison note:

- The strict file-split scratch references are:
  - `obj_bg`: `0.7831` (`patchcls_obj_bg_pmalign_splitfile_20260227_223435`)
  - `obj_only`: `0.7849` (`patchcls_objonly_factor4_20260228_070858`, `pcf_b_file`)
- Delta vs those file-split references:
  - `obj_bg`: `+0.0172` (`0.8003 - 0.7831`)
  - `obj_only`: `+0.0137` (`0.7986 - 0.7849`)

Reference logs for file-split scratch values:

- `logs/sanity/patchcls/patchcls_obj_bg_pmalign_splitfile_20260227_223435/obj_bg.out`
- `logs/sanity/patchcls/patchcls_objonly_factor4_20260228_070858/pcf_b_file.out`

## 9. DDP16 RFPS-bank ptonly completion + FT submission (2026-02-28)

Pretrain completion:

- job: `100010.qjcm` (`patchnepa_pt16`) -> `Exit_status=0`
- walltime: `00:35:02`
- run tag: `patchnepa_ptonly_ddp16_rfpsbank_20260228_223103`
- checkpoint:
  - `runs/patchnepa_pointonly/patchnepa_ptonly_ddp16_rfpsbank_20260228_223103/ckpt_latest.pt`

Note:

- Node logs contain a post-train wrapper warning (`pbsdsh ... exit status 127` and trailing `command not found`) after `Done. checkpoints ...`.
- Training itself reached epoch 100 and wrote full checkpoint set; keep this incident logged and audit launcher cleanup path separately.

Fine-tune jobs submitted from `100010` ckpt:

- run set:
  - `logs/sanity/patchcls/patchcls_ft_from_patchnepa_pt16_rfpsbank_20260228_231732`
- jobs:
  - `100028.qjcm` (`obj_bg_nepa2d`) running
  - `100029.qjcm` (`obj_only_nepa2d`) running
  - `100030.qjcm` (`pb_t50_rs_nepa2d`) running
- recipe:
  - `val_split_mode=file`, `aug_eval=1`, `mc_test=10`, `backbone_mode=nepa2d`

Update (partial completion):

- `100028.qjcm` -> `Exit_status=0`, `TEST acc=0.7401` (`obj_bg`)
- `100029.qjcm` -> `Exit_status=0`, `TEST acc=0.7849` (`obj_only`)
- `100030.qjcm` (`pb_t50_rs`) remains running at this snapshot.

## 10. Scheduler-policy default lock + controlled LR comparison (2026-02-28)

Policy update applied (for new submissions only):

- Stage-2 submit/launcher defaults are now fixed to:
  - `lr_scheduler=cosine`
  - `warmup_ratio=0.025`
- updated scripts:
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa.sh`

Controlled comparison run submitted (same recipe, scheduler change only):

- baseline reference completion (same recipe, fixed-LR run):
  - `100027.qjcm` (`patchnepa_ptE16`) finished, `Exit_status=0`
  - run set:
    - `patchnepa_pointonly_ddp16_encdec_split_20260228_230101`
  - key startup setting:
    - `lr_scheduler=none`, `warmup_epochs=0`

- job: `100042.qjcm` (`patchnepa_ptE16c`) running
- run set:
  - `patchnepa_pointonly_ddp16_encdec_split_cosine_20260301_000000`
- key config at startup log:
  - `lr_scheduler=cosine`
  - `warmup_epochs=2.5`
  - `warmup_ratio=0.025`
  - `batch_per_proc=8`, `num_processes=16`, `global_batch=128`

Reference log:

- `logs/patch_nepa_pretrain/patchnepa_pointonly_ddp16_encdec_split_cosine_20260301_000000/patchnepa_pointonly_ddp16_encdec_split_cosine_20260301_000000.mr0.log`
