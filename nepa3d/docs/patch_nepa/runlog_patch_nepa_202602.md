# Patch-NEPA Runlog (2026-02)

Last updated: 2026-03-24

Track note:

- This file is dedicated to Patch-NEPA Stage-2 runs only.
- Historical Query-NEPA/pre-Stage-2 run history stays in:
  - `nepa3d/docs/query_nepa/runlog_202602.md`

Restart policy note (2026-03-03):

- Data semantics restart memo is tracked in:
  - `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- New mainline comparisons should be recorded as data-v2 runs only.

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
- `100030.qjcm` (`pb_t50_rs`) log already shows `TEST acc=0.7078` (queue state was still `R` at capture time; treat as provisional until `F`).

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

Scheduler bug found from `100042` log inspection:

- symptom:
  - LR showed non-monotonic oscillation and rapid phase drift (for example
    `1.20e-04 -> 2.86e-04 -> 2.37e-04 -> ... -> 2.24e-06 -> ... -> 2.98e-04`)
  - loss rose from `~1e-2` to `~1e-1` range instead of converging.
- root cause:
  - scheduler was passed to `accelerator.prepare(...)`, which allows step-level
    scheduler advancement with optimizer updates under Accelerate.
  - code also called `scheduler.step()` at epoch end, causing schedule mismatch.
- fix:
  - `nepa3d/train/pretrain_patch_nepa.py`
    - do **not** pass scheduler into `accelerator.prepare`
    - keep explicit epoch-end `scheduler.step()` only

## 11. Stage-2 Transfer Path Correction (2026-03-01)

Decision:

- Stage-2 pretrain->finetune path is fixed to `model_source=patchnepa` (direct path).
- Adapter conversion into `PatchTransformerClassifier` is excluded from Stage-2 mainline reporting.

Reason:

- Adapter path discards pretrain-side components (`answer_embed`, `type_emb`, `center_mlp`, `sep/eos`, `pred_head`)
  and changes the token construction path.
- This breaks strict continuity with Query-NEPA-style "same backbone line for pretrain->finetune".

Operational rule from this point:

- All new Stage-2 transfer runs must use:
  - pretrain: `nepa3d/models/patch_nepa.py`
  - finetune: `nepa3d/models/patch_nepa_classifier.py`
  - launch arg: `--model_source patchnepa`

Note:

- `PatchTransformerClassifier` remains valid for Stage-1 scratch baseline only.
    - clamp warmup/cosine scale (`warmup<=1.0`, `t in [0,1]`).

Status after fix:

- `100042` is treated as invalid for LR comparison.
- replacement run submitted:
  - `100045.qjcm` (`patchnepa_ptE16cf`)
  - run set:
    - `patchnepa_pointonly_ddp16_encdec_split_cosine_fixsched_20260301_001200`

## 11. Run1 vs Run2 comparability audit (2026-03-01)

Question:

- why `Run1 (ptonly_onepass)` improved FT while `Run2 (pt16_rfpsbank)` did not?

Key finding:

- this is **not** a clean same-recipe A/B. Pretrain settings differ materially.

Run-1 pretrain (`99710.qjcm`, 8 GPU):

- ckpt:
  - `runs/patchnepa_pointonly/run_pointonly_patchnepa_onepass_rfpsbank_20260228_180841/ckpt_latest.pt`
- effective pretrain settings (from `qstat -xf` / log header):
  - `PT_XYZ_KEY=pc_xyz`
  - `PT_DIST_KEY=pt_dist_pool`
  - `ABLATE_POINT_DIST=1`
  - `PT_SAMPLE_MODE=rfps_cached`
  - `BATCH=16` per proc, `num_processes=8` (`global_batch=128`)

Run-2 pretrain (`100010.qjcm`, 16 GPU):

- ckpt:
  - `runs/patchnepa_pointonly/patchnepa_ptonly_ddp16_rfpsbank_20260228_223103/ckpt_latest.pt`
- effective pretrain settings:
  - `PT_XYZ_KEY=pt_xyz_pool`
  - `PT_DIST_KEY=pt_dist_pool`
  - `ABLATE_POINT_DIST=0`
  - `PT_SAMPLE_MODE=rfps_cached`
  - `BATCH=8` per proc, `num_processes=16` (`global_batch=128`)

Finetune comparability check:

- FT args between
  - `runs/sanity/patchcls/patchcls_obj_bg_nepa2d_patchcls_ft_from_patchnepa_ptonly_onepass_20260228_222007/args.json`
  - `runs/sanity/patchcls/patchcls_obj_bg_nepa2d_patchcls_ft_from_patchnepa_pt16_rfpsbank_20260228_231732/args.json`
- differ effectively only in `ckpt` path (same `val_split_mode=file`, same TTA/vote settings).

Interpretation:

- observed FT gap is attributable to **pretrain ckpt content/config drift**, not FT recipe drift.
- do not conclude "DDP16 is worse" from this pair.

## 12. `100045` early-phase LR/loss check (requested follow-up)

Run:

- `100045.qjcm`
- log:
  - `logs/patch_nepa_pretrain/patchnepa_pointonly_ddp16_encdec_split_cosine_fixsched_20260301_001200/patchnepa_pointonly_ddp16_encdec_split_cosine_fixsched_20260301_001200.mr0.log`

Epoch 1-5 summary (rank0 step logs):

- `epoch1`: `lr=2.40e-04`, `loss_avg=0.013500`, `loss_last=0.020427`
- `epoch2`: `lr=3.00e-04`, `loss_avg=0.084359`, `loss_last=0.144582`
- `epoch3`: `lr=3.00e-04`, `loss_avg=0.162487`, `loss_last=0.188165`
- `epoch4`: `lr=3.00e-04`, `loss_avg=0.212389`, `loss_last=0.228000`
- `epoch5`: `lr=3.00e-04`, `loss_avg=0.217572`, `loss_last=0.215163`

Status:

- scheduler bug itself (oscillatory LR from double stepping) is fixed.
- early loss trend remains unstable/high, so efficacy judgment must wait for full run + FT.

## 13. Meaningless-run cleanup + resubmit (`100047`, 2026-03-01)

User request:

- stop currently running non-comparable jobs and relaunch with comparable settings.

Queue status at cleanup:

- no stale Stage-2 run remained in `qstat`; active run is only:
  - `100047.qjcm` (`patchnepa_ptE16r1`)

Relaunch (comparison-oriented point-only run):

- job: `100047.qjcm`
- run set:
  - `patchnepa_pointonly_ddp16_cmp_run1cfg_20260301_001213`
- logs:
  - `logs/patch_nepa_pretrain/patchnepa_pointonly_ddp16_cmp_run1cfg_20260301_001213/run_pointonly_patchnepa_pointonly_ddp16_cmp_run1cfg_20260301_001213.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_pointonly_ddp16_cmp_run1cfg_20260301_001213/run_pointonly_patchnepa_pointonly_ddp16_cmp_run1cfg_20260301_001213.pbs.log`

Confirmed effective settings from startup log:

- topology: `16 GPU` (`4 nodes x 4`)
- patch: `fps_knn`, `num_groups=64`, `group_size=32`
- sample: `pt_sample_mode=rfps_cached` (`pt_rfps_key=auto`)
- keys: `pt_xyz_key=pc_xyz`, `pt_dist_key=pt_dist_pool`, `ablate_point_dist=1`
- sequence: `qa_tokens=1`, `qa_layout=split_sep`, `encdec_arch=1`
- global batch: `128` (`8 x 16 x 1`)

Stage-2 sanity gate (step 0 token count) passed:

- `Q_POINT=64`, `A_POINT=64`, `SEP=1`, `BOS=1`, `EOS=1`, `Q_RAY=0`, `A_RAY=0`

Early optimization signal (mr0 log):

- `[epoch 000 step 000000] loss=1.00048, lr=1.20e-04`
- `[epoch 000 step 000050] loss=2.35e-03`
- `[epoch 002 step 000800] loss=2.64e-03`

Interpretation:

- run is healthy (no zero-loss collapse, no token-layout mismatch), and Stage-2 pretrain is progressing normally.

## 14. Ray FT protocol fix (query-only ray inputs) + resubmit (2026-03-01)

Issue:

- previous Ray FT launch (`100069/100070/100071`) used `USE_RAY_PATCH=1`, but
  classifier ray branch still consumed `ray_t/ray_hit` (answer-side fields).
- this violates strict finetune protocol where ray branch must use query-only
  information (`ray_o/ray_d`) and not answer-like targets.

Code fix applied:

- `nepa3d/models/patch_classifier.py`
  - ray encoder input changed to query-only:
    - `[x_proxy-center (3), ray_d (3)]` (6 dims)
  - proxy anchor now uses `ray_o + ray_miss_t * ray_d` only
  - removed `ray_t/ray_hit` usage from ray-patch pooling features
  - `use_ray_patch=True` now requires only `ray_o/ray_d`
- `nepa3d/data/cls_patch_dataset.py`
  - ray-required keys relaxed to `ray_o_pool/ray_d_pool`
  - `ray_t/ray_hit` are optional outputs when present
- `nepa3d/train/finetune_patch_cls.py`
  - train/eval checks updated to require only `ray_o/ray_d`
  - `ray_t/ray_hit` optional pass-through only
  - startup log now prints `ray_query_only=1`

Job handling:

- invalid Ray FT jobs were cancelled:
  - `100069.qjcm`, `100070.qjcm`, `100071.qjcm`
- query-only Ray FT relaunched:
  - `100073.qjcm` (`obj_bg`, `nepa2d`)
  - `100074.qjcm` (`obj_only`, `nepa2d`)
  - `100075.qjcm` (`pb_t50_rs`, `nepa2d`)
- run set:
  - `patchnepa_rayqa_ft_from100011_queryonly_20260301_002508`
- logs:
  - `logs/sanity/patchcls/patchnepa_rayqa_ft_from100011_queryonly_20260301_002508`

Relaunch recipe snapshot:

- ckpt:
  - `runs/patchnepa_pointonly/patchnepa_rayqa_ddp16_rfpsbank_20260228_223103/ckpt_latest.pt`
- `USE_RAY_PATCH=1`, `N_RAY=1024`
- `RAY_SAMPLE_MODE_TRAIN=random`, `RAY_SAMPLE_MODE_EVAL=first`
- `VAL_SPLIT_MODE=file`
- `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`

Latest status update (2026-03-01 02:50 JST snapshot):

- This run set is **Ray-enabled FT** (`USE_RAY_PATCH=1`, query-only ray inputs).
- completed:
  - `100073.qjcm` (`obj_bg`): `TEST acc=0.7281`
  - `100074.qjcm` (`obj_only`): `TEST acc=0.7762`
- running:
  - `100075.qjcm` (`pb_t50_rs`): epoch progress continues (`ep 128/300` at snapshot), no `TEST acc` yet.
- log files:
  - `logs/sanity/patchcls/patchnepa_rayqa_ft_from100011_queryonly_20260301_002508/obj_bg_nepa2d.out`
  - `logs/sanity/patchcls/patchnepa_rayqa_ft_from100011_queryonly_20260301_002508/obj_only_nepa2d.out`
  - `logs/sanity/patchcls/patchnepa_rayqa_ft_from100011_queryonly_20260301_002508/pb_t50_rs_nepa2d.out`

## 15. Clarification: `100073/100074` vs split-x2 + new submissions (2026-03-01)

Correction (important):

- `100073/100074` are indeed **dual-mask Ray runs**.
- They correspond to the **encdec=0 + dual_mask** side of split-x2.
- They are **not** the full split-x2 comparison result by themselves, because the paired
  `encdec_arch=1` run was not completed as a matched A/B pair when those FT jobs were created.

Config snapshot of `100011` source pretrain used by `100073/100074`:

- pretrain run tag:
  - `patchnepa_rayqa_ddp16_rfpsbank_20260228_223103`
- key settings:
  - `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=0`
  - `dual_mask=(0.5, 0.1, w=32, type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=auto`
  - `lr_scheduler=none` (fixed-LR recipe)

New split-x2 resubmission (matched pair) submitted:

- pretrain:
  - `100118.qjcm` (`patchnepa_ryE16e`)
    - `encdec_arch=1`, `split_sep`, `dual_mask off`
    - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
  - `100119.qjcm` (`patchnepa_ryE16d`)
    - `encdec_arch=0`, `split_sep`, `dual_mask on`
    - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
  - common:
    - `use_ray_patch=1`, `n_ray=1024`, `fps_knn 64x32`, `16 GPU`
    - `lr_scheduler=cosine`, `warmup_ratio=0.025`

- dependent FT jobs (auto-start after pretrain success):
  - from `100118`: `100120` (`obj_bg`), `100121` (`obj_only`), `100122` (`pb_t50_rs`)
  - from `100119`: `100123` (`obj_bg`), `100124` (`obj_only`), `100125` (`pb_t50_rs`)
  - all are currently `H` (dependency hold) until corresponding pretrain completes.

## 16. Mainline policy switch: Point-only -> Ray-default (2026-03-01)

Decision:

- from this point onward, Stage-2 mainline no longer uses point-only pretrain.
- Ray-enabled pretrain is the mandatory default.

Default contract:

- `N_RAY=1024`
- `USE_RAY_PATCH=1`
- `MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_onepass.yaml`

Operational changes applied:

- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - defaults switched to Ray-enabled run naming/paths and config
  - added strict guard (`STAGE2_REQUIRE_RAY=1`) to reject `USE_RAY_PATCH!=1` or `N_RAY<=0`
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - defaults switched to Ray-enabled config (`N_RAY=1024`, `USE_RAY_PATCH=1`, mesh+UDF mix)
  - added same strict guard
- `scripts/pretrain/nepa3d_pretrain_patch_nepa.sh` (single-node helper)
  - defaults switched to Ray-enabled
  - added same strict guard

Note:

- point-only runs are now ablation/debug only and must be explicitly labeled as non-mainline.

## 17. Direct PatchNEPA-FT rerun matrix (2026-03-01)

Reason:

- Current transfer path was corrected to direct PatchNEPA finetune (`--model_source patchnepa`).
- Old adapter-path runs are excluded from Stage-2 mainline transfer claims.

Submitted matrix (pretrain + dependent FT):

- total jobs: `8`
  - pretrain: `2`
  - finetune: `6` (`obj_bg`, `obj_only`, `pb_t50_rs` for each pretrain)

Pretrain jobs:

1. Ray mainline (split + dual-mask):
- `100146.qjcm` (`patchnepa_rayDF`)
- `RUN_SET=patchnepa_ray_directft_20260301_013908`
- config:
  - `MIX_CONFIG=pretrain_mixed_shapenet_mesh_udf_onepass.yaml`
  - `USE_RAY_PATCH=1`, `N_RAY=1024`
  - `QA_TOKENS=1`, `QA_LAYOUT=split_sep`, `ENCDEC_ARCH=0`
  - `DUAL_MASK=(0.5,0.1,w=32,type_aware=1,warmup=0.05)`
  - `PATCH_EMBED=fps_knn`, `64x32`
  - `PT_SAMPLE_MODE=rfps_cached`, `PT_RFPS_KEY=pt_rfps_order_bank`
  - `BATCH(per-proc)=8`, `num_processes=16`, global batch `128`

2. point-only control (same recipe except ray off):
- initial submit `100147.qjcm` failed immediately by guard propagation gap
  (`STAGE2_REQUIRE_RAY` not forwarded to child launch env; node log showed
  `ERROR: Stage-2 mainline requires USE_RAY_PATCH=1 (got 0)`).
- fix applied:
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - now forwards `STAGE2_REQUIRE_RAY` through `qsub -v`.
- rerun:
  - `100154.qjcm` (`patchnepa_ptDF2`)
  - `RUN_SET=patchnepa_ptonly_directft_fix_20260301_014049`
  - config:
    - `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
    - `USE_RAY_PATCH=0`, `N_RAY=0`, `STAGE2_REQUIRE_RAY=0`
    - other settings matched to `100146` (`split_sep`, dual-mask, rfps_cached, 64x32, global batch 128)

Dependent finetune jobs (all use direct path):

- required setting: `MODEL_SOURCE=patchnepa`
- strict eval policy:
  - `VAL_SPLIT_MODE=file`
  - `AUG_EVAL=1`
  - `MC_EVAL_K_TEST=10`

From Ray pretrain `100146`:
- `100148.qjcm` (`obj_bg`)
- `100150.qjcm` (`obj_only`)
- `100152.qjcm` (`pb_t50_rs`)
- run set:
  - `patchnepaFT_from_ray_20260301_013921`

From point-only pretrain `100154`:
- `100155.qjcm` (`obj_bg`)
- `100156.qjcm` (`obj_only`)
- `100157.qjcm` (`pb_t50_rs`)
- run set:
  - `patchnepaFT_from_ptonly_fix_20260301_014058`

Operational note:

- a transient first point-only FT chain (`100149/100151/100153`) was created from failed `100147`; treated invalid and superseded by `100155/100156/100157`.

## 18. Ray split/interleave status + PatchNEPA-named FT launcher (2026-03-01)

Status at this point:

- Ray `split_sep` run: active
  - pretrain: `100146.qjcm`
  - dependent FT: `100148/100150/100152` (`afterok:100146`)
- Ray `interleave` run: newly submitted
  - pretrain: `100158.qjcm` (`patchnepa_rayIL`)
  - key delta vs `100146`:
    - `QA_LAYOUT=interleave`
    - `QA_SEP_TOKEN=0`
    - same ray/sampling/dual-mask/global-batch recipe otherwise
  - dependent FT: `100159/100160/100161` (`afterok:100158`)

Launcher naming cleanup:

- Added PatchNEPA-named FT scripts (to avoid Stage-2 operation under `patchcls` naming):
  - `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
  - `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
- Runtime model path is unchanged and explicit:
  - `MODEL_SOURCE=patchnepa` (direct PatchNEPA finetune path).

Additional causal probe submission:

- `100164.qjcm` (`pn_obj_bg`) submitted as a single-run A/B check
  against the active Ray split run (`afterok:100146`):
  - `MODEL_SOURCE=patchnepa`
  - `VARIANT=obj_bg`
  - `IS_CAUSAL=1` (all other strict-eval settings unchanged: `file + TTA10`)

## 19. Mode-wise Results Snapshot (2026-03-01, latest)

This snapshot summarizes what has actually produced `TEST acc` so far, grouped
by transfer/eval mode.

### 19.1 Ray pretrain (`100146`) -> direct PatchNEPA FT (`MODEL_SOURCE=patchnepa`)

Source pretrain:

- job: `100146.qjcm` (`patchnepa_rayDF`)
- status: `Exit_status=0`
- mode:
  - `USE_RAY_PATCH=1`, `N_RAY=1024`
  - `qa_tokens=1`, `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
  - `16 GPU`, global batch `128`

Dependent FT (strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`):

| job | variant | mode | status | test_acc |
|---|---|---|---|---:|
| `100148` | `obj_bg` | direct PatchNEPA FT (`is_causal=0`) | done | `0.7797` |
| `100150` | `obj_only` | direct PatchNEPA FT (`is_causal=0`) | done | `0.7952` |
| `100152` | `pb_t50_rs` | direct PatchNEPA FT (`is_causal=0`) | running | pending |

Logs:

- `logs/sanity/patchcls/patchnepaFT_from_ray_20260301_013921/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchnepaFT_from_ray_20260301_013921/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchnepaFT_from_ray_20260301_013921/pb_t50_rs_nepa2d.out`

### 19.2 Point-only control pretrain (`100154`) -> direct PatchNEPA FT

Dependent FT (strict eval: `file + TTA10`):

| job | variant | mode | status | test_acc |
|---|---|---|---|---:|
| `100155` | `obj_bg` | direct PatchNEPA FT (`is_causal=0`) | done | `0.7590` |
| `100156` | `obj_only` | direct PatchNEPA FT (`is_causal=0`) | done | `0.7797` |
| `100157` | `pb_t50_rs` | direct PatchNEPA FT (`is_causal=0`) | done | `0.7380` |

Logs:

- `logs/sanity/patchcls/patchnepaFT_from_ptonly_fix_20260301_014058/obj_bg_nepa2d.out`
- `logs/sanity/patchcls/patchnepaFT_from_ptonly_fix_20260301_014058/obj_only_nepa2d.out`
- `logs/sanity/patchcls/patchnepaFT_from_ptonly_fix_20260301_014058/pb_t50_rs_nepa2d.out`

### 19.3 Causal FT probe (same Ray pretrain, `is_causal=1`)

| job | variant | mode | status | test_acc |
|---|---|---|---|---:|
| `100164` | `obj_bg` | direct PatchNEPA FT (`is_causal=1`) | done | `0.7487` |

Log:

- `logs/sanity/patchnepa_ft/patchnepaFT_from_ray_causal_probe_20260301_020406/obj_bg.out`

### 19.4 Refactor reproducibility run (post class-split refactor)

| job | variant | mode | status | test_acc |
|---|---|---|---|---:|
| `100171` | `obj_only` | direct PatchNEPA FT (`is_causal=0`) | done | `0.8072` |

Log:

- `logs/sanity/patchnepa_ft/patchnepa_refactor_repro1_20260301_021900/obj_only.out`

### 19.5 Historical/aborted chains (for traceability)

Ray pretrain split-x2 candidates:

- `100118` (`encdec1 split_sep`) and `100119` (`dualmask split_sep`) ended with
  `Exit_status=265` (`terminated by qch10156fh@qes03`) before completion.
  - both are partial runs only (no final checkpoint claim for matched A/B).

Dependent FT jobs:

- `100120`~`100125` were submitted with `depend=afterok:100118/100119` and are
  finalized in hold/substate path without execution (no `.out/.err` payload).

Older pb_t50_rs FT jobs:

- `100075`, `100091` ended with `Exit_status=265` (terminated); no final
  `TEST acc` recorded for those runs.

## 20. Direct-FT diagnosis checklist (2026-03-01)

Purpose:

- isolate why direct PatchNEPA FT results vary (`100150` vs `100171`) even when
  recipe/ckpt are nominally identical.

### 20.1 Checkpoint adaptation/loading integrity (must pass first)

What to inspect:

- `nepa3d/train/finetune_patch_cls.py`
  - `_adapt_patchnepa_pretrain_to_patchnepa_classifier`
- FT logs:
  - `logs/sanity/patchcls/patchnepaFT_from_ray_20260301_013921/obj_only_nepa2d.out`
  - `logs/sanity/patchnepa_ft/patchnepa_refactor_repro1_20260301_021900/obj_only.out`

Pass criteria:

- `[ckpt-adapt] ... direct=0`
- `[ckpt-adapt] ... mapped=src`
- `Loaded ckpt: unexpected=0`
- `missing` is classifier-only head parameters

Current status:

- PASS on both `100150` and `100171`:
  - ray run: `mapped=247 direct=0 load=247 src=247 dst=263`, `missing=14 unexpected=0`
  - repro run: same values.

Interpretation:

- current evidence does **not** support “broken key mapping / wrong tensor load”
  as root cause.

### 20.2 Finetune best-checkpoint selection path

What to inspect:

- `nepa3d/train/finetune_patch_cls.py`
  - best checkpoint update condition (`val_acc` based save)
  - final test evaluation (ensures best checkpoint is loaded for test)
- FT logs:
  - grep for `saved best ->` progression and final `TEST acc=...`

Pass criteria:

- monotonically updated best marker by `val_acc`
- final test explicitly tied to the best checkpoint (not last epoch by accident)

Reason:

- `100150` and `100171` reached different trajectories while sharing almost
  identical args/ckpt; this is the most likely path for outcome divergence.

### 20.3 Eval variance under `file + TTA10`

What to inspect:

- `nepa3d/train/finetune_patch_cls.py`
  - `evaluate_local` and vote aggregation (`mc_test`, `aug_eval`)
  - seed handling (`_set_seed`, dataloader worker behavior)
- job args:
  - ensure `val_split_mode=file`, `aug_eval=1`, `mc_test=10`, `seed=0`
- reproduce:
  - rerun same ckpt/args at least 2 times (`obj_only`) and compare spread.

Pass criteria:

- repeated runs stay within expected noise band (target: <= ~1pt)
- if spread is larger, treat direct-FT delta claims as unstable until fixed.

### 20.4 Pretrain quality delta (ray/point-only, split/interleave, mask mode)

What to inspect:

- pretrain logs + resolved config header from:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - `nepa3d/train/pretrain_patch_nepa.py`
- required fixed fields:
  - `MIX_CONFIG` (one-pass parity)
  - `PT_SAMPLE_MODE=rfps_cached`, `PT_RFPS_KEY=pt_rfps_order_bank`
  - `qa_layout`, `encdec_arch`, `dual_mask*`
  - topology/global batch (`16 GPU`, global batch `128`)

Pass criteria:

- no recipe drift across compared runs
- no fallback sampling warnings
- token sanity print (step 0) matches intended layout.

Operational note:

- Until 20.2 and 20.3 are closed, prioritize paired reruns over single-shot
  conclusions when judging pretrain efficacy.

## 21. Split-x2 completion rerun (encdec1 vs dualmask, 2026-03-01)

User request:

- complete matched A/B comparison for Ray split setting:
  - A: `encdec_arch=1` (Q-bidir/A-causal path)
  - B: `encdec_arch=0 + dual_mask on`
- and run downstream FT for both branches.

Submitted pretrain jobs (16 GPU, 4x4):

- `100180.qjcm` (`patchnepa_ryE16e2`)
  - run set: `patchnepa_ray_splitx2_encdec1_20260301_035212`
  - key config:
    - `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=1`
    - `dual_mask=(0.0,0.0,w=32,type_aware=0)`
    - `use_ray_patch=1`, `n_ray=1024`
    - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
    - `pt_xyz_key=pt_xyz_pool`, `pt_dist_key=pt_dist_pool`, `ablate_point_dist=0`
    - global batch `128` (`batch=8`, `num_processes=16`)

- `100181.qjcm` (`patchnepa_ryE16d2`)
  - run set: `patchnepa_ray_splitx2_dualmask_20260301_035220`
  - key config:
    - `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=0`
    - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
    - all other knobs matched to `100180`.

Startup sanity checks (rank0 logs):

- both jobs show expected token layout at step0 (`Q_POINT=64`, `A_POINT=64`, `SEP=1`, Ray tokens present)
- both jobs started normal step logs (`epoch0 step0/50...`).

Dependent FT jobs submitted (hold until pretrain `afterok`):

- from `100180`:
  - `100182.qjcm` (`obj_bg`)
  - `100184.qjcm` (`obj_only`)
  - `100186.qjcm` (`pb_t50_rs`)
  - run set: `patchnepaFT_from_splitx2_encdec1_20260301_035230`

- from `100181`:
  - `100183.qjcm` (`obj_bg`)
  - `100185.qjcm` (`obj_only`)
  - `100187.qjcm` (`pb_t50_rs`)
  - run set: `patchnepaFT_from_splitx2_dualmask_20260301_035230`

FT policy (both branches):

- `MODEL_SOURCE=patchnepa`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
- `USE_RAY_PATCH=1`, `N_RAY=1024`.

## 20. FT mode split requeue (`baseline` vs `q_only`) + `mean_q` add-on (2026-03-01)

User-requested action:

- cancelled held FT jobs from previous chain:
  - `100182`, `100183`, `100184`, `100185`, `100186`, `100187`
- requeued with explicit finetune sequence mode separation:
  - `baseline = patchnepa_ft_mode=qa_zeroa` (keep QA layout with zero-A inputs)
  - `q_only = patchnepa_ft_mode=q_only` (remove A tokens at finetune)

Code path updates applied before resubmit:

- `nepa3d/models/patch_nepa.py`
  - `PatchTransformerNepaClassifier` now supports `ft_sequence_mode` (`qa_zeroa|q_only`)
  - added q-only sequence builder
  - added pooling alias `mean_q`
- `nepa3d/train/finetune_patch_cls.py`
  - new CLI arg `--patchnepa_ft_mode`
  - `--pooling` accepts `mean_q`
- launcher env wiring:
  - `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`

Requeued FT jobs (all strict eval: `val_split_mode=file`, `aug_eval=1`, `mc_test=10`):

From pretrain `100180` (`encdec1 split_sep`):

- baseline (`qa_zeroa`, `pooling=cls_max`):
  - `100188` (`obj_bg`), `100189` (`obj_only`), `100190` (`pb_t50_rs`)
- q_only (`q_only`, `pooling=cls_max`):
  - `100191` (`obj_bg`), `100192` (`obj_only`), `100193` (`pb_t50_rs`)

From pretrain `100181` (`dualmask split_sep`):

- baseline (`qa_zeroa`, `pooling=cls_max`):
  - `100194` (`obj_bg`), `100195` (`obj_only`), `100196` (`pb_t50_rs`)
- q_only (`q_only`, `pooling=cls_max`):
  - `100197` (`obj_bg`), `100198` (`obj_only`), `100199` (`pb_t50_rs`)
- q_only + mean pooling probe (`q_only`, `pooling=mean_q`):
  - `100200` (`obj_bg`), `100201` (`obj_only`), `100202` (`pb_t50_rs`)

Submission logs:

- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_baseline_20260301_040736`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_qonly_20260301_040738`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_20260301_040743`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745`

## 21. Split-x2 rerun results (`100180`-`100202`) (2026-03-01)

Status check target range:

- pretrain: `100180` (`encdec1`), `100181` (`dualmask`)
- FT baseline/q_only matrix: `100188`-`100202`

### 21.1 Pretrain completion

Both split-x2 pretrains completed successfully (`Exit_status=0`).

- `100180.qjcm` (`patchnepa_ryE16e2`, `encdec_arch=1`)
  - done: `epoch 099 step 073650`
  - log: `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_encdec1_20260301_035212/run_ray_splitx2_encdec1.mr0.log`
  - ckpt dir: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_encdec1_20260301_035212`
- `100181.qjcm` (`patchnepa_ryE16d2`, `encdec_arch=0 + dual_mask`)
  - done: `epoch 099 step 073650`
  - log: `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_20260301_035220/run_ray_splitx2_dualmask.mr0.log`
  - ckpt dir: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220`

### 21.2 FT results (baseline = `patchnepa_ft_mode=qa_zeroa`)

All baseline jobs completed (`Exit_status=0`) and produced TEST metrics.

From `100180` (encdec1):

- `100188` (`obj_bg`): `TEST acc=0.7797`
- `100189` (`obj_only`): `TEST acc=0.7900`
- `100190` (`pb_t50_rs`): `TEST acc=0.7502`
- run set: `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_baseline_20260301_040736`

From `100181` (dualmask):

- `100194` (`obj_bg`): `TEST acc=0.7900`
- `100195` (`obj_only`): `TEST acc=0.8193`
- `100196` (`pb_t50_rs`): `TEST acc=0.7519`
- run set: `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740`

### 21.3 FT results (q_only / q_only+mean_q)

All q_only branches failed with the same DDP error (`Exit_status=1`):

- `100191`-`100193` (`encdec1`, `q_only`, `cls_max`)
- `100197`-`100199` (`dualmask`, `q_only`, `cls_max`)
- `100200`-`100202` (`dualmask`, `q_only`, `mean_q`)

Error signature (all three run sets):

- `RuntimeError: Expected to have finished reduction in the prior iteration ...`
- cause: unused parameters under DDP when `q_only` bypasses part of the module graph
  (`find_unused_parameters=False` path).

Error logs:

- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_encdec1_qonly_20260301_040738/*.err`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_20260301_040743/*.err`
- `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_qonly_meanq_20260301_040745/*.err`

### 21.4 Legacy held jobs (cancelled/replaced)

- `100182`-`100187` were cancelled/replaced by the explicit mode matrix above.

## 22. DDP stability hardening from external audit (2026-03-01)

Applied fixes after audit feedback:

1. Pretrain ray-binding graph stability (`BoundRayPatchEmbed`)
- file: `nepa3d/models/bound_ray_patch_embed.py`
- change:
  - replaced post-pool `torch.where(has_ray, out, 0)` with multiplicative masking
    (`out = out * mask`) for both `amax` and `mean` pooling branches.
- reason:
  - avoids potential per-rank graph disconnection when `has_ray` is all-false,
    which can trigger DDP "unused parameter" reduction errors.

2. Finetune `q_only` DDP compatibility
- file: `nepa3d/train/finetune_patch_cls.py`
- change:
  - `Accelerator` now uses `DistributedDataParallelKwargs`.
  - for `model_source=patchnepa` + `patchnepa_ft_mode=q_only`,
    set `find_unused_parameters=True`.
- reason:
  - `q_only` intentionally bypasses parts of the PatchNEPA graph,
    so strict DDP reduction without unused-parameter detection can fail.

Note:
- This does not change baseline (`qa_zeroa`) behavior.
- q_only matrix should be rerun to validate that previous `Exit_status=1` jobs
  now finish and produce TEST metrics.

## 23. q_only fix rerun submission (2026-03-01)

Rerun trigger:
- apply DDP stability fixes from Section 22
  - multiplicative mask in `BoundRayPatchEmbed`
  - `find_unused_parameters=True` for PatchNEPA `q_only` FT path

Submitted rerun matrix (no dependency, immediate run):

1) encdec1 q_only (`pooling=cls_max`)
- run set: `patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155`
- jobs: `100257` (`obj_bg`), `100258` (`obj_only`), `100259` (`pb_t50_rs`)
- ckpt: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_encdec1_20260301_035212/ckpt_latest.pt`

2) dualmask q_only (`pooling=cls_max`)
- run set: `patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157`
- jobs: `100260` (`obj_bg`), `100261` (`obj_only`), `100262` (`pb_t50_rs`)
- ckpt: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt`

3) dualmask q_only + mean pooling (`pooling=mean_q`)
- run set: `patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200`
- jobs: `100263` (`obj_bg`), `100264` (`obj_only`), `100265` (`pb_t50_rs`)
- ckpt: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt`

Status at submission snapshot:
- `100257`-`100265` all `R`.

## 24. q_only rerun (post-DDP-fix) partial results + low-accuracy diagnosis (2026-03-01)

Current status:

- `obj_bg` / `obj_only` completed for all three rerun branches.
- `pb_t50_rs` jobs are still running:
  - `100259` (encdec1 q_only cls_max)
  - `100262` (dualmask q_only cls_max)
  - `100265` (dualmask q_only mean_q)

### 24.1 Completed metrics (so far)

| pretrain branch | FT mode | pooling | obj_bg | obj_only |
|---|---|---|---:|---:|
| encdec1 (`100180`) | `qa_zeroa` | `cls_max` | `0.7797` | `0.7900` |
| encdec1 (`100180`) | `q_only` | `cls_max` | `0.7797` | `0.7797` |
| dualmask (`100181`) | `qa_zeroa` | `cls_max` | `0.7900` | `0.8193` |
| dualmask (`100181`) | `q_only` | `cls_max` | `0.7573` | `0.7814` |
| dualmask (`100181`) | `q_only` | `mean_q` | `0.7986` | `0.8090` |

Source logs:

- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/obj_only.out`

### 24.2 What is *not* the root cause

Checkpoint transfer integrity is clean in all compared runs:

- encdec1 baseline (`100188`): `[ckpt-adapt] ... mapped=393 direct=0 ...`, `missing=14 unexpected=0`
- dualmask baseline (`100194`): `[ckpt-adapt] ... mapped=247 direct=0 ...`, `missing=14 unexpected=0`

Interpretation:

- no evidence of wrong-key direct load (`direct` is `0`)
- no unexpected leftover tensors (`unexpected=0`)
- current accuracy gap is not explained by broken state-dict mapping.

### 24.3 Most likely contributors to low accuracy

1. FT mode mismatch is real, but not a single-factor explanation.
- `qa_zeroa` vs `q_only` changes results materially.
- For dualmask branch:
  - `q_only + cls_max` drops hard vs `qa_zeroa`
  - `q_only + mean_q` recovers `obj_bg` and is closer on `obj_only`
- This indicates pooling and token-selection policy matters, but does not fully close the gap to target.

2. Ray supervision density is low at pretrain token level.
- step-0 sanity in split-x2 pretrains shows many `TYPE_MISSING_RAY` tokens:
  - encdec1: `Q_RAY=22`, `A_RAY=22`, `MISSING_RAY=84`
  - dualmask: `Q_RAY=18`, `A_RAY=18`, `MISSING_RAY=92`
- i.e., a large fraction of patch positions carry no real ray signal each sample.
- This weakens effective ray-conditioning strength despite `USE_RAY_PATCH=1`.

3. Pretrain optimization dynamics differ strongly by branch.
- encdec1 run shows early loss escalation to O(1e-1~2e-1) after warmup.
- dualmask run stays mostly in low O(1e-3~1e-2) range in the same early phase.
- Downstream trend is consistent with this: dualmask branch outperforms encdec1 in current FT outcomes.

### 24.4 Current hard conclusion (interim)

- With completed runs only, PatchNEPA is still below strict Point-MAE reference band.
- Main open item before final conclusion:
  - wait for `pb_t50_rs` completion for `100259/100262/100265`,
  - then lock best FT mode (`qa_zeroa` vs `q_only+mean_q`) per pretrain branch for next ablation stage.

## 25. PatchNEPA FT recipe alignment trial (last_q + LLRD + patch_embed freeze) (2026-03-01)

Purpose:
- apply the three high-priority FT-side changes together on direct PatchNEPA finetune:
  1) classification anchor token: `last_q`
  2) layer-wise LR decay: `llrd_start=0.35`, `llrd_end=1.0`
  3) freeze patch embedding: `patchnepa_freeze_patch_embed=1`

Code/script changes applied:
- `nepa3d/models/patch_nepa.py`
  - `PatchTransformerNepaClassifier`: added `cls_token_source={bos,last_q,eos}` (default `last_q`)
- `nepa3d/train/finetune_patch_cls.py`
  - added args: `patchnepa_cls_token_source`, `patchnepa_freeze_patch_embed`, `llrd_start`, `llrd_end`
  - implemented PatchNEPA-only optimizer param-group build with depth-wise LR scaling (LLRD)
  - patch_embed freeze gate for direct PatchNEPA FT
- `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - pass-through env/args for the new PatchNEPA FT controls
- `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
  - defaulted PatchNEPA direct-FT to `POOLING=cls`, `HEAD_MODE=linear`, `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`,
    `PATCHNEPA_FREEZE_PATCH_EMBED=1`, `LLRD_START=0.35`, `LLRD_END=1.0`
- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
  - aligned submit defaults to PatchNEPA recipe (`POOLING=cls`, `HEAD_MODE=linear`) and wired new env vars

Submission strategy:
- scoped first to `obj_only` only (fast signal before full 3-variant rollout)

Submitted jobs:
1. encdec1 source pretrain (`100180` ckpt)
- job: `100314.qjcm`
- run_set: `patchnepaFT_objonly_encdec1_lastq_llrd_20260301_154822`
- variant: `obj_only`
- ckpt: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_encdec1_20260301_035212/ckpt_latest.pt`

2. dualmask source pretrain (`100181` ckpt)
- job: `100315.qjcm`
- run_set: `patchnepaFT_objonly_dualmask_lastq_llrd_20260301_154828`
- variant: `obj_only`
- ckpt: `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt`

Result policy for this trial:
- if either run improves over prior `obj_only` direct-FT baseline, expand same recipe to `obj_bg` and `pb_t50_rs`
- if both are flat/down, keep recipe fixed and isolate one factor at a time (token source vs freeze vs LLRD)

## 26. q_only rerun completion (`100259/100262/100265`) (2026-03-01)

Completion status:
- `100259.qjcm` (`encdec1 q_only cls_max`) -> `Exit_status=0`
- `100262.qjcm` (`dualmask q_only cls_max`) -> `Exit_status=0`
- `100265.qjcm` (`dualmask q_only mean_q`) -> `Exit_status=0`

`pb_t50_rs` TEST results:
- encdec1 `q_only + cls_max` (`100259`): `TEST acc=0.7533`
- dualmask `q_only + cls_max` (`100262`): `TEST acc=0.7512`
- dualmask `q_only + mean_q` (`100265`): `TEST acc=0.7488`

Source logs:
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_encdec1_qonly_fix_20260301_133155/pb_t50_rs.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_fix_20260301_133157/pb_t50_rs.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_rerun_dualmask_qonly_meanq_fix_20260301_133200/pb_t50_rs.out`

Mean-q note (dualmask branch):
- `obj_bg`: `0.7986`
- `obj_only`: `0.8090`
- `pb_t50_rs`: `0.7488`
- In this completed rerun, `mean_q` improved over `cls_max` on `obj_bg/obj_only` but not on `pb_t50_rs`.

## 27. FT recipe alignment trial results (`100314/100315`) (2026-03-01)

Jobs:
- `100314.qjcm` (encdec1 source ckpt) -> `Exit_status=0`
- `100315.qjcm` (dualmask source ckpt) -> `Exit_status=0`

Recipe (both):
- `model_source=patchnepa(direct)`
- `pooling=cls`, `patchnepa_ft_mode=qa_zeroa`
- `cls_token=last_q`
- `head_mode=linear`
- `patchnepa_freeze_patch_embed=1`
- `llrd=(0.35->1.00)`

Result (`obj_only`):
- `100314` (encdec1): `TEST acc=0.7711`
- `100315` (dualmask): `TEST acc=0.7676`

Source logs:
- `logs/sanity/patchnepa_ft/patchnepaFT_objonly_encdec1_lastq_llrd_20260301_154822/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objonly_dualmask_lastq_llrd_20260301_154828/obj_only.out`

Observation:
- In this first aligned-trial (`obj_only` only), both runs underperform prior best direct-FT obj_only numbers.
- Next step should isolate factors one-by-one (keep `last_q`, then toggle `freeze_patch_embed`, then toggle `LLRD`).

## 28. FT scheduler parity update (2D-style LLRD cosine warmup) (2026-03-01)

Implemented:
- 3D PatchNEPA finetune now supports 2D-style LLRD schedulers in addition to legacy static LLRD.
  - new arg: `--llrd_scheduler {static,llrd_cosine,llrd_cosine_warmup}`
  - file: `nepa3d/train/finetune_patch_cls.py`
- For `model_source=patchnepa` and `llrd_scheduler=llrd_cosine*`:
  - optimizer groups carry `llrd` and `llrd_scale`
  - step-wise scheduler is used (`schedulers.get_llrd_cosine_schedule*`)
  - warmup is converted from epoch units to step units (`warmup_epochs * steps_per_epoch`)
- Legacy behavior kept:
  - `llrd_scheduler=static` keeps per-layer LR fixed and applies epoch-wise warmup+cosine.

Launcher defaults updated:
- `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - pass-through env `LLRD_SCHEDULER` (default `static`)
- `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
  - default `LLRD_SCHEDULER=llrd_cosine_warmup`
- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
  - default `LLRD_SCHEDULER=llrd_cosine_warmup`
  - forwards `PATCHNEPA_CLS_TOKEN_SOURCE`, `PATCHNEPA_FREEZE_PATCH_EMBED`, `LLRD_START/END`, `LLRD_SCHEDULER`

Risk note:
- Values can look different from legacy runs because scheduler stepping changes from epoch-wise to step-wise.
- This is expected; compare only within matched scheduler mode.

Planned ablation axes (direct PatchNEPA FT):
1. `llrd_scheduler`: `static` vs `llrd_cosine_warmup`
2. `patchnepa_cls_token_source`: `last_q` vs `bos` (optional `eos`)
3. `patchnepa_freeze_patch_embed`: `1` vs `0`

## 29. obj_only one-factor ablation launch (`llrd` / `cls_token` / `freeze`) (2026-03-01)

Purpose:
- quantify contribution of each FT-side factor on the same source ckpt.

Common setup:
- source ckpt:
  - `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt`
- variant: `obj_only`
- direct PatchNEPA FT (`model_source=patchnepa`)
- strict eval: `val_split_mode=file`, `aug_eval=1`, `mc_test=10`

Submitted jobs:
1. baseline (new parity recipe)
- job: `100322.qjcm`
- run_set: `patchnepaFT_ablate_objonly_base_llrdcw_lastq_freeze1_20260301_170200`
- overrides:
  - `LLRD_SCHEDULER=llrd_cosine_warmup`
  - `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
  - `PATCHNEPA_FREEZE_PATCH_EMBED=1`

2. LLRD scheduler ablation (legacy static)
- job: `100324.qjcm`
- run_set: `patchnepaFT_ablate_objonly_static_lastq_freeze1_20260301_170205`
- overrides:
  - `LLRD_SCHEDULER=static`
  - `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
  - `PATCHNEPA_FREEZE_PATCH_EMBED=1`

3. cls token ablation (`bos`)
- job: `100321.qjcm`
- run_set: `patchnepaFT_ablate_objonly_llrdcw_bos_freeze1_20260301_170210`
- overrides:
  - `LLRD_SCHEDULER=llrd_cosine_warmup`
  - `PATCHNEPA_CLS_TOKEN_SOURCE=bos`
  - `PATCHNEPA_FREEZE_PATCH_EMBED=1`

4. freeze ablation (`patch_embed` unfrozen)
- job: `100323.qjcm`
- run_set: `patchnepaFT_ablate_objonly_llrdcw_lastq_freeze0_20260301_170215`
- overrides:
  - `LLRD_SCHEDULER=llrd_cosine_warmup`
  - `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
  - `PATCHNEPA_FREEZE_PATCH_EMBED=0`

Status at submit-time:
- `100321/100322/100323/100324` all entered `R` state.

Reference note (historical):
- in prior `q_only` branch, `mean_q` was strongest on `obj_only` (0.8090),
  but not uniformly best on all variants (`pb_t50_rs` was lower than `cls_max`).

## 30. Stage-2 sanity ablation launch (short pretrain -> short FT) (2026-03-01)

Question:
- can current result table isolate which QueryNEPA controls are effective in PatchNEPA?

Answer:
- not fully. Existing results are still partially confounded (layout/mask/type-pos/ray differ across runs).
- to resolve this, a one-factor-at-a-time sanity matrix was launched.

Protocol (lightweight):
- pretrain: short run (`epochs=12`, 16GPU)
- finetune: short direct PatchNEPA FT (`obj_only`, `epochs=120`)
- fixed FT recipe across all cases:
  - `model_source=patchnepa`, `pooling=cls`, `head_mode=linear`
  - `cls_token=last_q`, `freeze_patch_embed=1`
  - `llrd=(0.35->1.0)`, `llrd_scheduler=llrd_cosine_warmup`

Matrix (baseline + 5 controls):
1. baseline
- `split_sep`, `dual_mask on`, `type_aware=1`, `type_specific_pos=0`, `use_ray_patch=1`
- pretrain: `100347`, FT(dep): `100348`

2. layout ablation
- `interleave` (others baseline)
- pretrain: `100349`, FT(dep): `100350`

3. dual_mask ablation
- `dual_mask off` (`near=0, far=0, type_aware=0`)
- pretrain: `100351`, FT(dep): `100352`

4. dual_mask_type_aware ablation
- `type_aware=0` (near/far keep baseline)
- pretrain: `100353`, FT(dep): `100354`

5. type_specific_pos ablation
- `type_specific_pos=1` (others baseline)
- pretrain: `100355`, FT(dep): `100356`

6. ray usage ablation
- `use_ray_patch=0`, `n_ray=0` (others baseline)
- pretrain: `100357`, FT(dep): `100358`

Queue state at launch:
- pretrain `100347/100349/100351/100353/100355/100357`: `R`
- dependent FT `100348/100350/100352/100354/100356/100358`: `H` (afterok dependency)

Tracking file:
- `logs/sanity/patchnepa_stage2_sanity/patchnepa_stage2_sanity_20260301_162213/submitted_jobs.tsv`

## 31. Full-factorial Stage-2 sanity matrix (32 conditions) launched (2026-03-01)

User decision:
- run full 2^5 matrix (32 conditions), including the logically redundant branch
  (`dual_mask=off` with `type_aware=1`) to verify "effectively no-op" empirically.

Action:
- previous 6-case sanity set (`100347`~`100358`) was cancelled to avoid resource duplication.
- new full-factorial launcher added:
  - `scripts/sanity/submit_patchnepa_stage2_sanity_full32_qf.sh`

Factors (binary):
1. `layout`: `split_sep` / `interleave`
2. `dual_mask`: on/off (`near=0.5, far=0.1` vs `0,0`)
3. `dual_mask_type_aware`: 1/0
4. `type_specific_pos`: 1/0
5. `use_ray_patch`: 1/0 (`n_ray=1024` vs `0`)

Protocol:
- pretrain: short (`epochs=12`, 16GPU), one-pass mix
- finetune: short direct PatchNEPA FT (`obj_only`, `epochs=120`), `afterok` dependency
- fixed FT controls across all 32 conditions:
  - `pooling=cls`, `head_mode=linear`, `cls_token=last_q`
  - `freeze_patch_embed=1`
  - `llrd=(0.35->1.0)`, `llrd_scheduler=llrd_cosine_warmup`

Run root:
- `patchnepa_stage2_sanity32_20260301_162811`

Submitted jobs:
- pretrain: `100359` ~ `100421` (odd IDs)
- dependent FT: `100360` ~ `100422` (even IDs)
- queue snapshot at launch:
  - all pretrain jobs: `R`
  - all dependent FT jobs: `H` (`afterok`)

Tracking tables:
- master TSV:
  - `logs/sanity/patchnepa_stage2_sanity/patchnepa_stage2_sanity32_20260301_162811/submitted_jobs.tsv`
- per-case submit logs:
  - `logs/sanity/patchnepa_stage2_sanity/patchnepa_stage2_sanity32_20260301_162811/*.pretrain.submit.log`
  - `logs/sanity/patchnepa_stage2_sanity/patchnepa_stage2_sanity32_20260301_162811/*.ft.submit.log`

## 32. Current best combo snapshot + E300 rerun launch (2026-03-01)

Current best (direct PatchNEPA FT, from completed runs):
- metric: `obj_only TEST acc=0.8193`
- log:
  - `logs/sanity/patchnepa_ft/patchnepaFT_splitx2_dualmask_baseline_20260301_040740/obj_only.out`
- FT recipe snapshot (from log header):
  - `model_source=patchnepa(direct)`
  - `pooling=cls_max`
  - `patchnepa_ft_mode=qa_zeroa`
  - `head_mode=pointmae_mlp`
  - `is_causal=0`
  - `use_ray_patch=1`, `n_ray=1024`
- source ckpt used by that FT:
  - `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt`

Note:
- absolute top over all historical logs includes non-mainline adapter path (`patchcls`) and is excluded from direct PatchNEPA mainline claim.

Requested rerun:
- run same split+dualmask ray pretrain recipe with `pretrain epochs=300`, then FT (`obj_only`, 300 epochs).

Submitted:
1. pretrain E300
- `100424.qjcm`
- run_set: `patchnepa_ray_splitx2_dualmask_E300_20260301_170800`
- key config:
  - `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
  - `epochs=300`

2. dependent FT (obj_only)
- `100425.qjcm` (`afterok:100424`)
- run_set: `patchnepaFT_objonly_fromE300_dualmask_splitsep_20260301_170815`
- explicit FT controls (old-best approximation):
  - `pooling=cls_max`
  - `head_mode=pointmae_mlp`
  - `patchnepa_ft_mode=qa_zeroa`
  - `patchnepa_cls_token_source=bos`
  - `patchnepa_freeze_patch_embed=0`
  - `llrd_scheduler=static`, `llrd_start=1.0`, `llrd_end=1.0`

Additional dependent FT submissions (same controls as `100425`):
- `100426.qjcm` (`afterok:100424`) `variant=obj_bg`
- `100427.qjcm` (`afterok:100424`) `variant=pb_t50_rs`
- run_set:
  - `patchnepaFT_bgpb_fromE300_dualmask_splitsep_20260301_163523`

## 33. B-width direct PatchNEPA probe launch (2026-03-01)

Goal:
- keep direct PatchNEPA path and run a single B-width probe:
  - `d_model=768`, `n_layers=12`, `n_heads=12`
- verify if width expansion improves transfer before broader rollout.

Submitted:
1. pretrain (Ray split+dualmask, B-width)
- `100430.qjcm` (`patchnepa_ryB768`)
- run_set: `patchnepa_ray_splitx2_dualmask_B768_probe_20260301_165314`
- key config:
  - `D_MODEL=768`, `N_LAYERS=12`, `N_HEADS=12`, `MLP_RATIO=4.0`
  - `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`

2. dependent FT (obj_only, direct PatchNEPA)
- `100431.qjcm` (`afterok:100430`)
- run_set: `patchnepaFT_objonly_fromB768_probe_20260301_165321`
- FT controls (aligned to current best direct recipe):
  - `pooling=cls_max`, `head_mode=pointmae_mlp`
  - `patchnepa_ft_mode=qa_zeroa`, `cls_token_source=bos`
  - `patchnepa_freeze_patch_embed=0`
  - `llrd_scheduler=static`, `llrd_start=1.0`, `llrd_end=1.0`
  - `val_split_mode=file`, `aug_eval=1`, `mc_test=10`

Implementation note:
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh` now forwards
  `D_MODEL/N_LAYERS/N_HEADS/MLP_RATIO` through `qsub -v`, so width/depth overrides
  are reliably propagated to multinode jobs.

## 34. B-width E300 launch (direct PatchNEPA, 2026-03-01)

Rationale:
- model width can materially change transfer quality in Stage-2; do not block on prior E300 result.
- launch B-width (`768/12/12`) at full pretrain horizon (`300 epochs`) as a direct comparand.

Submitted:
1. pretrain (Ray split+dualmask, B-width, E300)
- `100432.qjcm` (`patchnepa_ryB8E300`)
- run_set: `patchnepa_ray_splitx2_dualmask_B768_E300_20260301_165452`
- key config:
  - `D_MODEL=768`, `N_LAYERS=12`, `N_HEADS=12`, `MLP_RATIO=4.0`
  - `EPOCHS=300`
  - `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`

2. dependent FT (obj_only, direct PatchNEPA)
- `100433.qjcm` (`afterok:100432`)
- run_set: `patchnepaFT_objonly_fromB768_E300_20260301_165504`
- FT controls (kept consistent with current best direct baseline):
  - `pooling=cls_max`, `head_mode=pointmae_mlp`
  - `patchnepa_ft_mode=qa_zeroa`, `cls_token_source=bos`
  - `patchnepa_freeze_patch_embed=0`
  - `llrd_scheduler=static`, `llrd_start=1.0`, `llrd_end=1.0`
  - `val_split_mode=file`, `aug_eval=1`, `mc_test=10`

Operational note:
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh` now forwards
  `D_MODEL/N_LAYERS/N_HEADS/MLP_RATIO` through `qsub -v` to avoid silent fallback
  to default `384/12/6` in multinode jobs.

## 35. EMA implementation + raw vs ema A/B launch (2026-03-01)

Implementation (2D-NEPA style adaptation):
- pretrain now supports EMA model tracking and checkpoint save/load:
  - `nepa3d/train/pretrain_patch_nepa.py`
  - args: `--use_ema`, `--ema_decay`
  - checkpoint payload includes `model_ema` when enabled
- launcher plumbing added:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - envs: `USE_EMA`, `EMA_DECAY`
- finetune now supports selecting EMA weights from pretrain ckpt:
  - `nepa3d/train/finetune_patch_cls.py`
  - arg: `--ckpt_use_ema` (0: `model`, 1: `model_ema`)
- finetune launcher plumbing:
  - `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
  - env: `CKPT_USE_EMA`

A/B protocol (same pretrain settings, same FT recipe; only ckpt source differs):
- source pretrain recipe aligned to existing B-width probe (`768/12/12`, split+dualmask+ray, 100 epochs)

Submitted:
1. pretrain with EMA enabled
- `100434.qjcm` (`patchnepa_ryB8EMA`)
- run_set: `patchnepa_ray_splitx2_dualmask_B768_EMA100_20260301_170241`
- key delta vs baseline probe:
  - `USE_EMA=1`, `EMA_DECAY=0.9999`

2. dependent FT from same ckpt (raw path)
- `100435.qjcm` (`afterok:100434`)
- run_set: `patchnepaFT_objonly_fromEMA100_raw_20260301_170323`
- `CKPT_USE_EMA=0`

3. dependent FT from same ckpt (ema path)
- `100436.qjcm` (`afterok:100434`)
- run_set: `patchnepaFT_objonly_fromEMA100_ema_20260301_170323`
- `CKPT_USE_EMA=1`

FT controls kept identical between 100435 and 100436:
- `model_source=patchnepa`
- `pooling=cls_max`, `head_mode=pointmae_mlp`
- `patchnepa_ft_mode=qa_zeroa`, `cls_token_source=bos`
- `patchnepa_freeze_patch_embed=0`
- `llrd_scheduler=static`, `llrd_start=1.0`, `llrd_end=1.0`
- strict eval: `val_split_mode=file`, `aug_eval=1`, `mc_test=10`

## 36. Stage-2 sanity32 results snapshot (for factor discussion) (2026-03-01)

Target question:
- whether QueryNEPA controls reproduce in PatchNEPA:
  - `split_sep vs interleave`
  - `dual_mask on/off`
  - `dual_mask_type_aware on/off`
  - `type_specific_pos on/off`
  - `use_ray_patch on/off`

Case code mapping (`patchnepa_stage2_sanity32_20260301_162811`):
- `lS/lI`: `layout=split_sep/interleave`
- `d1/d0`: `dual_mask on/off`
- `ta1/ta0`: `dual_mask_type_aware on/off`
- `tp1/tp0`: `type_specific_pos on/off`
- `r1/r0`: `use_ray_patch on/off`

### 36.1 Completed FT results (valid: split branch)

All `lS_*` FT jobs (`100360`~`100390`, even IDs) finished with `Exit_status=0`.
`obj_only TEST acc`:

| FT job | case | TEST acc |
|---|---|---:|
| `100360` | `lS_d1_ta1_tp0_r1` | `0.5095` |
| `100362` | `lS_d1_ta1_tp0_r0` | `0.4699` |
| `100364` | `lS_d1_ta1_tp1_r1` | `0.3941` |
| `100366` | `lS_d1_ta1_tp1_r0` | `0.5594` |
| `100368` | `lS_d1_ta0_tp0_r1` | `0.3993` |
| `100370` | `lS_d1_ta0_tp0_r0` | `0.5938` |
| `100372` | `lS_d1_ta0_tp1_r1` | `0.4355` |
| `100374` | `lS_d1_ta0_tp1_r0` | `0.3838` |
| `100376` | `lS_d0_ta1_tp0_r1` | `0.3769` |
| `100378` | `lS_d0_ta1_tp0_r0` | `0.5525` |
| `100380` | `lS_d0_ta1_tp1_r1` | `0.4406` |
| `100382` | `lS_d0_ta1_tp1_r0` | `0.5594` |
| `100384` | `lS_d0_ta0_tp0_r1` | `0.3769` |
| `100386` | `lS_d0_ta0_tp0_r0` | `0.5525` |
| `100388` | `lS_d0_ta0_tp1_r1` | `0.4406` |
| `100390` | `lS_d0_ta0_tp1_r0` | `0.5594` |

### 36.2 Interleave branch status (invalid for comparison)

`lI_*` FT jobs (`100392`~`100422`, even IDs) are all `Exit_status=1` with no `TEST acc`.
Direct cause from FT logs:
- missing ckpt at expected path:
  - `runs/patchnepa_rayqa/patchnepa_stage2_sanity32_20260301_162811_lI_.../ckpt_latest.pt`

Root cause in corresponding `lI_*` pretrain logs:
- DDP reduction error (`Expected to have finished reduction... parameters that were not used`)
- therefore pretrain failed before checkpoint write, while outer job state still ended as `F/0`.

Representative logs:
- pretrain fail example:
  - `logs/patch_nepa_pretrain/patchnepa_stage2_sanity32_20260301_162811_lI_d1_ta1_tp0_r1/run_patchnepa_stage2_sanity32_20260301_162811_lI_d1_ta1_tp0_r1.mr0.log`
- dependent FT fail example:
  - `logs/sanity/patchnepa_ft/patchnepa_stage2_sanity32_20260301_162811_ft_lI_d1_ta1_tp0_r1/obj_only.err`

### 36.3 Factor-wise provisional effect (split-only subset)

Since `interleave` is currently invalid, the following are **split-only provisional deltas**.

Paired delta (`1 - 0`) over matched conditions (8 pairs each):
- `dual_mask`: `-0.0142`
- `dual_mask_type_aware`: `+0.0151`
- `type_specific_pos`: `-0.0073`
- `use_ray_patch`: `-0.1072`

Interpretation at this stage:
- only `split_sep` branch is analyzable right now.
- strongest observed effect in this short-protocol subset is negative shift from `ray on`.
- no claim on `split_sep vs interleave` yet, because `interleave` branch is currently invalid due to pretrain failure.

Required next step for the originally intended QueryNEPA-style discussion:
- rerun `lI_*` branch after DDP unused-parameter issue is fixed in pretrain path, then compare `split_sep vs interleave` on the same short protocol.

## 37. Interleave pretrain DDP fix + skip-k launch (2026-03-01)

Interleave failure reason (root cause):
- `qa_layout=interleave` with `qa_sep_token=0` does not consume `sep_token` in forward path.
- DDP (find_unused_parameters=off) reported unused parameter index `2` (matched `sep_token`) and aborted pretrain.
- downstream FT then failed with missing checkpoint (`ckpt_latest.pt` absent).

Fix applied:
- file: `nepa3d/models/patch_nepa.py`
- change: always connect `sep_token` to graph with a no-op before embedding add:
  - `tokens = tokens + (self.sep_token.sum() * 0.0)`
- effect: avoids layout-dependent unused-parameter reduction failure while keeping numerics unchanged.

Additional launch controls for skip-k:
- files:
  - `nepa3d/train/pretrain_patch_nepa.py`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
- added args/env:
  - `nepa_skip_k`, `nepa_multi_k`
  - policy switch: `SKIPK_DISABLE_DUAL_MASK=1` (if skip-k active, force dual-mask off)

Submitted jobs:
1. skip-k mainline probe (k=4, dual-mask off by policy)
- pretrain: `100437.qjcm` (`patchnepa_rySK4`)
- run_set: `patchnepa_ray_skipk4_20260301_173024`
- dependent FT (`obj_only`): `100438.qjcm` (`afterok:100437`)
- FT run_set: `patchnepaFT_objonly_from_skipk4_20260301_173024`

2. interleave fix verification run (short sanity)
- pretrain: `100439.qjcm` (`pnpa_lI_fix`)
- run_set: `patchnepa_stage2_interleave_fix_20260301_173036`
- config: `qa_layout=interleave`, `qa_sep_token=0`, ray on, short pretrain (`epochs=12`)
- dependent FT (`obj_only`): `100440.qjcm` (`afterok:100439`)
- FT run_set: `patchnepaFT_objonly_interleave_fix_20260301_173036`

Current status at submit snapshot:
- running: `100437`, `100439`
- hold (dependency): `100438`, `100440`

## 38. Policy correction: global_batch=128 fixed + SEP policy lock (2026-03-01)

User correction applied:
- Stage-2 pretrain must keep `effective_global_batch=128` fixed for comparability.
- `QA_SEP_TOKEN=0` is treated as invalid policy drift; Stage-2 now enforces `QA_SEP_TOKEN=1`.

Code/script updates:
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - default `BATCH=8` (16 GPU => global 128)
  - strict checks:
    - reject `QA_SEP_TOKEN!=1`
    - reject `effective_global_batch!=128` when `STAGE2_REQUIRE_GLOBAL_BATCH128=1`
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - default `BATCH=8`
  - strict checks:
    - reject `QA_SEP_TOKEN!=1`
    - reject `BATCH*NPROC_PER_NODE*RT_QF != 128` when `STAGE2_REQUIRE_GLOBAL_BATCH128=1`
- `scripts/sanity/submit_patchnepa_stage2_sanity_ablation_qf.sh`
  - interleave case changed to `QA_SEP_TOKEN=1`

Job handling:
- previous probes (`100437`, `100439`) are excluded from strict comparison due policy drift (`global_batch=256` / `QA_SEP_TOKEN=0` in interleave probe).
- replacement submissions (policy-compliant):
  - pretrain skip-k: `100443.qjcm` (`patchnepa_rySK4g`), `BATCH=8`
  - dependent FT: `100444.qjcm` (`obj_only`, `afterok:100443`)
  - pretrain interleave sanity: `100445.qjcm` (`pnpa_lI_fixg`), `QA_LAYOUT=interleave`, `QA_SEP_TOKEN=1`, `BATCH=8`
  - dependent FT: `100446.qjcm` (`obj_only`, `afterok:100445`)

## 39. Unlogged finished jobs found by audit (2026-03-01)

Audit scope:
- recent range check around Stage-2 sanity32 and skip-k/interleave probes.

Found finished jobs that were not explicitly listed before:
1. `100421.qjcm` (`pnpa32_lI_d0_ta0_tp1_r0`)
- status: `F`, `Exit_status=0`, walltime `00:00:39`
- effective config included `QA_LAYOUT=interleave`, `QA_SEP_TOKEN=0`.
- log shows DDP unused-parameter crash (`Parameter indices ...: 2`, i.e. `sep_token` path not used).
- result validity: **invalid pretrain** (no usable checkpoint for strict comparison).

2. `100420.qjcm` (`pn_obj_only`) [dependent on `100419`]
- status: `F`, `Exit_status=1`, walltime `00:00:16`
- output log only contains launcher header; no train/eval loop, no `TEST acc`.
- result validity: **invalid FT**.

3. `100422.qjcm` (`pn_obj_only`) [dependent on `100421`]
- status: `F`, `Exit_status=1`, walltime `00:00:16`
- output log only contains launcher header; no train/eval loop, no `TEST acc`.
- result validity: **invalid FT**.

Conclusion:
- no additional valid `TEST acc` records were discovered from these unlogged finished jobs.
- they are failure artifacts from the old `interleave + QA_SEP_TOKEN=0` branch and are excluded from analysis.

## 40. Multi-k comparison run added (2026-03-01)

Request:
- add `multi-k` as direct comparison target against current `single-k (k=4)` run.

Technical note:
- `qsub -v` cannot pass comma-containing env values safely in this launcher path.
- parser was updated to accept multiple separators for `nepa_multi_k`:
  - file: `nepa3d/train/pretrain_patch_nepa.py`
  - `_parse_skip_k_list` now splits by regex `[\s,;:+|]+`
- submission uses `NEPA_MULTI_K=1+2+4` (equivalent to `{1,2,4}`).

Submitted jobs (policy-compliant: `global_batch=128`, `QA_SEP_TOKEN=1`):
- pretrain: `100451.qjcm` (`patchnepa_ryMK124`)
  - run_set: `patchnepa_ray_multik124_gb128_20260301_174748`
  - key: `NEPA_SKIP_K=1`, `NEPA_MULTI_K=1+2+4`, `SKIPK_DISABLE_DUAL_MASK=1`
  - `BATCH=8`, `16GPU` => `global_batch=128`
- dependent FT (`obj_only`): `100452.qjcm` (`afterok:100451`)
  - run_set: `patchnepaFT_objonly_from_multik124_gb128_20260301_174748`

Comparator pair now:
- single-k: `100443` (`k=4`, dual-mask off by policy)
- multi-k: `100451` (`k in {1,2,4}`, dual-mask off by policy)

## 41. Completion check: recent Stage-2 jobs all finished (2026-03-01)

Queue snapshot:
- `qstat` returned empty (no running/holding jobs at check time).

Recent pretrain/FT chain outcomes:

1) E300 dualmask split_sep chain
- pretrain: `100424.qjcm` (`patchnepa_ryE300`) -> `Exit_status=0`, ckpt exists
- FT from `100424`:
  - `100425` (`obj_only`): `TEST acc=0.7866`
  - `100426` (`obj_bg`): `TEST acc=0.8072`
  - `100427` (`pb_t50_rs`): `TEST acc=0.7720`

2) B768 probe / E300 / EMA pair
- `100430` pretrain -> `100431` FT(obj_only): `0.7315`
- `100432` pretrain -> `100433` FT(obj_only): `0.7315`
- `100434` pretrain (EMA enabled) ->
  - `100435` FT from raw ckpt(obj_only): `0.7401`
  - `100436` FT from ema ckpt(obj_only): `0.7969`

3) Old policy-drift runs (excluded)
- `100437` (`skip-k`, old global-batch drift): `Exit_status=271` (excluded)
- `100438` dependent FT: no valid output (`out_missing`)
- `100439` old interleave probe (`QA_SEP_TOKEN=0` era): excluded
- `100440` dependent FT: no valid output (`out_missing`)

4) Policy-compliant replacements (`global_batch=128`, `QA_SEP_TOKEN=1`)
- single-k (`k=4`):
  - pretrain `100443` -> `Exit_status=0`, ckpt exists
  - FT `100444` (`obj_only`): `TEST acc=0.7160`
- interleave fix sanity:
  - pretrain `100445` -> `Exit_status=0`, ckpt exists
  - FT `100446` (`obj_only`): `TEST acc=0.7642`
- multi-k (`k={1,2,4}`):
  - pretrain `100451` -> `Exit_status=0`, ckpt exists
  - FT `100452` (`obj_only`): `TEST acc=0.7108`

Notes:
- `100420/100422` are failed FT jobs (`Exit_status=1`, no `TEST acc`) and remain invalid.
- `100421` is the old interleave pretrain artifact (`ckpt_latest.pt` absent) and remains invalid for comparison.

## 42. Log-backed insights for latest completed runs (2026-03-01)

This section summarizes *why* recent runs behaved as observed, based on actual pretrain/FT logs.

### 42.1 E300 split_sep + dual-mask (mainline-like) remains strongest

Run pair:
- pretrain: `100424` (`split_sep`, `dual_mask on`, type-aware on, E300)
- FT: `100425/100426/100427`

Observed:
- pretrain tail loss is very low and stable (`~7.22e-4` mean over last 10 rank0 logs).
  - source: `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_E300_20260301_170800/run_patchnepa_ray_splitx2_dualmask_E300_20260301_170800.mr0.log`
- FT results are the current best set in this latest batch:
  - `obj_only=0.7866` (`100425`)
  - `obj_bg=0.8072` (`100426`)
  - `pb_t50_rs=0.7720` (`100427`)

Insight:
- Long-horizon pretrain (E300) with the established split+dual-mask recipe still gives the most reliable transfer among tested variants.

### 42.2 B768 scaling did not improve transfer under current recipe

Run pairs:
- `100430 -> 100431` (B768 probe)
- `100432 -> 100433` (B768 + E300)

Observed:
- FT stayed low (`obj_only=0.7315` for both).
- Pretrain logs showed non-trivial noise/spikes (e.g., `100432` had a large spike `~9.27e-2`) and no transfer gain despite larger model.

Insight:
- Under current optimization/data recipe, scaling to B-size did not translate into better downstream accuracy; optimization stability/recipe mismatch is likely dominating capacity.

### 42.3 EMA helped substantially on same pretrain run

Run pair:
- source pretrain with EMA enabled: `100434`
- FT from raw: `100435` (`obj_only=0.7401`)
- FT from ema: `100436` (`obj_only=0.7969`)

Observed:
- same upstream run, only checkpoint source differs (raw vs ema), with `+0.0568` absolute gain for EMA.

Insight:
- EMA is a high-impact stabilizer in this setup and should be treated as near-default for this branch.

### 42.4 Policy-compliant skip-k / interleave / multi-k comparison (global_batch=128)

Runs:
- single-k: `100443 -> 100444` (`k=4`, dual-mask off by policy)
- interleave fix sanity: `100445 -> 100446` (`interleave`, `sep=1`, dual-mask off)
- multi-k: `100451 -> 100452` (`k in {1,2,4}`, dual-mask off)

Observed FT (`obj_only`):
- single-k: `0.7160`
- interleave: `0.7642`
- multi-k: `0.7108`

Pretrain-tail behavior:
- single-k (`100443`): low loss tail (`~1.23e-3` mean last10)
- multi-k (`100451`): high loss scale tail (`~1.67e-1` mean last10)

Insight:
- Current multi-k formulation is likely too hard/unbalanced without weighting/curriculum (large objective scale, weak transfer).
- single-k (k=4) converged numerically but still transferred poorly, suggesting low pretrain loss alone is not sufficient as a proxy for class-transfer quality in this setting.
- interleave (with corrected `sep=1`) is valid now, but under the short sanity protocol it is still below the E300 split+dual-mask line.

### 42.5 val/test gap warning (especially on pb_t50_rs)

Observed in `100427` (`pb_t50_rs`):
- best `val_acc` reached `0.9330`
- but `TEST acc=0.7720`

Insight:
- There is a large validation-to-test gap on this variant. Any recipe changes should be judged primarily by test and repeated runs, not best-val alone.

### 42.6 Invalid artifacts kept excluded

- `100420`, `100422`: `Exit_status=1`, no test result.
- `100421`: old interleave artifact (`QA_SEP_TOKEN=0` era), no valid `ckpt_latest.pt`.
- `100437/100439` families remain excluded from strict comparison due earlier policy drift.

## 43. Correction: B768 (100/300) pretrain validity (2026-03-01)

Important correction after full log inspection:
- `100430` (B768, `EPOCHS=100`) and `100432` (B768, `EPOCHS=300`) both contain distributed launcher failures.
- Although `qstat -xf` shows `Exit_status=0`, node logs include `ChildFailedError` and `pbsdsh ... exit status 1`.

Evidence:
- `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_B768_probe_20260301_165314/run_ray_splitx2_dualmask_B768_probe.mr0.log`
  - `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`
- `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_B768_probe_20260301_165314/run_ray_splitx2_dualmask_B768_probe.pbs.log`
  - `pbsdsh: task ... exit status 1`
- `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_B768_E300_20260301_165452/run_ray_splitx2_dualmask_B768_E300.mr0.log`
  - `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`
- `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_B768_E300_20260301_165452/run_ray_splitx2_dualmask_B768_E300.pbs.log`
  - `pbsdsh: task ... exit status 1`

Implication:
- The FT results from these partial checkpoints (`100431`, `100433`, both `TEST acc=0.7315`) are not strict-valid for final B768 100-vs-300 conclusions.
- Keep them as diagnostic-only until a clean B768 rerun (no ChildFailedError, explicit `Done. checkpoints` line) is completed.

## 44. B768 failure root-cause fix + strict rerun submission (2026-03-01)

Reason for rerun:
- `100430/100432` were partial-invalid not by model logic mismatch, but by distributed runtime failure.
- Root signals from per-node logs:
  - NCCL watchdog timeout on `ALLREDUCE`
  - `CUDA error: Invalid access of peer GPU memory over nvlink`
  - `CUDA error: uncorrectable ECC error encountered`
- Therefore previous B768 FT numbers were diagnostic-only and not strict-valid for 100-vs-300 comparison.

Stability fixes applied to launch path:
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - added optional ECC preflight (`ENABLE_ECC_PREFLIGHT=1`) using
    `nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total`
  - added optional NCCL stable mode (`NCCL_STABLE_MODE=1`) that sets:
    - `NCCL_P2P_DISABLE=1`
    - `NCCL_NET_GDR_LEVEL=0`
    - `TORCH_NCCL_ENABLE_MONITORING=1`
    - `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200`
  - node-entry log now prints effective NCCL stability flags.
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - added same stable-mode env handling and startup log line:
    - `nccl: stable_mode=...`
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - forwards NCCL/ECC knobs via `qsub -v`:
    - `NCCL_STABLE_MODE`, `ENABLE_ECC_PREFLIGHT`,
      `NCCL_P2P_DISABLE`, `NCCL_NET_GDR_LEVEL`,
      `TORCH_NCCL_ENABLE_MONITORING`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`

Reruns submitted (same B768 recipe, strict policy-compliant):

1. B768 E100 retry
- pretrain: `100547.qjcm` (`patchnepa_ryB768r2`)
- run_set:
  - `patchnepa_ray_splitx2_dualmask_B768_probe_retry_20260301_230000`
- config highlights:
  - `D_MODEL=768`, `N_LAYERS=12`, `N_HEADS=12`
  - `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
  - `global_batch=128` (`BATCH=8`, `16GPU`)
  - `NCCL_STABLE_MODE=1`, `ENABLE_ECC_PREFLIGHT=1`
- dependent FT:
  - `100548.qjcm` (`obj_only`, `afterok:100547`)
  - run_set: `patchnepaFT_objonly_fromB768_probe_retry_20260301_230010`

2. B768 E300 retry
- pretrain: `100549.qjcm` (`patchnepa_ryB8E3r2`)
- run_set:
  - `patchnepa_ray_splitx2_dualmask_B768_E300_retry_20260301_230020`
- config highlights: same as above, `EPOCHS=300`
- dependent FT:
  - `100550.qjcm` (`obj_only`, `afterok:100549`)
  - run_set: `patchnepaFT_objonly_fromB768_E300_retry_20260301_230030`

Status at submission check:
- running: `100547`, `100549`
- holding (dependency): `100548`, `100550`

## 45. Pointcloud distance fairness guard (2026-03-01)

Update:
- `nepa3d/backends/pointcloud_backend.py` now enforces fail-fast for pointcloud distance pools:
  - `pointcloud` / `pointcloud_noray` require `pt_dist_pc_pool`.
  - if missing, backend raises `KeyError` immediately.

Rationale:
- avoids silent fallback to legacy `pt_dist_pool` (which may be mesh-derived in old caches),
  preventing fairness leakage in point-only claims.

Compatibility:
- `pointcloud_meshray` keeps legacy fallback behavior (`pt_dist_pool`) for explicit ablation/repro paths.
- current `shapenet_cache_v0` already includes `pt_dist_pc_pool` (checked), so mainline pointcloud pretrain is unaffected.

## 46. FPS strict fail-fast + FPS/Random A-B submission (2026-03-01)

Code-policy hardening:

- `nepa3d/data/dataset.py`
  - `pt_sample_mode='fps'` now hard-fails when FPS order key is missing.
  - removed warning-only behavior that allowed silent on-the-fly FPS fallback.
- `nepa3d/token/tokenizer.py`
  - `_sample_point_indices(..., pt_sample_mode='fps')` now raises when cached FPS order is not provided.
  - removed on-the-fly FPS fallback path for strict reproducibility.

Rationale:

- Stage-2 strict comparisons should not run with mixed execution paths
  (`cached order` vs `on-the-fly FPS`) because throughput and token partitions differ.

Cache readiness check:

- `shapenet_cache_v0` sample check confirms `pt_fps_order` exists.
- no additional FPS backfill is required for this specific corpus.

2D-comparability A/B submission (sampling-mode cut only):

Pretrain A (FPS fixed):
- job: `100559.qjcm` (`patchnepa_ryFPS`)
- run_set: `patchnepa_ray_fpscmp_fps_20260301_205158`
- key settings:
  - `PT_SAMPLE_MODE=fps`
  - `PT_FPS_KEY=pt_fps_order`
  - `POINT_ORDER_MODE=morton`
  - `qa_layout=split_sep`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `use_ray_patch=1`, `n_ray=1024`
  - `global_batch=128` (16GPU)

Pretrain B (Random baseline):
- job: `100560.qjcm` (`patchnepa_ryRAND`)
- run_set: `patchnepa_ray_fpscmp_rand_20260301_205203`
- key settings:
  - `PT_SAMPLE_MODE=random`
  - all other recipe knobs matched to Pretrain A

Dependent FT (obj_only only, direct PatchNEPA FT):
- from `100559`:
  - `100561.qjcm` (`pn_obj_only`), hold via `depend=afterok:100559`
  - run_set: `patchnepaFT_from_fpscmp_fps_20260301_205210`
  - ckpt: `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_fps_20260301_205158/ckpt_latest.pt`
- from `100560`:
  - `100562.qjcm` (`pn_obj_only`), hold via `depend=afterok:100560`
  - run_set: `patchnepaFT_from_fpscmp_rand_20260301_205215`
  - ckpt: `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_rand_20260301_205203/ckpt_latest.pt`

## 47. Re-submit after launcher env propagation fix (2026-03-01)

Issue found immediately after Section 46 submission:

- first FPS/Random pair (`100559`/`100560`) aborted at init with
  `RuntimeError: Invalid value for environment variable: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`
- cause: empty NCCL env strings were propagated via `qsub -v` and interpreted by torch.distributed.

Fixes applied:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - sanitize optional NCCL vars by unsetting empty strings before launch.
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - same sanitization added to `node_entry.sh` after env import.
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - fixed env propagation for sampling keys:
    - now forwards `PT_FPS_KEY`, `PT_RFPS_M` (in addition to `PT_SAMPLE_MODE`, `PT_RFPS_KEY`).

Superseded/invalid jobs:

- pretrain: `100559`, `100560` (init-failed)
- dependent FT: `100561`, `100562` (finished without valid upstream)

Replacement submissions (valid):

Pretrain A (FPS fixed):
- `100563.qjcm` (`patchnepa_ryFPS`)
- run_set: `patchnepa_ray_fpscmp_fps_20260301_205529`
- verified startup:
  - `pt_sample_mode=fps`
  - `pt_fps_key=pt_fps_order`
  - `effective_global_batch=128`

Pretrain B (Random baseline):
- `100564.qjcm` (`patchnepa_ryRAND`)
- run_set: `patchnepa_ray_fpscmp_rand_20260301_205535`
- verified startup:
  - `pt_sample_mode=random`
  - `effective_global_batch=128`

Dependent FT (obj_only):
- from `100563`: `100565.qjcm` (hold, `afterok:100563`)
- from `100564`: `100566.qjcm` (hold, `afterok:100564`)

## 48. 300-epoch status check (2026-03-01)

User check request: verify whether 300-epoch branch has finished.

### 48.1 Ray E300 (base width) — strict-valid complete

Pretrain:
- `100424.qjcm` (`patchnepa_ryE300`) finished (`job_state=F`, `Exit_status=0`).
- rank0 log confirms full completion:
  - `epoch 299 ...`
  - `Done. checkpoints: runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_E300_20260301_170800`

Dependent FT from `100424` (all finished):
- `100425` (`obj_only`): `TEST acc=0.7866`
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_fromE300_dualmask_splitsep_20260301_170815/obj_only.out`
- `100426` (`obj_bg`): `TEST acc=0.8072`
  - `logs/sanity/patchnepa_ft/obj_bg.out`
- `100427` (`pb_t50_rs`): `TEST acc=0.7720`
  - `logs/sanity/patchnepa_ft/pb_t50_rs.out`

### 48.2 B768 E300 branches — pretrain invalid (runtime fail)

B768 E300 (first):
- `100432.qjcm` finished in scheduler metadata (`Exit_status=0`) but pretrain log has
  `ChildFailedError` / `Signal 6 (SIGABRT)`.
- not strict-valid as completed pretrain.

B768 E300 retry:
- `100549.qjcm` similarly shows scheduler `Exit_status=0` but rank0 log has
  NCCL watchdog + `CUDA error: uncorrectable ECC error encountered` and `ChildFailedError`.
- not strict-valid as completed pretrain.

Dependent FT note:
- `100550` (`obj_only`) ran and produced `TEST acc=0.8158`,
  but this comes from an invalid upstream pretrain run and is diagnostic-only (not strict-valid for comparison).

### 48.3 Current active jobs (sampling A/B)

- `100563` (`FPS fixed`) running
- `100564` (`Random`) running
- `100565/100566` (`obj_only` FT) are dependency-hold until pretrains complete

## 49. B768 E300 retry path (ECC-focused rerun, 2026-03-01)

Why previous B768 E300 was invalid:
- `100432` and `100549` failed inside distributed runtime (`ChildFailedError`),
  with `100549` showing explicit `CUDA error: uncorrectable ECC error encountered`.
- this is hardware/runtime-side instability (node/GPU), not NEPA objective logic.

Hardening change applied:
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - ECC preflight now checks both:
    - `ecc.errors.uncorrected.volatile.total`
    - `retired_pages.pending`
  - node fails fast (`exit 86`) if either indicates unhealthy GPU state.

Retry submissions:

1) `100568` (`patchnepa_ryB8E3r3`) — invalid by config drift / terminated
- intended as B768 E300 retry, but startup showed unintended `dual_mask off`
  (`near=0.0 far=0.0 type_aware=0`), so this run is excluded.
- scheduler status: `Exit_status=271` (terminated path), no valid comparison claim.

2) `100570` (`patchnepa_ryB8E34`) — current valid retry
- run_set: `patchnepa_ray_splitx2_dualmask_B768_E300_retry4_20260301_210347`
- fixed recipe (matches dualmask branch):
  - `D_MODEL=768, N_LAYERS=12, N_HEADS=12, EPOCHS=300`
  - `qa_layout=split_sep, qa_tokens=1, encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `pt_sample_mode=rfps_cached, pt_rfps_key=pt_rfps_order_bank`
  - `global_batch=128 (16GPU)`
  - `NCCL_STABLE_MODE=1`, `ENABLE_ECC_PREFLIGHT=1`
- status at append time: running.

Dependent FT:
- `100571` (`obj_only`, `afterok:100570`) submitted and on hold.

## 50. Invalid beforeok chain guard (2026-03-01)

Issue observed:
- some failed pretrains (`100432`, `100549`) still ended as scheduler `Exit_status=0`,
  so `beforeok` dependent FT could start and produce misleading numbers.
- practical impact: invalid upstream checkpoints were accidentally consumed in downstream FT.

Root cause:
- runtime failure happened inside distributed child processes (`ChildFailedError`), but the
  multinode launch wrapper path did not enforce a strict "clean completion marker" gate.

Fix applied (launcher hardening):

1) `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
- on successful completion, only `MACHINE_RANK=0` writes:
  - `${PRETRAIN_DONE_MARKER}`
- log now also prints marker path when written.

2) `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
- creates per-run marker path:
  - `${RUN_DIR}/pretrain_done.marker`
- passes `PRETRAIN_DONE_MARKER` to node launcher env.
- after `pbsdsh` returns, requires marker existence.
  - missing marker => hard fail `exit 97`.

Policy after this fix:
- dependent FT runs are strict-valid only when upstream pretrain has both:
  - normal training completion in log (`Done. checkpoints ...`) and
  - launcher completion marker present.
- this blocks accidental `beforeok` pass-through from partial/failed pretrains.

## 51. B768 E300 strict rerun after fail-safe patch (2026-03-01)

User action request:
- rerun B768 E300 after fixing failure handling, because previous B768 E300 branches were invalid.

Cancelled old chain:
- pretrain `100570` and dependent FT `100571` (launched before Section 50 guard).

Resubmitted with fail-safe launcher:
- pretrain: `100572.qjcm` (`patchnepa_ryB8E35`)
  - run_set: `patchnepa_ray_splitx2_dualmask_B768_E300_retry5_20260301_210956`
  - recipe:
    - `D_MODEL=768, N_LAYERS=12, N_HEADS=12, EPOCHS=300`
    - `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=0`
    - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
    - `pt_sample_mode=rfps_cached`, `pt_rfps_key=pt_rfps_order_bank`
    - global batch `128` (`BATCH=8`, `16 GPU`)
    - `NCCL_STABLE_MODE=1`, `ENABLE_ECC_PREFLIGHT=1`
  - verification:
    - generated env contains `PRETRAIN_DONE_MARKER=.../pretrain_done.marker`

- dependent FT: `100573.qjcm` (`obj_only`)
  - dependency: `afterok:100572`
  - strict FT settings:
    - `MODEL_SOURCE=patchnepa`
    - `PATCHNEPA_FT_MODE=qa_zeroa`
    - `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
    - `PATCHNEPA_FREEZE_PATCH_EMBED=1`
    - `LLRD=0.35->1.0` (`llrd_cosine_warmup`)
    - `file + TTA10`

Status at append time:
- `100572` running
- `100573` hold

## 52. Ray-bind vs Ray-independent patch (same recipe, `ray_num_groups=32`) (2026-03-01)

Goal:

- isolate only ray grouping strategy difference while keeping Stage-2 ray recipe fixed.
- report `MISSING_RAY` change at step-0 sanity token counts.

Baseline (current bind mode):

- job: `100643.qjcm` (`patchnepa_rayqa`)
- key settings:
  - `RAY_ASSIGN_MODE=proxy_sphere`
  - `RAY_NUM_GROUPS=32`, `RAY_GROUP_SIZE=32`
  - `qa_layout=split_sep`, `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `global_batch=128`, `USE_EMA=0`
  - `PT_SAMPLE_MODE=fps`, `PT_FPS_KEY=pt_fps_order`
  - aug parity: `scale=[0.667,1.5]`, `translate=0.2`
- step-0 sanity (`run_ray_fpscmp_fps_augpm_dm.mr0.log`):
  - `Q_RAY=21`, `A_RAY=21`, `MISSING_RAY=86`

New (ray-independent patch):

- transient smoke run: `100698.qjcm` (terminated by user after confirming token behavior)
- canonical matched run: `100699.qjcm` (`patchnepa_ryIND`)
- key settings (matched to baseline except ray assign mode):
  - `RAY_ASSIGN_MODE=independent_fps_knn`
  - `RAY_NUM_GROUPS=32`, `RAY_GROUP_SIZE=32`
  - same `qa_layout`, `dual_mask`, batch, optimizer, aug, and data keys as baseline
- step-0 sanity (`run_ray_fpscmp_indpatch_dm_augpm.mr0.log`):
  - `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`

Observed delta (sample-0 sanity count):

- `MISSING_RAY`: `86 -> 0` (absolute `-86`, relative `-100%`)
- `Q_RAY`: `21 -> 32`
- `A_RAY`: `21 -> 32`

Implementation note required for this run:

- `nepa3d/train/pretrain_patch_nepa.py` now accepts
  `--ray_assign_mode independent_fps_knn` in argparse choices.

## 53. `fpscmp` branch status audit (2026-03-01)

Scope:

- `patchnepa_ray_fpscmp_*` runs for FPS-vs-random and ray-bind-vs-independent checks.

Status table (pretrain only):

| job | run_set | key recipe | status | validity |
|---|---|---|---|---|
| `100559` | `patchnepa_ray_fpscmp_fps_20260301_205158` | `pt_sample_mode=fps` | failed at startup (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` invalid) | invalid |
| `100560` | `patchnepa_ray_fpscmp_rand_20260301_205203` | `pt_sample_mode=random` | failed at startup (`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` invalid) | invalid |
| `100563` | `patchnepa_ray_fpscmp_fps_20260301_205529` | `fps`, no aug, dualmask on | completed (`Done. checkpoints ...`) | valid |
| `100564` | `patchnepa_ray_fpscmp_rand_20260301_205535` | `random`, no aug, dualmask on | completed (`Done. checkpoints ...`) | valid |
| `100642` | `patchnepa_ray_fpscmp_fps_augpm_20260301_215528` | `fps`, PM-like aug, dualmask off | terminated by user (`Exit_status=265`) | invalid |
| `100643` | `patchnepa_ray_fpscmp_fps_augpm_dm_20260301_215549` | `fps`, PM-like aug, dualmask on | running | pending |
| `100698` | `patchnepa_ray_fpscmp_indpatch_dm_20260301_222239` | `independent ray patch`, no aug | terminated by user after sanity check (`Exit_status=271`) | invalid for final metric |
| `100699` | `patchnepa_ray_fpscmp_indpatch_dm_augpm_20260301_222542` | `independent ray patch`, PM-like aug, dualmask on | running | pending |

Clarification on augmentation-side question:

- augmentation configuration itself is **not** the failure cause.
- the only aug run that ended (`100642`) is a manual termination (not model crash/regression).
- aug+dualmask branches (`100643`, `100699`) are active; final verdict should use their completed logs.

## 54. Point-only + EMA + E100 submission (comparison branch attach) (2026-03-01)

User request:

- add `point-only + EMA + E100` into current comparison branch.

Submitted pretrain:

- job: `100700.qjcm` (`patchnepa_ptE100EM`)
- run_set:
  - `patchnepa_ptonly_ema_e100_fpscmp_20260301_223010`
- save/log:
  - `runs/patchnepa_pointonly/patchnepa_ptonly_ema_e100_fpscmp_20260301_223010`
  - `logs/patch_nepa_pretrain/patchnepa_ptonly_ema_e100_fpscmp_20260301_223010`
- key recipe:
  - point-only: `USE_RAY_PATCH=0`, `N_RAY=0`, `STAGE2_REQUIRE_RAY=0`
  - `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
  - `EPOCHS=100`, global batch `128` (`BATCH=8`, `16 GPU`)
  - `USE_EMA=1`, `EMA_DECAY=0.9999`
  - `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=0`
  - `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `pt_sample_mode=fps`, `pt_fps_key=pt_fps_order`
  - PM-like pretrain aug: `scale=[0.667,1.5]`, `translate=0.2`

Dependent direct PatchNEPA FT (afterok:100700):

- `100701.qjcm` (`obj_bg`)
- `100702.qjcm` (`obj_only`)
- `100703.qjcm` (`pb_t50_rs`)

FT recipe snapshot:

- `MODEL_SOURCE=patchnepa`, `CKPT_USE_EMA=1`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`
- `PATCHNEPA_FT_MODE=qa_zeroa`, `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
- `PATCHNEPA_FREEZE_PATCH_EMBED=1`, `LLRD=0.35->1.0`
- point-only FT side: `USE_RAY_PATCH=0`, `N_RAY=0`

Status at append time:

- `100700`: running
- `100701/100702/100703`: hold (afterok dependency)

## 55. 3x2 short-screen submission (pretrain+FT set) (2026-03-01)

User request:

- do not screen with pretrain-only; submit pretrain+FT as a set.
- 3x2 matrix over `{random,fps,rfps_cached} x {aug off,on}`.

Screening protocol:

- pretrain: `E12` (short-screen, ~7k-10k step band), `16 GPU`, global batch `128`
- FT: dependent `obj_only` run for each pretrain (`afterok`), `E120`
- fixed across all 6 cells:
  - `ray_assign_mode=independent_fps_knn`
  - `ray_num_groups=32`, `ray_group_size=32`
  - `qa_layout=split_sep`, `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - `USE_EMA=0`

Submitted pairs:

| mode | aug | pretrain job | finetune job (`obj_only`) |
|---|---|---|---|
| random | off | `100704` | `100705` |
| random | on | `100706` | `100707` |
| fps | off | `100708` | `100709` |
| fps | on | `100710` | `100711` |
| rfps_cached | off | `100712` | `100713` |
| rfps_cached | on | `100714` | `100715` |

Run-set summary file:

- `logs/patch_nepa_pretrain/patchnepa_sanity6_20260301_224515_submission_summary.tsv`

Status at append time:

- pretrain `100704/706/708/710/712/714`: running
- finetune `100705/707/709/711/713/715`: hold (afterok dependency)

## 56. Completion Update (2026-03-01, table-default append)

Finished jobs recorded in this update:

### 56.1 `100643` status correction (bind+aug+dualmask)

| job | run_set | previous note | actual final status | validity |
|---|---|---|---|---|
| `100643` | `patchnepa_ray_fpscmp_fps_augpm_dm_20260301_215549` | running/pending | `Exit_status=97` (`job_state=F`) | invalid |

Failure signature:
- `run_ray_fpscmp_fps_augpm_dm.pbs.log` reports missing completion marker.
- non-rank0 node logs show shell parse failure near end (`unexpected EOF while looking for matching '"'`).

### 56.2 Sanity6 matrix completion (`100704`-`100715`)

All 6 short pretrains and dependent FT jobs finished with `Exit_status=0`.

| pretrain job | ft job | condition | FT result (`obj_only`, TEST acc) |
|---|---|---|---:|
| `100704` | `100705` | `random`, aug off | `0.6833` |
| `100706` | `100707` | `random`, aug on | `0.6196` |
| `100708` | `100709` | `fps`, aug off | `0.6454` |
| `100710` | `100711` | `fps`, aug on | `0.5318` |
| `100712` | `100713` | `rfps_cached`, aug off | `0.6368` |
| `100714` | `100715` | `rfps_cached`, aug on | `0.6317` |

Current ordering in this E12->E120 short-screen branch:
- best: `random/off (0.6833)`
- worst: `fps/on (0.5318)`

### 56.3 Point-only EMA E100 completion (`100700`)

| job | run_set | status | note |
|---|---|---|---|
| `100700` | `patchnepa_ptonly_ema_e100_fpscmp_20260301_223010` | `Exit_status=0` | pretrain completed |

Dependent FT jobs from `100700`:
- `100701` (`obj_bg`) running
- `100702` (`obj_only`) running
- `100703` (`pb_t50_rs`) running

### 56.4 Branch state snapshot after this append

| branch | latest state |
|---|---|
| bind+aug+dualmask (`100643`) | finished invalid |
| independent+aug+dualmask (`100699`) | running |
| ptonly+EMA E100 (`100700`) | finished valid |
| FT from ptonly+EMA (`100701/100702/100703`) | running |

## 57. `100643` invalid root cause + fix/resubmission (2026-03-01)

Question addressed:

- why `100643` became invalid and whether it should be fixed/resubmitted.

Root cause (confirmed from logs):

- `100643` ended with `Exit_status=97` due missing completion marker, while rank0 log had reached `Done. checkpoints`.
- non-rank logs (for example `qh055.patchnepa.log`) ended with:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh: line 368: unexpected EOF while looking for matching '"'`
- this means worker nodes parsed a broken/changed launcher script at runtime.

Fix applied:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - now snapshots launcher script at job start:
    - source: `${WORKDIR}/scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
    - snapshot: `${RUN_DIR}/nepa3d_pretrain_patch_nepa_qf.snapshot.sh`
  - node entry executes the snapshot (immutable per-job), not repo-head script.
  - topology and node logs now print snapshot path + sha256.

Resubmission (same recipe as `100643`):

| job | run_set | purpose | status |
|---|---|---|---|
| `100741` | `patchnepa_ray_fpscmp_fps_augpm_dm_fixlaunch_20260301_233109` | replacement of invalid `100643` under fixed launcher | running |

Early verification in `100741` PBS log:

- `LAUNCH_SCRIPT_RUN=.../ddp_patchnepa_100741.../nepa3d_pretrain_patch_nepa_qf.snapshot.sh`
- `launch_script_sha256=...`

So this branch is now corrected and re-running under deterministic launcher artifact.

## TODO (Hold Until Current Results Complete) — 2026-03-01

Policy (requested):
- Do not submit new jobs now; wait for current running jobs to finish and evaluate first.

Next action after current results are in:
- Promote selected settings from short-screen to full ShapeNet-standard pretrain (`E300`), with FT chained after pretrain completion.

Mandatory pretrain contract for that E300 promotion:
- `RAY_ASSIGN_MODE=independent_fps_knn`
- `RAY_NUM_GROUPS=32`
- `RAY_GROUP_SIZE=32`
- Step-0 sanity gate must show `MISSING_RAY=0` before accepting run as valid.

Candidate E300 settings to evaluate after hold is lifted:
- `random/off`
- `rfps_cached/off`
- `rfps_cached/on` (with `AUG_RECOMPUTE_DIST=1` for xyz-dist consistency under augmentation)

Note:
- Current phase is "result waiting" only; no additional submissions until this hold is explicitly cleared.

## 58. Additional completion check (2026-03-01)

Checked jobs:

| job | role | status | result |
|---|---|---|---|
| `100699` | pretrain (`ray_indpatch + aug + dualmask`) | `Exit_status=0` | completed, checkpoint written |
| `100700` | pretrain (`point-only + EMA E100`) | `Exit_status=0` | completed |
| `100701` | FT from `100700` (`obj_bg`) | `Exit_status=0` | `TEST acc=0.6799` |
| `100702` | FT from `100700` (`obj_only`) | `Exit_status=0` | `TEST acc=0.7074` |
| `100703` | FT from `100700` (`pb_t50_rs`) | `Exit_status=0` | `TEST acc=0.6072` |
| `100741` | pretrain replacement for invalid `100643` | running | pending |

Notes:
- `100699` log ended with `Done. checkpoints ...` and done marker emitted.
- NCCL `destroy_process_group` warning observed in old run logs is now addressed in training script for future runs.

## 59. LLRD-off default probe + independent-ray FT launch (2026-03-02)

Requested action:

- run one immediate FT probe with current defaults, but with LLRD effectively off.

Code/config default state used for this probe:

- `LLRD_START=1.0`
- `LLRD_END=1.0`
- `LLRD_SCHEDULER=static`
- `LLRD_MODE=linear` (default mode; no effect when start=end=1.0 and static scheduler)

Submission:

| job | run_set | source ckpt | variant | purpose | status |
|---|---|---|---|---|---|
| `100742` | `patchnepaFT_indpatch_llrdoff_default_20260302_0015` | `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_indpatch_dm_augpm_20260301_222542/ckpt_latest.pt` | `obj_only` | LLRD-off baseline FT probe | completed (`TEST acc=0.7762`) |

Context linkage (invalid branch + replacement):

| job | role | status |
|---|---|---|
| `100643` | bind+aug+dualmask old branch | invalid (`Exit_status=97`) |
| `100741` | fixed-launch replacement of `100643` | completed (`Exit_status=0`) |
| `100699` | independent-ray pretrain (`MISSING_RAY=0`) | completed (`Exit_status=0`) |
| `100742` | FT from `100699` | completed (`TEST acc=0.7762`) |

Note:
- This explicitly answers the gap: independent-ray (`MISSING_RAY=0`) branch now has downstream FT in queue (`100742`).
- `LLRDなし` in this run means **layer-wise LR decay is disabled** (`llrd_start=end=1.0`, `llrd_scheduler=static`).
- The base epoch schedule is still active (warmup + cosine), which is why per-epoch LR values still change in the log.
- Verification log header: `logs/sanity/patchnepa_ft/patchnepaFT_indpatch_llrdoff_default_20260302_0015/obj_only.out` (`llrd=(1.00->1.00) llrd_scheduler=static llrd_mode=linear`).


## 60. Point-only + EMA rerun under fixed FT defaults (2026-03-02)

Reason:
- previous point-only+EMA branch (`100700` -> `100701/100702/100703`) used old FT LLRD settings
  (`llrd=(0.35->1.00)`, `llrd_scheduler=llrd_cosine_warmup`) and is not suitable for
  strict comparison against current LLRD-off default policy.

Resubmission performed:

| job | role | run_set | key config | status |
|---|---|---|---|---|
| `100743` | pretrain | `patchnepa_ptonly_ema_e100_fpscmp_rerun_20260302_003621` | point-only (`USE_RAY_PATCH=0`, `N_RAY=0`), EMA on (`0.9999`), E100, batch-per-proc=8 (global 128), `pt_sample_mode=fps` | completed (`Exit_status=0`) |
| `100744` | FT `obj_bg` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_20260302_003621` | `depend=afterok:100743`, `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear` | hold |
| `100745` | FT `obj_only` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_20260302_003621` | same as above | hold |
| `100746` | FT `pb_t50_rs` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_20260302_003621` | same as above | hold |

Checkpoint path contract for dependent FT:
- `runs/patchnepa_pointonly/patchnepa_ptonly_ema_e100_fpscmp_rerun_20260302_003621/ckpt_latest.pt`

## 61. FT default policy update (Point-MAE-fair head/pooling) (2026-03-02)

Decision:
- PatchNEPA FT defaults are aligned to avoid weaker readout than Point-MAE.

Applied default changes:
- `POOLING=cls_max`
- `HEAD_MODE=pointmae_mlp`

Updated scripts:
- `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`

Also updated (hardcoded sanity launchers):
- `scripts/sanity/submit_patchnepa_stage2_sanity_ablation_qf.sh`
- `scripts/sanity/submit_patchnepa_stage2_sanity_full32_qf.sh`

Note:
- Existing completed runs with `pooling=cls` / `head_mode=linear` remain valid as historical results,
  but they should not be used as strict-comparable baselines against new default-policy runs.


## 62. FT policy enforcement for active point-only rerun chain (2026-03-02)

Adjustment after default update:
- previously queued dependent FT jobs from Section 60 (`100744`/`100745`/`100746`) were cancelled,
  because they were created before the head/pooling default switch and used `POOLING=cls`, `HEAD_MODE=linear`.

Replaced with policy-aligned dependent jobs:

| job | variant | dependency | effective FT readout defaults | status |
|---|---|---|---|---|
| `100747` | `obj_bg` | `afterok:100743` | `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp` | completed (`TEST acc=0.7711`) |
| `100748` | `obj_only` | `afterok:100743` | `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp` | completed (`TEST acc=0.7659`) |
| `100749` | `pb_t50_rs` | `afterok:100743` | `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp` | running |

New FT run set:
- `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316`

## 63. Missing-ray comparison status update (`100699` vs `100741`) (2026-03-02)

Pretrain completion update:

| job | branch | key setting | status |
|---|---|---|---|
| `100699` | independent ray patch | `ray_assign_mode=independent_fps_knn`, `pt_sample_mode=fps`, `aug on` | `Exit_status=0` |
| `100741` | bind ray replacement of invalid `100643` | `ray_assign_mode=proxy_sphere`, `pt_sample_mode=fps`, `aug on` | `Exit_status=0` |

Step-0 token sanity (rank0 sample-0):
- `100699`: `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`
- `100741`: `Q_RAY=21`, `A_RAY=21`, `MISSING_RAY=86`

Important comparability note:
- The frequently cited pair `0.8193` (`100181` family) vs `0.7762` (`100699 -> 100742`) is **not** a strict missing-ray A/B.
- `100181` family differs in pretrain recipe (`proxy_sphere + rfps_cached + no aug`) and FT readout defaults (`cls_max + pointmae_mlp`) from `100742` (`llrd-off static + cls + linear`).
- Therefore, current evidence is sufficient to say "`MISSING_RAY=0` alone did not trivially improve score under the `100699->100742` recipe", but **not** sufficient to conclude "missing-ray has no effect" in a strict single-factor sense.

Strict A/B requirement for final claim:
- run FT from `100741` with the **same** FT recipe used in `100742` (same pooling/head/LLRD/scheduler/eval),
- then compare `100741->FT` vs `100699->100742`.

## 64. Strict missing-ray A/B FT launch (`100741` side) (2026-03-02)

Action:
- submitted bind-side FT with the **same FT recipe** as `100742`, changing only source ckpt from independent (`100699`) to bind (`100741`).

Submission:

| job | run_set | variant | source ckpt | status |
|---|---|---|---|---|
| `100750.qjcm` | `patchnepaFT_bindfix_llrdoff_default_20260302_005751` | `obj_only` | `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_fps_augpm_dm_fixlaunch_20260301_233109/ckpt_latest.pt` | completed (`TEST acc=0.7367`) |

FT recipe parity check vs `100742`:
- same: `MODEL_SOURCE=patchnepa`, `PATCHNEPA_FT_MODE=qa_zeroa`, `POOLING=cls`, `HEAD_MODE=linear`,
  `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`, `PATCHNEPA_FREEZE_PATCH_EMBED=1`,
  `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`,
  `EPOCHS=300`, `BATCH=64(global)`, `VAL_SPLIT_MODE=file`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`.
- only difference: `CKPT` path (`100699` source in `100742` vs `100741` source in `100750`).

Expected output path:
- `logs/sanity/patchnepa_ft/patchnepaFT_bindfix_llrdoff_default_20260302_005751/obj_only.out`

## 65. Point-MAE official ckpt mapping sanity (`obj_bg`/`obj_only`) (2026-03-02)

Question addressed:

- why local Point-MAE `--test --ckpts scan_*.pth` numbers were far below README values,
  and whether this affects PatchNEPA interpretation.

Historical as-launched sanity (already recorded):

- `obj_bg + scan_objbg.pth` -> `73.3219`
- `obj_only + scan_objonly.pth` -> `81.7556`
- logs:
  - `logs/sanity/pointmae/pointmae_h5_parity_v3nonorm_fix_20260227_060439/pm_obj_bg_official_pointmae_h5_parity_v3nonorm_fix_20260227_060439.out`
  - `logs/sanity/pointmae/pointmae_h5_parity_v3nonorm_fix_20260227_060439/pm_obj_only_official_pointmae_h5_parity_v3nonorm_fix_20260227_060439.out`

Metadata evidence from those same logs:

- `scan_objbg.pth` loaded metric: `acc=88.2960` (`ckpts @ 166 epoch`)
- `scan_objonly.pth` loaded metric: `acc=90.0172` (`ckpts @ 250 epoch`)

Interpretation:

- `obj_bg` and `obj_only` checkpoint labels are effectively swapped for the historical as-launched sanity path.

Correction run (ckswap) submitted and completed:

| job | variant config | ckpt | result |
|---|---|---|---:|
| `100752.qjcm` | `obj_bg` | `scan_objonly.pth` | `TEST acc=90.1893` |
| `100753.qjcm` | `obj_only` | `scan_objbg.pth` | `TEST acc=87.9518` |

Logs:

- `logs/sanity/pointmae/pointmae_ckswap_objbg_from_objonly_20260302_0135.out`
- `logs/sanity/pointmae/pointmae_ckswap_objonly_from_objbg_20260302_0135.out`

Impact boundary for PatchNEPA discussion:

- PatchNEPA intra-run A/B conclusions are unchanged (same code/data pipeline within PatchNEPA).
- External gap-to-Point-MAE references for `obj_bg`/`obj_only` must use the variant-aligned sanity values above (or README targets), not the historical as-launched `73.3219/81.7556` pair.


## 66. New completion snapshot + summary (2026-03-02)

Confirmed completions since the previous append:

| job | run_set | variant | status | test_acc |
|---|---|---|---|---:|
| `100747` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `obj_bg` | `Exit_status=0` | `0.7711` |
| `100748` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `obj_only` | `Exit_status=0` | `0.7659` |
| `100750` | `patchnepaFT_bindfix_llrdoff_default_20260302_005751` | `obj_only` | `Exit_status=0` | `0.7367` |

Still running:

| job | run_set | variant | state |
|---|---|---|---|
| `100749` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `pb_t50_rs` | `R` |

Strict missing-ray A/B summary (same FT recipe):

- independent (`100699 -> 100742`): `obj_only=0.7762`
- bind (`100741 -> 100750`): `obj_only=0.7367`
- delta (`independent - bind`) = `+0.0395`

Interpretation (current evidence):

- under this matched recipe, reducing `MISSING_RAY` (`86 -> 0`) improved `obj_only`.
- this does not override other recipe families (for example `100181` line with different pretrain/FT settings).

## 67. Copy-risk diagnostic logging (`cos_tgt` vs `cos_prev`) + probe submit (2026-03-02)

Purpose:

- add a lightweight training-time diagnostic to detect copy-like behavior:
  - `cos_tgt = cos(z_hat[t], z[t+k])` (true target alignment)
  - `cos_prev = cos(z_hat[t], z[t])` (identity/copy tendency)
  - `gap = cos_tgt - cos_prev`
  - `copy_win = fraction(cos_prev >= cos_tgt)`
- all metrics are computed under the same NEPA target mask (answer-priority + `TYPE_MISSING_RAY` excluded).

Code change:

- `nepa3d/train/pretrain_patch_nepa.py`
  - added args: `--diag_copy`, `--diag_every`, `--diag_k`
  - added masked diagnostic computation and periodic log print in the main training loop.

Quick verification probe submitted:

- pretrain job: `100755.qjcm` (`patchnepa_dcopy`)
- run set: `patchnepa_diagcopy_probe_20260302_022115`
- run tag: `run_ray_diagcopy_probe`
- config intent:
  - `ray + split_sep + dualmask` (`near=0.5, far=0.1, type_aware=1`)
  - short run (`epochs=2`) for instrumentation sanity check only.
- logs:
  - PBS: `logs/patch_nepa_pretrain/patchnepa_diagcopy_probe_20260302_022115/run_ray_diagcopy_probe.pbs.log`
  - train (mr0): `logs/patch_nepa_pretrain/patchnepa_diagcopy_probe_20260302_022115/run_ray_diagcopy_probe.mr0.log`

Interpretation rule (for this probe):

- concerning sign: `gap <= 0` and/or `copy_win` stays high.
- healthy sign: `gap > 0` with moderate/low `copy_win`.

## 68. Completion update (`100749`) + full-step `gap/copy_win` trend (`100755`) (2026-03-02)

### 68.1 FT completion update (previously running)

Section 66 had `100749` as running. Final status is now confirmed:

| job | run_set | variant | status | test_acc |
|---|---|---|---|---:|
| `100749` | `patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316` | `pb_t50_rs` | `Exit_status=0` | `0.7207` |

Source log:
- `logs/sanity/patchnepa_ft/patchnepaFT_from_ptonlyema_rerun_llrdoff_pmhead_20260302_004316/pb_t50_rs.out`

### 68.2 Diagcopy probe completion and all-step summary

Probe job:
- `100755.qjcm` (`patchnepa_dcopy`) finished with `Exit_status=0`.
- `epochs=2`, `diag_every=50`, so diagnostic points are sampled at steps `0..1450` (30 points).

Full-step extract (step/gap/copy_win) is saved to:
- `nepa3d/docs/patch_nepa/diagcopy_probe_100755_gap_copywin.tsv`

Aggregate summary over all 30 points:

- `gap`:
  - mean: `0.0818`
  - min/max: `0.0008` / `0.1184`
  - first -> last: `0.0008 -> 0.0703` (delta `+0.0695`)
- `copy_win`:
  - mean: `0.3550`
  - min/max: `0.3048` / `0.4518`
  - first -> last: `0.4518 -> 0.3512` (delta `-0.1006`)

Early vs late windows:
- early (`step 0..700`, n=15): `gap=0.0778`, `copy_win=0.3678`
- late (`step 750..1450`, n=15): `gap=0.0858`, `copy_win=0.3422`

Interpretation for this probe run:
- `gap` stays positive at all logged points and is slightly higher in late phase.
- `copy_win` decreases from early to late, consistent with weaker copy tendency over training.

## 69. Priority-1 long-run health proof submit (`point-only + EMA + E300 + diag`) (2026-03-02)

User request:
- run long pretrain with continuous `gap/copy_win` monitoring (reviewer-priority #1).

Launcher pass-through fix (required before submission):
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - added env defaults and train args wiring:
    - `DIAG_COPY` (default `1`)
    - `DIAG_EVERY` (default `100`)
    - `DIAG_K` (default `1`)
  - startup header now prints `diag: copy=..., every=..., k=...`.
- `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - added the same `DIAG_*` vars in submit defaults.
  - forwarded `DIAG_*` through `qsub -v`.

Long-run submission:

| job | run_set | run_tag | status |
|---|---|---|---|
| `100756.qjcm` | `patchnepa_ptonly_ema_e300_diagcopy_20260302_024614` | `run_ptonly_ema_e300_diagcopy` | `R` |

Effective pretrain recipe:
- point-only:
  - `MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
  - `USE_RAY_PATCH=0`, `N_RAY=0`, `STAGE2_REQUIRE_RAY=0`
- optimization/topology:
  - `EPOCHS=300`, `USE_EMA=1`, `EMA_DECAY=0.9999`
  - `BATCH=8`, `RT_QF=4`, `NPROC_PER_NODE=4` -> global batch `128`
- sequence/objective:
  - `QA_LAYOUT=split_sep`, `QA_SEP_TOKEN=1`, `QA_TOKENS=1`, `ENCDEC_ARCH=0`
  - `DUAL_MASK_NEAR=0.5`, `DUAL_MASK_FAR=0.1`, `DUAL_MASK_TYPE_AWARE=1`, `DUAL_MASK_WINDOW=32`
- point sampling:
  - `PT_SAMPLE_MODE=fps`, `PT_FPS_KEY=pt_fps_order`, `PT_RFPS_KEY=pt_rfps_order_bank`
- diagnostics:
  - `DIAG_COPY=1`, `DIAG_EVERY=100`, `DIAG_K=1`

Paths:
- pretrain logs:
  - `logs/patch_nepa_pretrain/patchnepa_ptonly_ema_e300_diagcopy_20260302_024614`
- checkpoint root:
  - `runs/patchnepa_pointonly/patchnepa_ptonly_ema_e300_diagcopy_20260302_024614`

Planned post-run extraction:
- full-step `gap/copy_win` time series export to a docs TSV
- early/mid/late aggregate summary and final interpretation in this runlog.

## 70. Checkpoint-level copy-baseline diagnostic (`cos(pred,target)` vs `cos(prev,target)`) (2026-03-02)

User-requested verification:
- evaluate pretrained checkpoints directly and test whether prediction is distinguishable from a trivial copy baseline.
- criterion:
  - copy-like concern if `cos(pred,target) ≈ cos(prev,target)`
  - healthy if `cos(pred,target) >> cos(prev,target)`.

Implementation added:
- `scripts/sanity/diag_ckpt_copy_probe.py`
  - loads a pretrain ckpt (`model` or `model_ema`)
  - rebuilds the corresponding PatchNEPA model from ckpt args
  - runs mixed-pretrain batches and reports:
    - `cos(pred,target)`
    - `cos(prev,target)`
    - `cos(pred,prev)`
    - `lift = cos(pred,target) - cos(prev,target)`
    - `win_rate = P(cos(pred,target) > cos(prev,target))`

Runs executed (`k=1`, `max_batches=20`):

| ckpt | state | tokens | cos(pred,target) | cos(prev,target) | lift | win_rate |
|---|---|---:|---:|---:|---:|---:|
| `runs/patchnepa_pointonly/patchnepa_ptonly_ema_e100_fpscmp_rerun_20260302_003621/ckpt_latest.pt` | `model_ema` | 10240 | 0.982379 | 0.383854 | 0.598525 | 1.000000 |
| `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_fps_augpm_dm_fixlaunch_20260301_233109/ckpt_latest.pt` | `model` | 11654 | 0.997960 | 0.360807 | 0.637154 | 1.000000 |
| `runs/patchnepa_rayqa/patchnepa_ray_fpscmp_indpatch_dm_augpm_20260301_222542/ckpt_latest.pt` | `model` | 12512 | 0.997974 | 0.395841 | 0.602133 | 1.000000 |

Output logs:
- `logs/sanity/patchnepa_ft/diag_ckpt_copy_probe_ptonlyema_e100_20260302_0252.out`
- `logs/sanity/patchnepa_ft/diag_ckpt_copy_probe_bindfix_ema_e100_20260302_0255.out`
- `logs/sanity/patchnepa_ft/diag_ckpt_copy_probe_indpatch_ema_e100_20260302_0255.out`

Interpretation:
- for the tested checkpoints, `cos(pred,target)` is far above `cos(prev,target)` with large positive lift (`~0.60`) and `win_rate=1.0`.
- this does **not** support a simple "prediction is just previous-token copy" explanation.
- therefore, current evidence points to other bottlenecks (transfer recipe / objective-task mismatch / modality-use strategy), not trivial copy collapse.

## 71. Strict A/B submit for missing-token handling at best-known setting (+ split comparison) (2026-03-02)

User request:
- compare missing-token handling under the strongest direct PatchNEPA FT family (`splitx2 dualmask baseline` line with `obj_only=0.8193`).
- also check split (`encdec1`) counterpart under the same missing-token treatment.

Submission strategy:
- keep pretrain recipe aligned to splitx2 baseline family and change only ray patch assignment to:
  - `RAY_ASSIGN_MODE=independent_fps_knn` (missing-aware path; target is `MISSING_RAY=0` in ray-bearing samples).
- run both branches:
  - dualmask (`encdec_arch=0`, dualmask on),
  - split (`encdec_arch=1`, dualmask off).
- dependent FT is fixed to baseline-style settings:
  - `PATCHNEPA_FT_MODE=qa_zeroa`
  - `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
  - `PATCHNEPA_CLS_TOKEN_SOURCE=bos`, `PATCHNEPA_FREEZE_PATCH_EMBED=0`
  - `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`
  - strict eval unchanged (`file + TTA10`).

Submitted pretrains:

| job | run_set | branch | key delta | status |
|---|---|---|---|---|
| `100757.qjcm` | `patchnepa_ray_splitx2_dualmask_indpatch_cmp_20260302_031129` | dualmask (`encdec_arch=0`) | `RAY_ASSIGN_MODE=independent_fps_knn` | `R` |
| `100758.qjcm` | `patchnepa_ray_splitx2_encdec1_indpatch_cmp_20260302_031129` | split (`encdec_arch=1`) | `RAY_ASSIGN_MODE=independent_fps_knn` | `R` |

Submitted dependent FT jobs:

From `100757`:
- `100759.qjcm` (`obj_bg`)
- `100760.qjcm` (`obj_only`)
- `100761.qjcm` (`pb_t50_rs`)
- run set:
  - `patchnepaFT_from_dualmask_indpatch_cmp_20260302_031129`

From `100758`:
- `100762.qjcm` (`obj_bg`)
- `100763.qjcm` (`obj_only`)
- `100764.qjcm` (`pb_t50_rs`)
- run set:
  - `patchnepaFT_from_encdec1_indpatch_cmp_20260302_031129`

Submit logs/paths:
- pretrain:
  - `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_indpatch_cmp_20260302_031129`
  - `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_encdec1_indpatch_cmp_20260302_031129`
- finetune:
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_dualmask_indpatch_cmp_20260302_031129`
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_encdec1_indpatch_cmp_20260302_031129`

Comparison note:
- yes, split comparison is now possible by design:
  - `dualmask-indpatch` (`100757 -> 100759~100761`)
  - vs `encdec1-indpatch` (`100758 -> 100762~100764`)
  - both under identical FT recipe and global-batch policy.

## 72. Missing-safe patch update (loss+K/V guard) and resubmission (`100765~100772`, 2026-03-02)

User-requested pre-run update for missing-version was applied before the new submit chain.

Applied safety updates:

1. `nepa3d/models/causal_transformer.py`
- added 3D attention-mask support in self-attention (`(B,T,T)` bool/additive masks)
- added missing-token K/V blocking path when `dual_mask_type_aware=1`:
  - keys with `TYPE_MISSING_RAY` are masked out
  - missing-query self-diagonal is kept to avoid all-masked rows

2. `nepa3d/train/pretrain_patch_nepa.py`
- fail-fast guard added:
  - if `use_ray_patch=1` and `ray_assign_mode=independent_fps_knn`, then
    `dual_mask_type_aware` must be `1`; otherwise `ValueError`.

Status of previous cmp chain:

- `100757/100758` finished with `Exit_status=271` and are superseded.
- dependent FT `100759~100764` finalized without execution payload (dependency path from superseded pretrain).

Resubmitted safe chain:

Pretrain:

| job | run_set | branch | key settings | status |
|---|---|---|---|---|
| `100765.qjcm` | `patchnepa_ray_splitx2_dualmask_indpatch_safe_20260302_031834` | dualmask (`encdec_arch=0`) | `ray_assign=independent_fps_knn`, `dual_mask_type_aware=1`, global batch `128` | `R` |
| `100766.qjcm` | `patchnepa_ray_splitx2_encdec1_indpatch_safe_20260302_031834` | encdec1 (`encdec_arch=1`) | `ray_assign=independent_fps_knn`, `dual_mask_type_aware=1`, global batch `128` | `R` |

Startup sanity (both jobs):
- `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0` at step0 sample snapshot.

Dependent FT (`afterok`):

From `100765`:
- `100767` (`obj_bg`)
- `100768` (`obj_only`)
- `100769` (`pb_t50_rs`)
- run_set: `patchnepaFT_from_dualmask_indpatch_safe_20260302_031834`

From `100766`:
- `100770` (`obj_bg`)
- `100771` (`obj_only`)
- `100772` (`pb_t50_rs`)
- run_set: `patchnepaFT_from_encdec1_indpatch_safe_20260302_031834`

FT recipe (fixed for A/B parity):
- `PATCHNEPA_FT_MODE=qa_zeroa`
- `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
- `PATCHNEPA_CLS_TOKEN_SOURCE=bos`, `PATCHNEPA_FREEZE_PATCH_EMBED=0`
- `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`

Paths:
- pretrain logs:
  - `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_indpatch_safe_20260302_031834`
  - `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_encdec1_indpatch_safe_20260302_031834`
- finetune logs:
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_dualmask_indpatch_safe_20260302_031834`
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_encdec1_indpatch_safe_20260302_031834`

## 73. Strict FT A/B: pretrain vs scratch under identical PatchNEPA FT recipe (`obj_only`, 2026-03-02)

Purpose:
- isolate whether val->test gap behavior is due to pretrain itself or generic FT overfit by running strict A/B with only init-source changed.

A/B contract (fixed identical FT settings):
- variant: `obj_only`
- model path: `model_source=patchnepa(direct)`
- `PATCHNEPA_FT_MODE=qa_zeroa`
- `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
- `PATCHNEPA_CLS_TOKEN_SOURCE=last_q`
- `PATCHNEPA_FREEZE_PATCH_EMBED=0`
- `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`
- strict eval unchanged: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`

Submitted jobs:

| arm | run_set | init source | job | status |
|---|---|---|---|---|
| pretrain-init | `patchnepaFT_ab_objonly_pre_20260302_035338` | `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_20260301_035220/ckpt_latest.pt` | `100773.qjcm` | `R` |
| scratch-init | `patchnepaFT_ab_objonly_scratch_20260302_035345` | no ckpt (`patchnepa scratch init`) | `100774.qjcm` | `R` |

Logs:
- `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_pre_20260302_035338/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_scratch_20260302_035345/obj_only.out`

Code update enabling this strict A/B:
- `nepa3d/train/finetune_patch_cls.py`
  - `model_source=patchnepa` now allows `--ckpt` omitted (scratch init) for strict A/B only.
- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
  - added opt-in gate for empty ckpt submission: `PATCHNEPA_ALLOW_SCRATCH=1`.

## 74. W&B default enable (2026-03-02)

Policy update:

- from this point, both Stage-2 pretrain and finetune launchers enable W&B by default.
- logging is best-effort:
  - if `wandb` package/import/init fails, training continues with local logs.

Applied paths:

- pretrain trainer:
  - `nepa3d/train/pretrain_patch_nepa.py`
  - added args: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_run_name`,
    `--wandb_group`, `--wandb_tags`, `--wandb_mode`, `--wandb_log_every`
  - logs: step loss/lr, dual-mask ramp, copy-diagnostic metrics, epoch-end lr.
- finetune trainer:
  - `nepa3d/train/finetune_patch_cls.py`
  - added args: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_run_name`,
    `--wandb_group`, `--wandb_tags`, `--wandb_mode`
  - logs: epoch train/val metrics, best val tracking, final test metrics.

Launcher defaults:

- pretrain:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - default env: `USE_WANDB=1`, `WANDB_PROJECT=patchnepa-pretrain`.
- finetune:
  - `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
  - `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh`
  - default env: `USE_WANDB=1`, `WANDB_PROJECT=patchnepa-finetune`.

Runtime override examples:

- disable W&B for a run: `USE_WANDB=0`
- offline mode: `WANDB_MODE=offline`
- set entity/project/group/run:
  - `WANDB_ENTITY=<entity>`
  - `WANDB_PROJECT=<project>`
  - `WANDB_GROUP=<group>`
  - `WANDB_RUN_NAME=<run_name>`

## 75. Quick context-limitation probe (`local_only` vs `far_only`) on pretrain ckpt (2026-03-02)

Purpose:

- verify whether current objective is only "prev-copy trivial" vs requiring broader context.
- fast check using existing pretrain ckpt without launching a new job.

Probe setup:

- ckpt:
  - `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_indpatch_safe_20260302_031834/ckpt_latest.pt`
- batches:
  - `24` mini-batches from mixed-pretrain train stream (no aug).
- evaluated masks (all with `is_causal=1`, `window=32`):
  - `full_unmasked`: `near=0, far=0`
  - `local_only`: `near=0, far=1` (drop far history)
  - `far_only`: `near=1, far=0` (drop near history)
- metrics:
  - `nepa_loss`, `cos_tgt`, `cos_prev`, `gap=cos_tgt-cos_prev`, `copy_win`.

Results:

| type_aware | condition | loss | cos_tgt | cos_prev | gap | copy_win |
|---|---|---:|---:|---:|---:|---:|
| `1` | full_unmasked | 0.00142 | 0.99858 | 0.41996 | 0.57862 | 0.00000 |
| `1` | local_only | 0.00141 | 0.99859 | 0.42051 | 0.57808 | 0.00000 |
| `1` | far_only | 0.00148 | 0.99852 | 0.42085 | 0.57767 | 0.00000 |
| `0` | full_unmasked | 0.00381 | 0.99619 | 0.42364 | 0.57255 | 0.00000 |
| `0` | local_only | 0.15380 | 0.84620 | 0.49573 | 0.35046 | 0.00014 |
| `0` | far_only | 0.14520 | 0.85480 | 0.47989 | 0.37491 | 0.00395 |

Interpretation (strictly from this probe):

- `copy trivial` (prev-copy win) is still rejected: `copy_win` stays ~0 across settings.
- with `dual_mask_type_aware=1`, local/far restriction barely changes metrics because masking applies mainly to Q/Q edges; this probe is not strong enough to answer receptive-field needs under that mode.
- with `dual_mask_type_aware=0`, both local-only/far-only severely degrade vs unmasked, indicating non-trivial context dependence when masking is allowed to affect answer-side history.

Actionable note:

- to answer "local-only is enough?" for mainline (`type_aware=1`), add a targeted probe that directly limits Q->A / A->A visible context (not only Q/Q).

## 76. FT augmentation parity note vs Point-MAE (2026-03-02)

Clarification:

- current FT path is **Point-MAE-aligned**, but **not an exact implementation match**.
- mismatch is in scaling behavior:
  - current PatchNEPA FT aug path uses a **single scalar scale** (`s`) for xyz
  - Point-MAE `PointcloudScaleAndTranslate` uses **per-axis scale** (`sx, sy, sz`)

Code evidence:

- PatchNEPA FT aug preset (`pointmae`):
  - `nepa3d/train/finetune_patch_cls.py`
  - `nepa3d/data/cls_patch_dataset.py` -> `_apply_point_aug(...)`
  - `nepa3d/data/dataset.py` (`scale = uniform(...)` scalar; `dist *= scale`)
- Point-MAE transform:
  - `Point-MAE/datasets/data_transforms.py`
  - `PointcloudScaleAndTranslate` samples `size=[3]` scale.

Policy implication:

- describe current FT as "Point-MAE-like" or "Point-MAE-aligned", not strict-equal.
- for strict parity claims against Point-MAE, scale implementation should be matched first.

Required follow-up:

- update FT augmentation to support Point-MAE-equivalent anisotropic scale (x/y/z independent),
  then rerun strict comparison branch under matched FT defaults.

## 77. FT Point-MAE augmentation parity fix implemented (2026-03-02)

Decision:

- close the known FT augmentation gap: PatchNEPA FT `aug_preset=pointmae` now matches
  Point-MAE `PointcloudScaleAndTranslate` scale semantics.

What was different before:

- legacy PatchNEPA FT used isotropic scalar scale (`xyz *= s`).
- Point-MAE uses anisotropic per-axis scale (`xyz *= [sx, sy, sz]`).

Code changes:

- `nepa3d/data/cls_patch_dataset.py`
  - `PointAugConfig` extended with `pointmae_exact` flag.
  - added `_apply_pointmae_scale_translate(...)`:
    - per-axis scale + per-axis translation (Point-MAE equivalent)
    - normal-vector update under anisotropic scale (`inv(S)^T` + renorm)
  - `PatchClsPointDataset._maybe_augment` / `PatchClsArrayDataset._maybe_augment`
    now use this exact path when `pointmae_exact=1`.
- `nepa3d/train/finetune_patch_cls.py`
  - added arg: `--pointmae_exact_aug {0,1}` (default `1`)
  - `aug_preset=pointmae` now sets `pointmae_exact=bool(pointmae_exact_aug)`.
  - test dataset now sets `aug=bool(args.aug_eval)` so TTA transform is actually applied during eval votes.

Operational policy from now:

- strict Point-MAE parity FT comparisons should keep:
  - `aug_preset=pointmae`
  - `pointmae_exact_aug=1` (default)
- legacy isotropic-scalar behavior is allowed only for controlled ablation:
  - `pointmae_exact_aug=0`.

## 78. Val/Test gap insight (pb_t50_rs dominance) (2026-03-02)

Question:
- Is the largest val->test gap concentrated on `pb_t50_rs`, and does that imply current task design is too easy/shortcut-prone?

Log-backed observation:
- In strict-matched direct FT (`splitx2_dualmask_baseline_20260301_040740`), gap by variant is:
  - `obj_bg`: `best_val=0.8520`, `test=0.7900`, `gap=0.0620`
  - `obj_only`: `best_val=0.8924`, `test=0.8193`, `gap=0.0731`
  - `pb_t50_rs`: `best_val=0.9145`, `test=0.7519`, `gap=0.1626` (largest)
- Cross-run scan over completed direct FT logs shows top gap rows are repeatedly `pb_t50_rs` (`~0.15-0.16` band).

Interpretation (current evidence boundary):
- Yes, largest gap is consistently concentrated on `pb_t50_rs`.
- "`pb_t50_rs` has more information" alone is not sufficient to explain the behavior:
  - this variant also has much larger train/val cardinality (`train=10282`, `val=1134` in logs),
    which can raise validation fit,
  - but the persistent large drop at test indicates a robustness/generalization issue, not pure capacity shortage.
- Practical reading: current recipe can still exploit easier-in-split cues and is not yet forcing sufficiently hard transfer-relevant invariances on the hardest variant.

Action direction:
- treat `pb_t50_rs` as primary stress metric for objective/aug/task redesign decisions.
- avoid single-run conclusion from best-val only; lock claims on repeated TEST metrics under fixed recipe.
- prioritize ablations that increase "hardness" of transferable signals (instead of only increasing model/data scale).

## 79. Missing-ray A/B setting delta vs best-line gap (2026-03-02)

Question:
- "setting delta は何か？"
- "missing-ray 改善があるのに、なぜ current-best をまだ超えないのか？"

### 79.1 Strict missing-ray A/B (same FT recipe)

Pair used for strict A/B:
- independent side: `100699 -> 100742` (`obj_only=0.7762`)
- bind side: `100741 -> 100750` (`obj_only=0.7367`)
- delta: `+0.0395` (independent better)

Pretrain delta (A/Bで変えた主因):
- `ray_assign_mode` only:
  - independent: `independent_fps_knn`
  - bind: `proxy_sphere`
- token-level effect at step0:
  - independent: `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`
  - bind: `Q_RAY=21`, `A_RAY=21`, `MISSING_RAY=86`

FT parity check for this A/B:
- same FT recipe (`qa_zeroa`, `pooling=cls`, `head=linear`, `freeze_patch_embed=1`, `LLRD off/static`, `file+TTA10`)
- hence this pair supports: reducing missing-ray helped under that FT setting.

### 79.2 Why still below current-best (`obj_only=0.8193`)

Current best reference line:
- `100181 -> 100195` family (`splitx2_dualmask_baseline`), `obj_only=0.8193`

Critical config differences vs strict A/B line (`100699/100741 -> 100742/100750`):

1) FT head/pooling family differs (major)
- current-best line:
  - `pooling=cls_max`, `head_mode=pointmae_mlp`
- strict A/B line:
  - `pooling=cls`, `head_mode=linear`, `freeze_patch_embed=1`

2) Pretrain sampling/augmentation family differs (major)
- current-best pretrain (`100181`):
  - `pt_sample_mode=rfps_cached`, `aug off (scale=1, translate=0)`
- strict A/B pretrains (`100699/100741`):
  - `pt_sample_mode=fps`, `aug on (scale=[0.667,1.5], translate=0.2)`

3) Objective-side control differs
- strict A/B branch logs include `skip_k=[1]` path in pretrain header.
- current-best line was from earlier splitx2 baseline family without that surfaced control in header.

Interpretation:
- "missing-ray reduction" is a valid positive factor (A/B proves this),
  but it is not the only moving part.
- gap to `0.8193` is dominated by recipe-family differences (especially FT readout/head and pretrain sampling+aug regime),
  so current evidence does not support "missing-ray alone should beat best".

### 79.3 Latest in-progress split comparison status

From safe indpatch chain (`100765/100766` source):
- completed:
  - dualmask: `obj_bg=0.8021` (`100767`), `obj_only=0.7900` (`100768`)
  - encdec1: `obj_bg=0.7969` (`100770` log), `obj_only=0.8038` (`100771`)
- running/pending at snapshot:
  - `pb_t50_rs`: `100769` (dualmask), `100772` (encdec1)

Operational note:
- decide final branch ranking after both `pb_t50_rs` complete,
  then compare against current-best line with the same metric triplet (`obj_bg/obj_only/pb_t50_rs`).

## 80. Completion update (`100765`~`100774`, 2026-03-02)

Status at check time:

- active jobs in `qstat`: `100769.qjcm`, `100772.qjcm` (both `pb_t50_rs` FT).

Pretrain completion (safe indpatch chain):

- `100765.qjcm` (`patchnepa_ryDMs`) -> `job_state=F`, `Exit_status=0`
  - log: `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_dualmask_indpatch_safe_20260302_031834/run_ray_splitx2_dualmask_indpatch_safe.mr0.log`
  - checkpoint:
    - `runs/patchnepa_rayqa/patchnepa_ray_splitx2_dualmask_indpatch_safe_20260302_031834/ckpt_latest.pt`
- `100766.qjcm` (`patchnepa_ryEMs`) -> `job_state=F`, `Exit_status=0`
  - log: `logs/patch_nepa_pretrain/patchnepa_ray_splitx2_encdec1_indpatch_safe_20260302_031834/run_ray_splitx2_encdec1_indpatch_safe.mr0.log`
  - checkpoint:
    - `runs/patchnepa_rayqa/patchnepa_ray_splitx2_encdec1_indpatch_safe_20260302_031834/ckpt_latest.pt`

Step-0 token sanity (both pretrains):

- `Q_RAY=32`, `A_RAY=32`, `MISSING_RAY=0`

FT completion from safe indpatch chain:

| job | branch | variant | status | test_acc |
|---|---|---|---|---:|
| `100767` | dualmask (`100765` source) | `obj_bg` | `Exit_status=0` | `0.8021` |
| `100768` | dualmask (`100765` source) | `obj_only` | `Exit_status=0` | `0.7900` |
| `100770` | encdec1 (`100766` source) | `obj_bg` | `Exit_status=0` | `0.7969` |
| `100771` | encdec1 (`100766` source) | `obj_only` | `Exit_status=0` | `0.8038` |

Current running FT jobs (no final `TEST acc` yet):

- `100769` (`dualmask`, `pb_t50_rs`)
  - latest tail: `ep 246/300`, best logged `val_acc=0.9092`
  - log:
    - `logs/sanity/patchnepa_ft/patchnepaFT_from_dualmask_indpatch_safe_20260302_031834/pb_t50_rs.out`
- `100772` (`encdec1`, `pb_t50_rs`)
  - latest tail: `ep 104/300`, best logged `val_acc=0.8739`
  - log:
    - `logs/sanity/patchnepa_ft/patchnepaFT_from_encdec1_indpatch_safe_20260302_031834/pb_t50_rs.out`

Strict FT A/B (`pretrain-init vs scratch-init`) completion:

| job | arm | status | best_val | test_acc |
|---|---|---|---:|---:|
| `100773` | pretrain-init | `Exit_status=0` | `0.8610` | `0.8021` |
| `100774` | scratch-init | `Exit_status=0` | `0.8475` | `0.8055` |

Logs:

- `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_pre_20260302_035338/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_ab_objonly_scratch_20260302_035345/obj_only.out`

Interim interpretation:

- under the current fixed FT recipe, `obj_only` shows no clear pretrain-init advantage over scratch-init (`0.8021` vs `0.8055`, delta `-0.0034` for pretrain-init).
- final branch conclusion remains pending until `pb_t50_rs` (`100769/100772`) finishes.

## 81. Q1/Q2 validation matrix submitted (2026-03-02)

User-requested validation:

- Q1 (task diversity): point-only `random` vs point-only `rfps_cached`.
- Q2 (modal-specific transfer): `mix(mesh+udf)` vs point-only control under matched recipe.
- all branches use pretrain+dependent FT (`obj_only`) chaining.

Submission summary:

| role | question | job | run_set | state |
|---|---|---|---|---|
| pretrain | Q1-diversity | `100778.qjcm` | `patchnepa_q1_ptonly_random_e100_20260302_053501` | `R` |
| pretrain | Q1/Q2-control | `100779.qjcm` | `patchnepa_q1q2_ptonly_rfps_e100_20260302_053501` | `R` |
| pretrain | Q2-modal | `100780.qjcm` | `patchnepa_q2_mix_meshudf_rfps_e100_20260302_053501` | `R` |
| FT (`obj_only`) | from `100778` | `100781.qjcm` | `patchnepaFT_objonly_from_q1_ptonly_random_e100_20260302_053501` | `H` (`afterok:100778`) |
| FT (`obj_only`) | from `100779` | `100782.qjcm` | `patchnepaFT_objonly_from_q1q2_ptonly_rfps_e100_20260302_053501` | `H` (`afterok:100779`) |
| FT (`obj_only`) | from `100780` | `100783.qjcm` | `patchnepaFT_objonly_from_q2_mix_meshudf_rfps_e100_20260302_053501` | `H` (`afterok:100780`) |

Pretrain contract (common unless noted):

- topology/global batch: `16 GPU (4x4)`, `batch=8` per proc, global `128`
- sequence/objective: `qa_layout=split_sep`, `qa_tokens=1`, `encdec_arch=0`
- dual-mask: `near=0.5`, `far=0.1`, `window=32`, `type_aware=1`
- point-only path for Q1/control: `USE_RAY_PATCH=0`, `N_RAY=0`, `STAGE2_REQUIRE_RAY=0`
- answer-side signal: `use_pt_dist=1`, `pt_dist_key=pt_dist_pool`, `ablate_point_dist=0`
- optimizer: `lr=3e-4`, `scheduler=cosine`, `warmup_ratio=0.025`
- EMA: `use_ema=1`, `ema_decay=0.9999`

Axis-specific deltas:

- Q1-diversity (`100778` vs `100779`):
  - only sampling mode differs (`random` vs `rfps_cached`) under point-only pretrain recipe.
- Q2-modal (`100780` vs `100779`):
  - same sampling mode (`rfps_cached`) and optimizer topology;
  - mix source differs (`mesh+udf one-pass` vs `pointcloud-only one-pass`).

Dependent FT recipe (fixed across `100781`~`100783`):

- direct PatchNEPA FT (`MODEL_SOURCE=patchnepa`), `variant=obj_only`
- point-only FT side: `USE_RAY_PATCH=0`, `N_RAY=0`
- readout/head: `pooling=cls_max`, `head_mode=pointmae_mlp`
- mode/token: `patchnepa_ft_mode=qa_zeroa`, `cls_token_source=bos`
- LLRD/freeze: `freeze_patch_embed=0`, `llrd_start=end=1.0`, `llrd_scheduler=static`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`
- checkpoint source: `CKPT_USE_EMA=1` for all three branches.

Key logs:

- pretrain logs:
  - `logs/patch_nepa_pretrain/patchnepa_q1_ptonly_random_e100_20260302_053501/run_q1_ptonly_random.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_q1q2_ptonly_rfps_e100_20260302_053501/run_q1q2_ptonly_rfps.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_q2_mix_meshudf_rfps_e100_20260302_053501/run_q2_mix_meshudf_rfps.mr0.log`
- FT logs:
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1_ptonly_random_e100_20260302_053501/obj_only.out`
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1q2_ptonly_rfps_e100_20260302_053501/obj_only.out`
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q2_mix_meshudf_rfps_e100_20260302_053501/obj_only.out`
- submission table:
  - `logs/sanity/patchnepa_ft/patchnepa_q1q2_matrix_20260302_053501_jobs.tsv`

## 82. Q1/Q2 matrix expansion (`obj_bg`/`pb_t50_rs`) (2026-03-02)

User request:

- `obj_only` だけでなく、同一3系統で `obj_bg` / `pb_t50_rs` も追加。

Added dependent FT jobs (all `afterok`, same FT recipe as `100781`~`100783`):

| question | source pretrain | obj_bg | pb_t50_rs |
|---|---|---|---|
| Q1-diversity (`random`) | `100778` | `100784.qjcm` | `100785.qjcm` |
| Q1/Q2-control (`rfps_cached`) | `100779` | `100786.qjcm` | `100787.qjcm` |
| Q2-modal (`mix mesh+udf`) | `100780` | `100788.qjcm` | `100789.qjcm` |

Status at submission check:

- `100784`~`100789`: all `H` (`afterok` hold)
- existing `obj_only` chain remains:
  - `100781`~`100783`: `H`

FT recipe parity (all 9 FT jobs):

- `MODEL_SOURCE=patchnepa`, `PATCHNEPA_FT_MODE=qa_zeroa`
- `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
- `PATCHNEPA_CLS_TOKEN_SOURCE=bos`, `PATCHNEPA_FREEZE_PATCH_EMBED=0`
- `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`
- point-only FT side: `USE_RAY_PATCH=0`, `N_RAY=0`
- `CKPT_USE_EMA=1`

New FT log roots:

- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q1_ptonly_random_e100_20260302_053501`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q1_ptonly_random_e100_20260302_053501`
- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q1q2_ptonly_rfps_e100_20260302_053501`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q1q2_ptonly_rfps_e100_20260302_053501`
- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q2_mix_meshudf_rfps_e100_20260302_053501`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q2_mix_meshudf_rfps_e100_20260302_053501`

Tracking table updated:

- `logs/sanity/patchnepa_ft/patchnepa_q1q2_matrix_20260302_053501_jobs.tsv`

## 83. Q1/Q2 matrix completion + result readout (2026-03-02)

Completion status:

- pretrain `100778/100779/100780`: all `Exit_status=0`
- dependent FT `100781`~`100789`: all `Exit_status=0`

Pretrain status snapshot:

| job | question arm | status |
|---|---|---|
| `100778` | Q1-diversity (`point-only + random`) | `Exit_status=0` |
| `100779` | Q1/Q2-control (`point-only + rfps_cached`) | `Exit_status=0` |
| `100780` | Q2-modal (`mix(mesh+udf) + rfps_cached`) | `Exit_status=0` |

FT results (`TEST acc`, strict eval `file + TTA10`):

| question arm | obj_bg | obj_only | pb_t50_rs |
|---|---:|---:|---:|
| Q1-diversity (`100778` source) | `0.8158` (`100784`) | `0.8176` (`100781`) | `0.7724` (`100785`) |
| Q1/Q2-control (`100779` source) | `0.7900` (`100786`) | `0.8124` (`100782`) | `0.7474` (`100787`) |
| Q2-modal (`100780` source) | `0.8244` (`100788`) | `0.7935` (`100783`) | `0.7661` (`100789`) |

Q1 answer (`random` vs `rfps_cached`, point-only):

- `obj_bg`: `+0.0258` (`0.8158 - 0.7900`)
- `obj_only`: `+0.0052` (`0.8176 - 0.8124`)
- `pb_t50_rs`: `+0.0250` (`0.7724 - 0.7474`)

Readout:
- in this matched run, `random` is better on all three variants.
- this supports the hypothesis that fixed-order (`rfps_cached`) can underprovide task diversity relative to `random` in the current recipe.

Q2 answer (`mix(mesh+udf)` vs point-only control, both `rfps_cached`):

- `obj_bg`: `+0.0344` (`0.8244 - 0.7900`)
- `obj_only`: `-0.0189` (`0.7935 - 0.8124`)
- `pb_t50_rs`: `+0.0187` (`0.7661 - 0.7474`)

Readout:
- modal-mix is not uniformly better, but improves `obj_bg/pb_t50_rs` and drops `obj_only`.
- current evidence is compatible with "modal-specific signal can help some tasks", but not yet a stable all-variant gain.

Caution for interpretation:

- this matrix is single-seed/single-run per arm; claims should remain provisional until repeated runs close variance.
- for paper-level claim, repeat at least one additional seed per arm under the same contract.

Key logs:

- `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1_ptonly_random_e100_20260302_053501/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1q2_ptonly_rfps_e100_20260302_053501/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q2_mix_meshudf_rfps_e100_20260302_053501/obj_only.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q1_ptonly_random_e100_20260302_053501/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q1q2_ptonly_rfps_e100_20260302_053501/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_objbg_from_q2_mix_meshudf_rfps_e100_20260302_053501/obj_bg.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q1_ptonly_random_e100_20260302_053501/pb_t50_rs.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q1q2_ptonly_rfps_e100_20260302_053501/pb_t50_rs.out`
- `logs/sanity/patchnepa_ft/patchnepaFT_pbt50rs_from_q2_mix_meshudf_rfps_e100_20260302_053501/pb_t50_rs.out`

## 84. Loss-axis parity note (`1-cos` vs `-cos`) + logging update (2026-03-02)

Clarification:

- 3D PatchNEPA pretrain objective uses `loss_3d = 1 - cos(pred, target)`.
- 2D ViT-NEPA pretrain reports `loss_2d = -cos(pred, target)`.
- exact relation: `loss_2d = loss_3d - 1`.

Operational update:

- `nepa3d/train/pretrain_patch_nepa.py` now logs both:
  - console step log: `loss=` (`1-cos`) and `loss2d=` (`loss-1`)
  - W&B keys:
    - `train/loss` (`1-cos`, existing)
    - `train/loss_2d_equiv` (`-cos` equivalent, newly added)

Purpose:

- align PatchNEPA curves with 2D-NEPA paper-style y-axis without changing optimization behavior.

## 85. Content-only target ablation launch (`loss_target_mode=content_tokens`, 2026-03-02)

Purpose:

- test the root hypothesis that pretrain target leakage from `pos/type` makes the task too easy.
- run a minimal strict A/B by changing only loss target definition.

A/B contract:

- baseline reference (already completed):
  - pretrain `100778` (`point-only + random`) -> FT `100781` (`obj_only`)
- new ablation chain (this update):
  - same recipe as `100778 -> 100781`, except:
    - pretrain `loss_target_mode=content_tokens` (target=`tokens.detach()` before pos/type add)

Code updates for this ablation:

- `nepa3d/models/patch_nepa.py`
  - `PatchNepaOutput` now exposes `tokens` sequence.
  - `PatchTransformerNepa.nepa_loss(...)` now accepts optional `target=` tensor.
- `nepa3d/train/pretrain_patch_nepa.py`
  - new arg: `--loss_target_mode {full_z,content_tokens}` (default `full_z`)
  - training loss uses:
    - `full_z`: existing target (`out.z`)
    - `content_tokens`: content-only target (`out.tokens`)
- launcher plumbing:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh`
  - new env/arg pass-through: `LOSS_TARGET_MODE`.

Submitted jobs:

| role | job | run_set | key delta | status |
|---|---|---|---|---|
| pretrain | `101051.qjcm` | `pt_q1tc_0302_133221` | `LOSS_TARGET_MODE=content_tokens` | `F` (`Exit_status=271`) |
| FT (`obj_only`) | `101052.qjcm` | `patchnepaFT_objonly_from_q1_ptonly_random_tgtcontent_20260302_133330` | `afterok:101051` | `F` (dependency-hold path, `Time_Use=0`) |

Pretrain recipe parity vs `100778`:

- point-only (`USE_RAY_PATCH=0`, `N_RAY=0`, `STAGE2_REQUIRE_RAY=0`)
- `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
- sampling `PT_SAMPLE_MODE=random`
- dual-mask `near=0.5`, `far=0.1`, `window=32`, `type_aware=1`
- `EPOCHS=100`, `BATCH=8`, global batch `128`, `USE_EMA=1`, `EMA_DECAY=0.9999`
- only intentional change: `loss_target_mode`.

FT recipe parity vs `100781`:

- `MODEL_SOURCE=patchnepa`, `variant=obj_only`
- `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
- `PATCHNEPA_FT_MODE=qa_zeroa`, `PATCHNEPA_CLS_TOKEN_SOURCE=bos`
- `PATCHNEPA_FREEZE_PATCH_EMBED=0`
- `LLRD_START=1.0`, `LLRD_END=1.0`, `LLRD_SCHEDULER=static`, `LLRD_MODE=linear`
- strict eval: `val_split_mode=file`, `AUG_EVAL=1`, `MC_TEST=10`
- `CKPT_USE_EMA=1`.

Paths:

- pretrain logs:
  - `logs/patch_nepa_pretrain/pt_q1tc_0302_133221`
- finetune logs:
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1_ptonly_random_tgtcontent_20260302_133330`

Operational note:

- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh` hit PBS env-size error (`qsub: cannot send environment with the job`) in this environment.
- dependent FT was submitted directly via `qsub` + `scripts/finetune/patchnepa_scanobjectnn_finetune.sh` with minimal `-v` set to keep recipe parity.

## 86. Content-target rerun with W&B enabled (`USE_WANDB=1`) (2026-03-02)

Reason:

- Section 85 chain (`101051`) was launched with `USE_WANDB=0`, so run was not visible in W&B.
- user requested same-condition rerun with W&B logging enabled.

Submitted replacement/parallel chain (same recipe as Section 85 except W&B on):

| role | job | run_set | key delta | status |
|---|---|---|---|---|
| pretrain | `101062.qjcm` | `pt_q1tcwb2_0302_134311` | `USE_WANDB=1` | `R` |
| FT (`obj_only`) | `101063.qjcm` | `patchnepaFT_objonly_from_q1_ptonly_random_tgtcontent_wb_20260302_134322` | `afterok:101062` | `H` |

W&B contract for `101062`:

- `WANDB_PROJECT=patchnepa-pretrain`
- `WANDB_RUN_NAME=r_q1tcwb2`
- `WANDB_GROUP=patchnepa-pretrain`
- `WANDB_TAGS=t`
- startup header confirms:
  - `wandb: use=1 project=patchnepa-pretrain ... run=r_q1tcwb2 ...`

Paths:

- pretrain logs:
  - `logs/patch_nepa_pretrain/pt_q1tcwb2_0302_134311`
  - per-node rank0 launcher log:
    - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101062.qjcm_r_q1tcwb2/logs/qh168.patchnepa.log`
- FT logs:
  - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1_ptonly_random_tgtcontent_wb_20260302_134322`

Operational note:

- submit helper hit PBS env-size/hook limits for this rerun path.
- this chain was submitted via direct `qsub` with reduced `-v` keys while keeping recipe parity.

## 87. Diag-space alignment fix + rerun (`content_tokens`, 2026-03-02)

Reason:

- in Sections 85/86, pretrain loss target was switched to `content_tokens`, but copy diagnostics
  (`cos_tgt/cos_prev/gap/copy_win`) were still computed in `z` space.
- this makes `cos_prev` interpretation inconsistent with the active objective space.

Code fix:

- `nepa3d/train/pretrain_patch_nepa.py`
  - `_compute_copy_diag(...)` now receives `target_seq` and computes:
    - `tgt = target_seq[:, k:, :]`
    - `prev = target_seq[:, :-k, :]`
  - training loop now passes the same `target_seq` used in loss:
    - `full_z` -> `out.z`
    - `content_tokens` -> `out.tokens`

Action:

- old content-target runs were stopped for diagnostic comparison purposes:
  - `101051.qjcm`, `101062.qjcm` (and dependent held FT jobs).
- corrected rerun submitted:
  - pretrain `101091.qjcm`
  - run_set: `pt_q1tcwb4_fixdiag_20260302_135718`
  - run_tag: `r_q1tcwb4_fixdiag`
  - status at submission check: `R`

Recipe parity (kept same as Section 86):

- point-only: `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
- `USE_RAY_PATCH=0`, `N_RAY=0`, `PT_SAMPLE_MODE=random`
- `LOSS_TARGET_MODE=content_tokens`
- dual-mask: `near=0.5`, `far=0.1`, `window=32`, `type_aware=1`
- `USE_EMA=1`, `EMA_DECAY=0.9999`
- `EPOCHS=100`, `BATCH=8`, global batch `128`
- W&B on:
  - project `patchnepa-pretrain`
  - run `r_q1tcwb4_fixdiag`

Startup sanity from rank0 log (`qh004`):

- configuration is correctly applied:
  - `workdir=/groups/qgah50055/ide/VGI/3D-NEPA`
  - `loss_target_mode=content_tokens`
  - `pt_sample_mode=random`, `n_ray=0`, `use_ray_patch=0`
- W&B run URL:
  - `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/5emh8ppf`

Observed early-step diagnostic behavior (same run):

- `step 0`: `cos_tgt=-0.1111`, `cos_prev=-0.1110`
- `step 100`: `cos_tgt=0.9912`, `cos_prev=0.9812`, `gap=0.0099`

Paths:

- pretrain logs:
  - `logs/patch_nepa_pretrain/pt_q1tcwb4_fixdiag_20260302_135718`
- per-node rank0 log:
  - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101091.qjcm_r_q1tcwb4_fixdiag/logs/qh004.patchnepa.log`

## 88. Dual-mask on/off loss-curve note (2026-03-02)

Purpose:

- answer whether dual-mask presence changes the *early loss drop shape*.

Findings:

- in a matched stage2 sanity pair (same launcher family, same scheduler/seed family),
  early-step loss drop is nearly identical between dual-mask off/on.
- example (`mr0`, step-0 to step-100):
  - off (`lS_d0_ta1_tp0_r0`): `0.9750 -> 0.0048597`
  - on  (`lS_d1_ta1_tp0_r0`): `0.9750 -> 0.0048567`
- this indicates dual-mask is **not** the primary driver of the initial rapid convergence.

Secondary observation:

- later phase can diverge by run; in the same matched pair, dual-mask on trends slightly higher loss.
- with ray-enabled sibling pair (`..._r1`), the late-phase gap (on > off) is larger.
- interpretation: dual-mask effect is more visible in sustained training dynamics than in initial drop.

Important limitation:

- there is currently **no completed strict Q1-random point-only dual-mask-off control**.
- therefore, for Q1 random mainline (`100778` family), we cannot yet claim an apples-to-apples
  on/off conclusion; current statement is based on nearest matched sanity pairs.

Reference logs used for this note:

- point-only pair:
  - `logs/patch_nepa_pretrain/patchnepa_stage2_sanity32_20260301_162811_lS_d0_ta1_tp0_r0/run_patchnepa_stage2_sanity32_20260301_162811_lS_d0_ta1_tp0_r0.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_stage2_sanity32_20260301_162811_lS_d1_ta1_tp0_r0/run_patchnepa_stage2_sanity32_20260301_162811_lS_d1_ta1_tp0_r0.mr0.log`
- ray+point sibling pair:
  - `logs/patch_nepa_pretrain/patchnepa_stage2_sanity32_20260301_162811_lS_d0_ta1_tp0_r1/run_patchnepa_stage2_sanity32_20260301_162811_lS_d0_ta1_tp0_r1.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_stage2_sanity32_20260301_162811_lS_d1_ta1_tp0_r1/run_patchnepa_stage2_sanity32_20260301_162811_lS_d1_ta1_tp0_r1.mr0.log`

## 89. Chain result update (`101051` / `101052`, 2026-03-02)

Status (final, from `qstat -x`):

- `101051.qjcm` (`patchnepa_rayqa`): `F` (finished), `Exit_status=271`
- `101052.qjcm` (`pn_obj_only`): `F` (finished without run), dependent hold path

Pretrain (`101051`) summary:

- run set:
  - `logs/patch_nepa_pretrain/pt_q1tc_0302_133221`
- config highlights:
  - point-only (`USE_RAY_PATCH=0`, `N_RAY=0`)
  - `LOSS_TARGET_MODE=content_tokens`
  - `PT_SAMPLE_MODE=random`
  - dual-mask on (`near=0.5`, `far=0.1`, `window=32`, `type_aware=1`)
  - W&B off (`USE_WANDB=0`)
- runtime/result:
  - reached approximately `epoch 54`, `step 20200` before termination
  - last logged line (`mr0`):
    - `loss=5.0332e-04`, `loss2d=-9.99496678e-01`
    - `cos_tgt=0.4274`, `cos_prev=0.4120`, `gap=0.0154`, `copy_win=0.3223`
  - checkpoints saved up to `ckpt_epoch_0053.pt` + `ckpt_latest.pt`:
    - `runs/patchnepa_rayqa/pt_q1tc_0302_133221`
- operational note:
  - PBS history shows the job was terminated from submit host side (`Exit_status=271`), so this chain is not a clean `afterok` success completion.

Finetune dependent job (`101052`) summary:

- dependency:
  - `afterok:101051`
- observed state:
  - finished with hold/dependency path (`Hold_Types=s`, `Time_Use=0`)
  - no `.out/.err` payload produced in:
    - `logs/sanity/patchnepa_ft/patchnepaFT_objonly_from_q1_ptonly_random_tgtcontent_20260302_133330`
- interpretation:
  - `101051` not satisfying a clean `afterok` completion prevented `101052` from executing.

Primary references:

- pretrain launcher PBS log:
  - `logs/patch_nepa_pretrain/pt_q1tc_0302_133221/r_q1tc.pbs.log`
- pretrain rank0 log (metrics):
  - `logs/patch_nepa_pretrain/pt_q1tc_0302_133221/r_q1tc.mr0.log`
- pretrain artifacts:
  - `runs/patchnepa_rayqa/pt_q1tc_0302_133221`

## 90. FT all-variant relaunch from content-target ckpt (`101105`~`101107`, 2026-03-02)

Purpose:

- user-requested completion of FT for all three ScanObjectNN variants so results can be compared against prior 3-variant baselines (`100784`/`100781`/`100785` etc.).

Source checkpoint:

- `runs/patchnepa_rayqa/pt_q1tc_0302_133221/ckpt_latest.pt`
  - pretrain run corresponds to Section 89 (`LOSS_TARGET_MODE=content_tokens` branch).

Submitted FT jobs (independent, no `afterok`):

| variant | job | status at submit check |
|---|---|---|
| `obj_bg` | `101105.qjcm` (`pn_obj_bg`) | `R` |
| `obj_only` | `101106.qjcm` (`pn_obj_only`) | `R` |
| `pb_t50_rs` | `101107.qjcm` (`pn_pb_t50_rs`) | `R` |

Run set / paths:

- run set: `patchnepaFT_from_q1_tgtcontent_all3_20260302_1436`
- logs:
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_q1_tgtcontent_all3_20260302_1436`
- submitted job list:
  - `logs/sanity/patchnepa_ft/patchnepaFT_from_q1_tgtcontent_all3_20260302_1436/job_ids.txt`

FT recipe alignment (kept comparable to prior direct PatchNEPA FT):

- `MODEL_SOURCE=patchnepa`, `PATCHNEPA_FT_MODE=qa_zeroa`
- `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
- `PATCHNEPA_CLS_TOKEN_SOURCE=bos`, `PATCHNEPA_FREEZE_PATCH_EMBED=0`
- `CKPT_USE_EMA=1`
- point-only FT side: `USE_RAY_PATCH=0`, `N_RAY=0`
- strict eval: `VAL_SPLIT_MODE=file`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
- W&B on: `USE_WANDB=1`, project `patchnepa-finetune`, group `patchnepa-ft-contenttok`.

Operational note:

- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh` again hit
  `qsub: cannot send environment with the job`.
- workaround used: per-variant PBS wrapper scripts under the run-set log directory
  with env exports inside the script body (no large `-v` payload).

## 91. FT all-variant result check (completed) (`101105`~`101107`, 2026-03-02)

Status snapshot (`qstat -x`):

- `101105.qjcm` (`obj_bg`): `F`, `Exit_status=0`
- `101106.qjcm` (`obj_only`): `F`, `Exit_status=0`
- `101107.qjcm` (`pb_t50_rs`): `F`, `Exit_status=0`

Confirmed final metrics:

- `obj_bg` (`101105`):
  - `TEST acc=0.8279` (`obj_bg.out`)
- `obj_only` (`101106`):
  - `TEST acc=0.8417` (`obj_only.out`)
- `pb_t50_rs` (`101107`):
  - `TEST acc=0.7915` (`pb_t50_rs.out`)

Comparison vs prior Q1 point-only baseline (`100784/100781/100785` from Section 83):

- baseline `obj_bg=0.8158` -> now `0.8279` (`+0.0121`)
- baseline `obj_only=0.8176` -> now `0.8417` (`+0.0241`)
- baseline `pb_t50_rs=0.7724` -> now `0.7915` (`+0.0191`)

W&B runs for this set:

- `obj_bg`: `https://wandb.ai/ide_koh/patchnepa-finetune/runs/3ibchg5w`
- `obj_only`: `https://wandb.ai/ide_koh/patchnepa-finetune/runs/uyxwve25`
- `pb_t50_rs`: `https://wandb.ai/ide_koh/patchnepa-finetune/runs/coo51vsk`

## 92. Convergence-speed comparison vs Q1 baseline (2026-03-02)

Comparison target:

- baseline (Q1 point-only random branch, Section 83):
  - `obj_bg`: `100784` (`.../patchnepaFT_objbg_from_q1_ptonly_random_e100_20260302_053501/obj_bg.out`)
  - `obj_only`: `100781` (`.../patchnepaFT_objonly_from_q1_ptonly_random_e100_20260302_053501/obj_only.out`)
- current content-target branch:
  - `obj_bg`: `101105` (`.../patchnepaFT_from_q1_tgtcontent_all3_20260302_1436/obj_bg.out`)
  - `obj_only`: `101106` (`.../patchnepaFT_from_q1_tgtcontent_all3_20260302_1436/obj_only.out`)

Epoch-to-threshold (lower is faster):

| variant | run | t80 | t82 | t84 | t86 | best val (epoch) |
|---|---:|---:|---:|---:|---:|---:|
| `obj_bg` | baseline (`100784`) | 40 | 51 | 51 | 104 | `0.8744` (173) |
| `obj_bg` | content-target (`101105`) | 65 | 83 | 91 | 160 | `0.8744` (222) |
| `obj_only` | baseline (`100781`) | 46 | 59 | 98 | 147 | `0.8700` (203) |
| `obj_only` | content-target (`101106`) | 62 | 71 | 83 | 119 | `0.8924` (271) |

Readout:

- `obj_bg`: content-target branch is consistently slower to reach the same val-acc levels
  (`+25`/`+32`/`+40`/`+56` epochs for t80/t82/t84/t86), while final test is higher (`0.8279` vs `0.8158`).
- `obj_only`: early rise is slower at low thresholds (t80/t82), but overtakes at higher thresholds
  (t84/t86 faster by `15`/`28` epochs) and reaches a higher ceiling
  (`best val 0.8924`, `TEST 0.8417` vs baseline `0.8176`).

Walltime check (full 300-epoch completed jobs):

- `obj_bg`: baseline `00:10:49` (`100784`) vs content-target `00:10:41` (`101105`)
- `obj_only`: baseline `00:10:58` (`100781`) vs content-target `00:11:00` (`101106`)
- throughput is essentially unchanged; difference is in optimization trajectory, not runtime speed.

## 93. Pretrain convergence-speed comparison (`100778` vs `101091`) (2026-03-02)

Comparison pair (both completed, same Q1 point-only random base recipe):

- baseline (`full_z` target): `100778.qjcm`
  - log: `logs/patch_nepa_pretrain/patchnepa_q1_ptonly_random_e100_20260302_053501/run_q1_ptonly_random.mr0.log`
- content-target (`content_tokens`): `101091.qjcm`
  - log: `logs/patch_nepa_pretrain/pt_q1tcwb4_fixdiag_20260302_135718/r_q1tcwb4_fixdiag.mr0.log`

Scheduler/runtime parity:

- `100778`: walltime `00:38:37`, exit `0`
- `101091`: walltime `00:38:28`, exit `0`
- both end at global step `36750` (throughput difference is negligible).

Important caveat:

- objective target differs (`full_z` vs `content_tokens`), so absolute loss values are not strictly apples-to-apples.
- therefore we compare *shape/speed indicators* at matched steps and threshold crossings.

Loss threshold crossing (first step; lower is faster):

| metric | `100778` full_z | `101091` content_tokens |
|---|---:|---:|
| `loss <= 1e-2` | `100` | `100` |
| `loss <= 5e-3` | `250` | `150` |
| `loss <= 3e-3` | `6050` | `4300` |
| `loss <= 2e-3` | `6200` | `4950` |
| `loss <= 1e-3` | `6800` | `7600` |
| `loss <= 8e-4` | `7100` | `7900` |
| `loss <= 6e-4` | `7500` | `9450` |

Fixed-step loss snapshot:

| step | `100778` full_z | `101091` content_tokens |
|---:|---:|---:|
| `100` | `7.69e-03` | `8.85e-03` |
| `1,000` | `3.87e-03` | `4.59e-03` |
| `3,000` | `1.70e-02` | `1.17e-02` |
| `5,000` | `9.90e-03` | `2.49e-03` |
| `10,000` | `1.52e-03` | `4.31e-04` |
| `20,000` | `1.96e-03` | `4.26e-04` |
| `36,700` | `9.16e-04` | `6.39e-04` |

Readout (pretrain convergence):

- very early phase (`~0` to `1k` steps): baseline (`full_z`) is slightly lower loss.
- from mid phase (`~3k` onward): content-target run trends lower loss at matched steps.
- crossing of tighter thresholds (`<=1e-3` and below) is mixed:
  - first `1e-3` crossing is earlier in baseline,
  - but sustained mid/late-step loss is generally lower in content-target run.

## 94. Q-mask ablation launch (`q_mask_prob=0.6/0.8`) + launcher fix (2026-03-02)

Objective:

- run the suggested split-layout "Q-token masking" ablation against the same base recipe as `101091`
  (`content_tokens`, point-only, dual-mask on), changing only `q_mask_prob`.

Failed first attempt (recorded):

- `101153.qjcm` (`patchnepa_qmask6`) and `101155.qjcm` (`patchnepa_qmask8`) ended `F` quickly.
- failure mode:
  - `ModuleNotFoundError: No module named 'nepa3d'`
  - launch header showed wrong `workdir=/groups/qgah50055/ide/VGI/3D-NEPA/logs`
  - defaults were unintentionally used (`loss_target_mode=full_z`, `q_mask_prob=0.0`, `use_ray_patch=1`).
- root cause:
  - multinode wrapper executed a snapshot launch script under `RUN_DIR`; when `WORKDIR` was not exported
    into node-entry child env, the script resolved default workdir from its snapshot path (`../.. -> logs`).

Launcher hardening:

- updated:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
- fix:
  - node-entry now passes `WORKDIR` and `PYTHONPATH=${WORKDIR}:${PYTHONPATH:-}` explicitly to child launch.
- intent:
  - avoid snapshot-path-dependent module import failures in reduced-env submit paths.

Relaunch (corrected):

| setting | job | run_set | run_tag | status (launch check) |
|---|---|---|---|---|
| `q_mask_prob=0.6` | `101156.qjcm` | `pt_q1tc_qmask60_20260302_155835` | `r_q1tc_qmask60` | `R` |
| `q_mask_prob=0.8` | `101157.qjcm` | `pt_q1tc_qmask80_20260302_155835` | `r_q1tc_qmask80` | `R` |

Parity controls vs `101091` (kept fixed):

- `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
- `USE_RAY_PATCH=0`, `N_RAY=0`, `PT_SAMPLE_MODE=random`
- `LOSS_TARGET_MODE=content_tokens`
- dual-mask on: `near=0.5`, `far=0.1`, `window=32`, `type_aware=1`
- `USE_EMA=1`, `EMA_DECAY=0.9999`
- `EPOCHS=100`, `BATCH=8`, global batch `128`
- W&B on (`patchnepa-pretrain`).

Startup sanity (rank0):

- `101156` (`qh043`):
  - `q_mask_prob=0.600`, first step:
    - `qmask=(0.60,ramp=1.00,keep=0.4238)`
  - W&B:
    - `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/9iezgn16`
- `101157` (`qh167`):
  - `q_mask_prob=0.800`, first step:
    - `qmask=(0.80,ramp=1.00,keep=0.2207)`
  - W&B:
    - `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/190gxmkq`

Primary logs:

- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101156.qjcm_r_q1tc_qmask60/logs/qh043.patchnepa.log`
- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101157.qjcm_r_q1tc_qmask80/logs/qh167.patchnepa.log`

Follow-up issue found during this relaunch:

- non-rank0 workers hit:
  - `UnboundLocalError: local variable 'loss_val' referenced before assignment`
  - source: `nepa3d/train/pretrain_patch_nepa.py` (logging block scope at step print)
- affected jobs:
  - `101156.qjcm`, `101157.qjcm` (both ended `F`, `Exit_status=271`)

Fix:

- `nepa3d/train/pretrain_patch_nepa.py`
  - moved `diag/print/wandb` logging block fully under:
    - `if accelerator.is_main_process and (global_step % 50 == 0):`
  - this restores rank-local variable scope correctness and avoids non-main-process logging crashes.

Relaunch after print-scope fix:

| setting | job | run_set | run_tag | status (launch check) |
|---|---|---|---|---|
| `q_mask_prob=0.6` | `101160.qjcm` | `pt_q1tc_qmask60_v2_20260302_160325` | `r_q1tc_qmask60_v2` | `R` |
| `q_mask_prob=0.8` | `101161.qjcm` | `pt_q1tc_qmask80_v2_20260302_160325` | `r_q1tc_qmask80_v2` | `R` |

Startup sanity (`v2`, rank0):

- `101160`:
  - W&B: `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/wde12g97`
  - step0: `qmask=(0.60,ramp=1.00,keep=0.4238)`
  - step100: `loss=7.8058e-03`, `qmask keep=0.4004`
- `101161`:
  - W&B: `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/a9eq02vw`
  - step0: `qmask=(0.80,ramp=1.00,keep=0.2207)`
  - step100: `loss=7.8086e-03`, `qmask keep=0.1973`

Current active logs:

- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101160.qjcm_r_q1tc_qmask60_v2/logs/qh043.patchnepa.log`
- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101161.qjcm_r_q1tc_qmask80_v2/logs/qh129.patchnepa.log`

## 95. FT x6 submission from qmask-v2 pretrains (`101160`/`101161`) (2026-03-02)

Request:

- submit all 3 ScanObjectNN variants for both running qmask-v2 pretrains:
  - parent `101160` (`qmask=0.6`)
  - parent `101161` (`qmask=0.8`)
- total `3 x 2 = 6` jobs, all with dependency hold (`afterok`).

Submission method:

- direct `qsub -v` helper can hit env-size limits, so per-variant PBS wrapper scripts were generated under each run-set log directory.
- each child job runs:
  - `scripts/finetune/patchnepa_scanobjectnn_finetune.sh`
  - with recipe parity to Section 90/91 FT:
    - `MODEL_SOURCE=patchnepa`
    - `PATCHNEPA_FT_MODE=qa_zeroa`
    - `POOLING=cls_max`, `HEAD_MODE=pointmae_mlp`
    - `PATCHNEPA_CLS_TOKEN_SOURCE=bos`
    - `PATCHNEPA_FREEZE_PATCH_EMBED=0`
    - `USE_RAY_PATCH=0`, `N_RAY=0`
    - `VAL_SPLIT_MODE=file`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
    - `CKPT_USE_EMA=1`
    - `batch=64 (global)`, `epochs=300`, `nproc_per_node=4`.

Submitted jobs:

| parent pretrain | variant | FT job | status at submit check |
|---|---|---|---|
| `101160` | `obj_bg` | `101200.qjcm` | `H` (`afterok:101160`) |
| `101160` | `obj_only` | `101201.qjcm` | `H` (`afterok:101160`) |
| `101160` | `pb_t50_rs` | `101202.qjcm` | `H` (`afterok:101160`) |
| `101161` | `obj_bg` | `101203.qjcm` | `H` (`afterok:101161`) |
| `101161` | `obj_only` | `101204.qjcm` | `H` (`afterok:101161`) |
| `101161` | `pb_t50_rs` | `101205.qjcm` | `H` (`afterok:101161`) |

Run-set paths:

- `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask60v2_all3_20260302_161951`
  - job list: `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask60v2_all3_20260302_161951/job_ids.txt`
- `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask80v2_all3_20260302_161951`
  - job list: `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask80v2_all3_20260302_161951/job_ids.txt`

## 96. 2DNEPA pretrain failure cause + interface-fix relaunch (2026-03-02)

Observed failure:

- job: `101197.qjcm` (`nepa2d_bdef`) finished `F` (`Exit_status=1`).
- dependency preflight passed (`transformers/timm` was OK), but DDP init failed with NCCL bootstrap:
  - `Bootstrap : no socket interface found`
  - `DistBackendError ... ncclInvalidUsage`
- root cause: fixed interface pin in `scripts/pretrain/nepa_b.sh`:
  - `NCCL_SOCKET_IFNAME=eth0`
  - compute node did not provide a valid `eth0` route for NCCL bootstrap.

Fix:

- updated `scripts/pretrain/nepa_b.sh`:
  - removed hard-coded `NCCL_SOCKET_IFNAME=eth0`
  - added auto-pick logic from available interfaces (`ibp|ib|en|eth`, optional `PREFERRED_IFNAME`)
  - exports both `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME`
  - prints selected interface in startup log.

Relaunch:

- job: `101248.qjcm` (`nepa2d_bdef`)
- run tag: `nepa2d_bdef_ifauto_20260302_164309`
- status at check: `R`

Startup evidence (`nepa2d_bdef.o101248`):

- `[deps] preflight OK`
- `[nccl] socket_ifname=enp0s20f0u5u2`
- no `Bootstrap : no socket interface found` error
- distributed ranks reached trainer initialization normally.

## 97. 2DNEPA copy-diagnostic logging parity with PatchNEPA (2026-03-02)

Goal:

- make 2DNEPA pretrain logs/W&B expose the same copy diagnostics as PatchNEPA:
  - `diag/cos_tgt`
  - `diag/cos_prev`
  - `diag/gap`
  - `diag/copy_win`

Code changes:

- `run_nepa.py`:
  - `EnhancedTrainer.compute_loss` now computes and logs copy diagnostics from
    model internals every `diag_every` steps (`diag_copy=1` enabled).
  - added CLI args: `--diag_copy`, `--diag_every`, `--diag_k`.
- `models/vit_nepa/modeling_vit_nepa.py`:
  - `EmbeddedModelingOutput` now returns `sequence_input` / `sequence_output`
    so trainer can compute diagnostics without recomputation.
- launch scripts:
  - `scripts/pretrain/nepa_b.sh`, `scripts/pretrain/nepa_l.sh` pass
    `--diag_copy/--diag_every/--diag_k` to `run_nepa.py`.
  - `scripts/pretrain/nepa2d_pretrain_b_qf.sh` logs and exports
    `DIAG_COPY/DIAG_EVERY/DIAG_K`.

Expected log line example:

- `[diag-copy] step=... kdiag=1 cos_tgt=... cos_prev=... gap=... copy_win=...`

## 98. 2DNEPA re-submit with copy diagnostics (2026-03-02)

Request:

- re-submit 2DNEPA pretrain after enabling PatchNEPA-parity diagnostics.

Submissions:

- `101257.qjcm` (`nepa2d_bdiag`) failed immediately (`Exit_status=1`).
  - stdout: `mkdir: cannot create directory '/var/spool/pbs/logs': Permission denied`
  - root cause: `WORKDIR` env was not passed at submit, so script resolved default
    to a PBS spool path and attempted to create logs there.
- `101258.qjcm` (`nepa2d_bdiag`) re-submitted with explicit `WORKDIR` and dataset vars.
  - status at check: `R`
  - run tag: `nepa2d_bdiagcopy_20260302_172631`
  - log: `logs/pretrain/nepa2d/nepa2d_bdiagcopy_20260302_172631.log`
  - startup confirmed:
    - `diag_copy=1 diag_every=50 diag_k=1`
    - `[deps] preflight OK`
    - NCCL interface auto-pick active.

## 99. FT x6 completed (`101200`-`101205`) and comparison vs baseline (2026-03-02)

Completion status:

- all 6 child FT jobs finished successfully (`Exit_status=0`):
  - `101200`, `101201`, `101202`, `101203`, `101204`, `101205`
- last two (`pb_t50_rs`) completed at:
  - `101202`: `obittime=2026-03-02 17:29:27`
  - `101205`: `obittime=2026-03-02 17:30:02`

Final TEST metrics:

Baseline for comparison (`q1_tgtcontent_all3`, Section 93):
- `obj_bg=0.8279`, `obj_only=0.8417`, `pb_t50_rs=0.7915`

`qmask60v2` parent (`101160`) -> run dir:
- `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask60v2_all3_20260302_161951`
- `obj_bg`: `TEST acc=0.8193` (delta vs baseline: `-0.0086`)
- `obj_only`: `TEST acc=0.8348` (delta vs baseline: `-0.0069`)
- `pb_t50_rs`: `TEST acc=0.7977` (delta vs baseline: `+0.0062`)

`qmask80v2` parent (`101161`) -> run dir:
- `logs/sanity/patchnepa_ft/patchnepaFT_from_q1tc_qmask80v2_all3_20260302_161951`
- `obj_bg`: `TEST acc=0.8262` (delta vs baseline: `-0.0017`)
- `obj_only`: `TEST acc=0.8176` (delta vs baseline: `-0.0241`)
- `pb_t50_rs`: `TEST acc=0.7845` (delta vs baseline: `-0.0070`)

Quick read:

- `qmask60v2` is mostly neutral/slightly worse, but improved on `pb_t50_rs`.
- `qmask80v2` underperformed baseline on all three variants, with the largest drop on `obj_only`.

## 100. 2x2 content-token run: merged-log root cause, kill, and clean re-submit (2026-03-02)

Observed issue:

- the two pretrain jobs from Section 2x2 submission (`101289`, `101290`) wrote into the same default run/log:
  - `logs/patch_nepa_pretrain/patchnepa_rayqa_20260302_183441.mr0.log`
- node-local logs showed defaults were used (not wrapper values), e.g.:
  - `run_tag=patchnepa_rayqa_20260302_183441`
  - `loss_target_mode=full_z`
  - `use_ray_patch=1`, `n_ray=1024`

Root cause:

- in `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`, per-node `pbsdsh` launch path did not reliably propagate wrapper env vars (`RUN_TAG`, `SAVE_DIR`, `PT_SAMPLE_MODE`, `LOSS_TARGET_MODE`, etc.).
- node entry therefore fell back to launch-script defaults.

Fix applied:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - added explicit `PROPAGATE_VARS` list and persisted values into `${RUN_DIR}/env.conf`
  - node entry now uses `set -a; source env.conf; set +a` to export all persisted vars.

Operational action:

- killed wrong chain:
  - pretrain: `101289`, `101290`
  - dependent FT holds: `101291`..`101296`
- re-submitted pretrain:
  - `101297.qjcm` (`pn_rfpsc`, `pt_sample_mode=rfps_cached`)
  - `101298.qjcm` (`pn_rfps`, `pt_sample_mode=rfps`)
- re-submitted dependent FT x6 (`afterok`):
  - `101299` (`rfpsc`, `obj_bg`)
  - `101300` (`rfps`, `obj_bg`)
  - `101301` (`rfpsc`, `obj_only`)
  - `101302` (`rfps`, `obj_only`)
  - `101303` (`rfpsc`, `pb_t50_rs`)
  - `101304` (`rfps`, `pb_t50_rs`)

Verification (node logs, startup):

- `101297` (`qh076.patchnepa.log`):
  - `run_tag=r_patchnepa_content_2x2_20260302_183413_rfpsc`
  - `save_dir=runs/patchnepa_rayqa/patchnepa_content_2x2_20260302_183413_pt_rfpsc`
  - `pt_sample_mode=rfps_cached`
  - `loss_target_mode=content_tokens`
- `101298` (`qh164.patchnepa.log`):
  - `run_tag=r_patchnepa_content_2x2_20260302_183413_rfps`
  - `save_dir=runs/patchnepa_rayqa/patchnepa_content_2x2_20260302_183413_pt_rfps`
  - `pt_sample_mode=rfps`
  - `loss_target_mode=content_tokens`

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/submitted_jobs_resub_20260302_184319.txt`

## 101. Patch-order diversity support (`fps_knn`) + sweep comparison submit (2026-03-02)

Issue raised:

- `point_order_mode=morton` reorders sampled points, but in `patch_embed=fps_knn` the
  patch-token sequence order remained the FPS center-selection order.
- no patch-level reorder hook existed in `PatchTransformerNepa.forward()`.

Code changes:

- `nepa3d/models/patch_nepa.py`
  - added `patch_order_mode` to `PatchTransformerNepa`:
    - `native` (default, old behavior)
    - `morton_xyz`, `morton_xyz_rev`
    - `morton_yzx`, `morton_yzx_rev`
    - `morton_zxy`, `morton_zxy_rev`
    - `random_sweep`, `random_sweep_rev`
  - implemented `_build_patch_perm(centers_xyz)` and `_apply_patch_perm(...)`.
  - applies patch permutation right after patch embedding (reorders `q_tok`,
    `centers_xyz`, `group_idx` coherently).
  - q-only classifier path now applies the same core patch-order logic.
- `nepa3d/train/pretrain_patch_nepa.py`
  - added CLI arg `--patch_order_mode` and passed it into `PatchTransformerNepa`.
  - startup summary now prints `patch_order_mode`.
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh`
  - added env `PATCH_ORDER_MODE` (default `native`), log print, and CLI forwarding.
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh`
  - added `PATCH_ORDER_MODE` to propagated env list.

Sanity:

- local forward sanity (venv) confirmed for modes:
  - `native`, `morton_xyz`, `morton_yzx_rev`, `random_sweep`, `random_sweep_rev`
  - output tensor shapes matched baseline.

Comparison submission (best recipe + patch-order sweep):

- submit root:
  - `logs/sanity/patchnepa_submit/patchnepa_patchorder_sweepcmp_20260302_195955`
- pretrain:
  - `101312.qjcm` (`pn_patchord_swpt`)
  - recipe parity with best content-target point-only run:
    - `loss_target_mode=content_tokens`
    - `pt_sample_mode=random`
    - `qa_tokens=1`, `qa_layout=split_sep`
    - `use_ray_patch=0`, `n_ray=0`
    - only delta: `PATCH_ORDER_MODE=random_sweep`
- dependent FT (afterok:101312):
  - `101313.qjcm` (`obj_bg`)
  - `101314.qjcm` (`obj_only`)
  - `101315.qjcm` (`pb_t50_rs`)

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_patchorder_sweepcmp_20260302_195955/submitted_jobs.txt`

## 102. 2x2 launch: SEP presence x patch-order mode (2026-03-02)

Goal:

- jointly test two axes under otherwise identical recipe:
  - axis A (SEP): `qa_layout=split` + `qa_sep_token=0` vs `qa_layout=split_sep` + `qa_sep_token=1`
  - axis B (patch-order): `patch_order_mode=morton` vs `patch_order_mode=random_sweep`

Design (2x2):

- `lS_poM`: `qa_layout=split`, `qa_sep_token=0`, `patch_order_mode=morton`
- `lS_poR`: `qa_layout=split`, `qa_sep_token=0`, `patch_order_mode=random_sweep`
- `lSS_poM`: `qa_layout=split_sep`, `qa_sep_token=1`, `patch_order_mode=morton`
- `lSS_poR`: `qa_layout=split_sep`, `qa_sep_token=1`, `patch_order_mode=random_sweep`

Common pretrain recipe:

- `loss_target_mode=content_tokens`
- `pt_sample_mode=random`
- `point_order_mode=morton`
- `patch_embed=fps_knn`
- `use_ray_patch=0`
- `dual_mask=(near=0.5, far=0.1, window=32, type_aware=1)`
- `epochs=100`, `batch=8/proc`, DDP16

Submission root:

- `logs/sanity/patchnepa_submit/patchnepa_sep_patchorder_2x2_20260302_203259`

Submitted jobs:

- pretrain:
  - `101318.qjcm` (`lS_poM`)
  - `101322.qjcm` (`lS_poR`)
  - `101326.qjcm` (`lSS_poM`)
  - `101330.qjcm` (`lSS_poR`)
- dependent FT x 12 (`obj_bg`, `obj_only`, `pb_t50_rs` per arm):
  - from `101318`: `101319`, `101320`, `101321`
  - from `101322`: `101323`, `101324`, `101325`
  - from `101326`: `101327`, `101328`, `101329`
  - from `101330`: `101331`, `101332`, `101333`

Runtime status at submit:

- pretrains `R`, dependent FTs `H` (afterok dependency)

Record file:

- `logs/sanity/patchnepa_submit/patchnepa_sep_patchorder_2x2_20260302_203259/submitted_jobs.txt`

### 102.1 Added FT extension: `pooling=sep` on split-sep arms (2026-03-02)

Reason:

- to explicitly evaluate SEP-token pooling effect in the same 2x2 context.
- `pooling=sep` is valid only when sequence contains `TYPE_SEP`.

Scope:

- added only on split-sep arms:
  - `lSS_poM` (depends on pretrain `101326`)
  - `lSS_poR` (depends on pretrain `101330`)
- variants: `obj_bg`, `obj_only`, `pb_t50_rs` (total +6 FT jobs)

Submitted pool-sep FT jobs:

- from `101326` (`lSS_poM`):
  - `101335` (`obj_bg`)
  - `101336` (`obj_only`)
  - `101337` (`pb_t50_rs`)
- from `101330` (`lSS_poR`):
  - `101338` (`obj_bg`)
  - `101339` (`obj_only`)
  - `101340` (`pb_t50_rs`)

Wrapper diff note:

- `POOLING=sep`
- `PATCHNEPA_FT_MODE=qa_zeroa` (kept)
- `HEAD_MODE=linear` (explicit for sep pooling stability/consistency)

Record:

- appended into `logs/sanity/patchnepa_submit/patchnepa_sep_patchorder_2x2_20260302_203259/submitted_jobs.txt`

## 103. `content_tokens` 2x2 resub completion (`101297`-`101304`) (2026-03-02)

Completion status:

- pretrain:
  - `101297` (`rfps_cached`) -> `F / Exit_status=0` (completed)
  - `101298` (`rfps`) -> `F / Exit_status=0` (completed)
- dependent FT x6:
  - `101299`..`101304` all `F / Exit_status=0` (completed)

Pretrain end snapshot (from `.mr0.log` run summary):

- `rfps_cached` (`101297`):
  - `diag/cos_prev=0.97214`
  - `diag/cos_tgt=0.99932`
  - `diag/gap=0.02717`
  - `diag/copy_win=0.04297`
- `rfps` (`101298`):
  - `diag/cos_prev=0.97154`
  - `diag/cos_tgt=0.99924`
  - `diag/gap=0.02770`
  - `diag/copy_win=0.04883`

FT final TEST metrics:

- `rfps_cached` branch:
  - `101299` (`obj_bg`): `TEST acc=0.8193` (`loss=0.7996`)
  - `101301` (`obj_only`): `TEST acc=0.8348` (`loss=0.7227`)
  - `101303` (`pb_t50_rs`): `TEST acc=0.7665` (`loss=1.2742`)
- `rfps` branch:
  - `101300` (`obj_bg`): `TEST acc=0.8262` (`loss=0.7542`)
  - `101302` (`obj_only`): `TEST acc=0.8417` (`loss=0.6907`)
  - `101304` (`pb_t50_rs`): `TEST acc=0.7845` (`loss=1.2720`)

Result read:

- this resub confirms the same ranking trend as earlier:
  - `rfps` > `rfps_cached` on all three downstream variants in this setting.
- current top in this pair remains:
  - `obj_only=0.8417` (`101302`, `rfps`)

Source logs:

- pretrain summaries:
  - `logs/patch_nepa_pretrain/patchnepa_content_2x2_20260302_183413_pt_rfpsc/r_patchnepa_content_2x2_20260302_183413_rfpsc.mr0.log`
  - `logs/patch_nepa_pretrain/patchnepa_content_2x2_20260302_183413_pt_rfps/r_patchnepa_content_2x2_20260302_183413_rfps.mr0.log`
- FT qsub logs:
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfpsc_obj_bg_resub.out`
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfps_obj_bg_resub.out`
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfpsc_obj_only_resub.out`
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfps_obj_only_resub.out`
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfpsc_pb_t50_rs_resub.out`
  - `logs/sanity/patchnepa_submit/patchnepa_content_2x2_20260302_183413/qsub_logs/ft_rfps_pb_t50_rs_resub.out`

## 104. Patch-order sweep pretrain completed (`101312`), dependent FT running (`101313`-`101315`) (2026-03-02)

Pretrain status:

- `101312` (`patch_order_mode=random_sweep`) -> `F / Exit_status=0` (completed)

Pretrain end snapshot:

- `diag/cos_prev=0.9713`
- `diag/cos_tgt=0.99968`
- `diag/gap=0.02838`
- `diag/copy_win=0.03516`
- checkpoint:
  - `runs/patchnepa_rayqa/patchnepa_patchorder_sweepcmp_20260302_195955_pt_sweep`

Dependent FT status (currently running):

- `101313` (`obj_bg`): `R`
- `101314` (`obj_only`): `R`
- `101315` (`pb_t50_rs`): `R`

Source logs:

- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101312.qjcm_r_patchnepa_patchorder_sweepcmp_20260302_195955_sweep/logs/qh137.patchnepa.log`
- `logs/sanity/patchnepa_submit/patchnepa_patchorder_sweepcmp_20260302_195955/qsub_logs/pretrain_sweep.out`

## 105. `sep x patch_order` 2x2: split(without SEP) arms failed by policy gate (2026-03-02)

Observed:

- `101318` (`lS_poM`) -> `F / Exit_status=97`
- `101322` (`lS_poR`) -> `F / Exit_status=97`
- immediate error in node logs:
  - `ERROR: Stage-2 policy requires QA_SEP_TOKEN=1 (got 0)`

Impact on dependent jobs:

- from `101318`: `101319`/`101320`/`101321` -> `F` (dependency chain failed)
- from `101322`: `101323`/`101324`/`101325` -> `F` (dependency chain failed)

Current unaffected split-sep arms:

- pretrains still running:
  - `101326` (`lSS_poM`) -> `R`
  - `101330` (`lSS_poR`) -> `R`
- their dependent FT jobs (`101327`..`101333`, `101335`..`101340`) remain `H` (`afterok` wait).

Source logs:

- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101318.qjcm_r_patchnepa_sep_patchorder_2x2_20260302_203259_lS_poM/logs/qh141.patchnepa.log`
- `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_101322.qjcm_r_patchnepa_sep_patchorder_2x2_20260302_203259_lS_poR/logs/qh002.patchnepa.log`

## 106. Submitted: `pt_sample_mode x patch_order_mode` 2x2 (`random` vs `rfps`) (2026-03-02)

Reason:

- explicit request to validate:
  - `pt_sample_mode=rfps` + `patch_order_mode=morton`
  - against matched `random` arms under the same recipe.
- previous patch-order sweep jobs were `PT_SAMPLE_MODE=random` only.

Design (2x2):

- `r_poM`: `pt_sample_mode=random`, `patch_order_mode=morton`
- `r_poR`: `pt_sample_mode=random`, `patch_order_mode=random_sweep`
- `f_poM`: `pt_sample_mode=rfps`, `patch_order_mode=morton`
- `f_poR`: `pt_sample_mode=rfps`, `patch_order_mode=random_sweep`

Common controls:

- `loss_target_mode=content_tokens`
- `qa_layout=split_sep`, `qa_sep_token=1` (policy-compliant)
- `use_ray_patch=0`, `n_ray=0`
- `point_order_mode=morton`, `patch_embed=fps_knn`
- `epochs=100`, DDP16 (`rt_QF=4`, `batch=8/proc`)
- dependent FT x3 per arm (`obj_bg`, `obj_only`, `pb_t50_rs`) with `afterok`.

Submitted jobs:

- pretrain:
  - `101348` (`r_poM`)
  - `101352` (`r_poR`)
  - `101356` (`f_poM`)
  - `101360` (`f_poR`)
- dependent FT:
  - from `101348`: `101349`, `101350`, `101351`
  - from `101352`: `101353`, `101354`, `101355`
  - from `101356`: `101357`, `101358`, `101359`
  - from `101360`: `101361`, `101362`, `101363`

Queue snapshot right after submit:

- pretrains: `R`
- dependent FTs: `H` (`afterok` wait)

Record:

- root:
  - `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_20260302_205849`
- submit script:
  - `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_20260302_205849/submit.sh`
- job list:
  - `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_20260302_205849/submitted_jobs.txt`

## 107. Confirmed results update (`sweep` + `sep x patch_order`) and 2D-NEPA stop request (2026-03-02)

Confirmed final FT metrics from available `test_metrics.json`:

- `patch_order_sweepcmp` FT (`101313`-`101315` lineage):
  - `obj_bg`: `acc=0.8365` (`loss=0.7865`)
  - `obj_only`: `acc=0.8262` (`loss=0.7935`)
  - `pb_t50_rs`: `acc=0.8064` (`loss=1.1250`)
- `sep x patch_order` (`split_sep`, `pooling=cls_max`) `lSS_poM`:
  - `obj_bg`: `acc=0.8090` (`loss=0.8239`)
  - `obj_only`: `acc=0.8158` (`loss=0.8616`)
  - `pb_t50_rs`: `acc=0.7738` (`loss=1.2827`)
- `sep x patch_order` (`split_sep`, `pooling=cls_max`) `lSS_poR`:
  - `obj_bg`: `acc=0.8365` (`loss=0.7865`)
  - `obj_only`: `acc=0.8262` (`loss=0.7935`)
  - `pb_t50_rs`: `acc=0.8064` (`loss=1.1250`)
- `sep x patch_order` + `pooling=sep` `lSS_poM_poolsep`:
  - `obj_bg`: `acc=0.7642` (`loss=1.0268`)
  - `obj_only`: `acc=0.7780` (`loss=1.2165`)
  - `pb_t50_rs`: `acc=0.7491` (`loss=1.5934`)
- `sep x patch_order` + `pooling=sep` `lSS_poR_poolsep`:
  - `obj_bg`: `acc=0.8055` (`loss=0.9226`)
  - `obj_only`: `acc=0.8176` (`loss=0.9576`)
  - `pb_t50_rs`: `acc=0.7793` (`loss=1.3367`)

Readout:

- In this recipe, `patch_order_mode=poR` outperforms `poM` across all three variants.
- `pooling=sep` underperforms `pooling=cls_max` for both `poM` and `poR`.
- Overall best remains unchanged: `obj_only=0.8417` (content-target `rfps` branch, prior run).

Job control:

- `101258.qjcm` (`nepa2d_bdiag`) received delete request and is now removed from current queue listing.

## 108. `pt_sample_mode x patch_order_mode` 2x2 (`101348`-`101363`) final status: failed before training (2026-03-02)

Summary:

- All jobs are now finished (`qstat` empty), but this sweep did **not** produce valid experiment metrics.
- Root cause was pretrain rendezvous failure on all 4 pretrain arms:
  - `101348` (`r_poM`), `101352` (`r_poR`), `101356` (`f_poM`), `101360` (`f_poR`)
  - all `job_state=F`, `Exit_status=1`
  - common error:
    - `torch.distributed.DistStoreError: Timed out after 901 seconds waiting for clients. 1/4 clients joined.`
  - source:
    - `logs/patch_nepa_pretrain/patchnepa_rfps_patchorder_2x2_20260302_205849_pt_*/r_*.mr0.log`

Impact on dependent FT jobs:

- `101349`..`101351`, `101353`..`101355`, `101357`..`101359`, `101361`..`101363`
- all ended as `job_state=F` with dependency hold/finalized state (not successful FT execution)
- therefore no `test_metrics.json` generated for this sweep.

Notes:

- W&B was enabled (`use=1`, `project=patchnepa-pretrain`) in all four pretrain wrappers.
- this batch should be treated as infrastructure failure (multi-node rendezvous), not a model comparison result.

## 109. Re-submit of `pt_sample_mode x patch_order_mode` 2x2 with single-node DDP (`101420`-`101435`) (2026-03-02)

Reason:

- retry the exact 2x2 comparison after `DistStoreError` in multi-node launch.
- launcher changed to `nepa3d_pretrain_patch_nepa_qf.sh` with explicit single-node DDP:
  - `NUM_PROCESSES=4`, `NUM_MACHINES=1`, `MACHINE_RANK=0`, `MAIN_PROCESS_IP=127.0.0.1`

Design (same 2x2 factors):

- `r_poM`: `pt_sample_mode=random`, `patch_order_mode=morton`
- `r_poR`: `pt_sample_mode=random`, `patch_order_mode=random_sweep`
- `f_poM`: `pt_sample_mode=rfps`, `patch_order_mode=morton`
- `f_poR`: `pt_sample_mode=rfps`, `patch_order_mode=random_sweep`

Common controls:

- `loss_target_mode=content_tokens`
- `qa_layout=split_sep`, `qa_sep_token=1`
- `use_ray_patch=0`, `n_ray=0`
- `point_order_mode=morton`, `patch_embed=fps_knn`
- effective global batch kept at 128 (`batch=8`, `grad_accum=4`, `num_processes=4`)
- W&B on: `patchnepa-pretrain` / `patchnepa-finetune`

Submitted jobs:

- pretrain:
  - `101420` (`r_poM`)
  - `101424` (`r_poR`)
  - `101428` (`f_poM`)
  - `101432` (`f_poR`)
- dependent FT:
  - from `101420`: `101421`, `101422`, `101423`
  - from `101424`: `101425`, `101426`, `101427`
  - from `101428`: `101429`, `101430`, `101431`
  - from `101432`: `101433`, `101434`, `101435`

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_retry_20260302_223332/submitted_jobs.txt`
- `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_retry_20260302_223332/submit.sh`

Outcome update:

- this retry also failed immediately on all 4 pretrains (`101420`,`101424`,`101428`,`101432`), each `job_state=F`, `Exit_status=1`, `walltime=00:00:01`.
- direct error:
  - `tee: logs/patch_nepa_pretrain/... .mr0.log: No such file or directory`
- root cause:
  - `nepa3d_pretrain_patch_nepa_qf.sh` creates `LOG_ROOT` before `cd ${WORKDIR}`.
  - wrapper had relative `LOG_ROOT` / `SAVE_DIR`, so runtime `tee` path was unresolved after `cd`.
  - fix is to pass absolute paths.

## 110. Re-re-submit with absolute `LOG_ROOT`/`SAVE_DIR` (`101436`-`101451`) (2026-03-02)

Fix applied:

- keep same 2x2 experiment design and hyper-parameters.
- wrapper changed to absolute paths:
  - `SAVE_DIR=/groups/qgah50055/ide/VGI/3D-NEPA/runs/...`
  - `LOG_ROOT=/groups/qgah50055/ide/VGI/3D-NEPA/logs/...`
- launcher remains single-node DDP (`NUM_PROCESSES=4`, `NUM_MACHINES=1`).

Submitted jobs:

- pretrain:
  - `101436` (`r_poM`: random + morton)
  - `101440` (`r_poR`: random + random_sweep)
  - `101444` (`f_poM`: rfps + morton)
  - `101448` (`f_poR`: rfps + random_sweep)
- dependent FT:
  - from `101436`: `101437`, `101438`, `101439`
  - from `101440`: `101441`, `101442`, `101443`
  - from `101444`: `101445`, `101446`, `101447`
  - from `101448`: `101449`, `101450`, `101451`

Startup sanity (confirmed in `pretrain_r_poM.out`):

- run header is emitted normally (no missing log-path error).
- W&B line present:
  - `wandb: use=1 project=patchnepa-pretrain ...`

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950/submit.sh`
- `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950/submitted_jobs.txt`

## 111. Best-recipe backbone swap submit (`pointmae_conv + fps_random_start=1`) (2026-03-03)

Purpose:

- keep the current best PatchNEPA recipe family fixed (`content_tokens + rfps + point-only + split_sep`),
- swap only local patch backbone path to Point-MAE-style:
  - `PATCH_LOCAL_ENCODER=pointmae_conv`
  - `PATCH_FPS_RANDOM_START=1`

Recipe controls (kept aligned with best branch):

- pretrain:
  - `MIX_CONFIG=pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
  - `pt_sample_mode=rfps`, `loss_target_mode=content_tokens`
  - `patch_embed=fps_knn`, `patch_order_mode=none`
  - `use_ray_patch=0`, `n_ray=0`
  - single-node DDP4 (`NUM_PROCESSES=4`, `GRAD_ACCUM=4`, effective global batch 128)
- finetune:
  - direct PatchNEPA FT (`model_source=patchnepa`)
  - `qa_zeroa`, `pooling=cls_max`, `head=pointmae_mlp`, strict eval (`file + TTA10`)
  - variants: `obj_bg`, `obj_only`, `pb_t50_rs`

Submitted jobs:

- pretrain:
  - `101468.qjcm` (`pnpmcv_pt`)
- dependent FT:
  - `101469.qjcm` (`pnpmcv_obj_bg`, `afterok:101468`)
  - `101470.qjcm` (`pnpmcv_obj_only`, `afterok:101468`)
  - `101471.qjcm` (`pnpmcv_pb_t50_rs`, `afterok:101468`)

Queue snapshot right after submit:

- `101468`: `R`
- `101469`/`101470`/`101471`: `H` (dependency wait)

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_backbone_pmconv_frombest_20260303_000550/submit.sh`
- `logs/sanity/patchnepa_submit/patchnepa_backbone_pmconv_frombest_20260303_000550/submitted_jobs.txt`

## 112. Local-encoder strict control arm submitted (`mlp`, same recipe as 111) (2026-03-03)

Reason:

- to complete strict A/B where only local patch encoder differs.
- arm 111 (`101468`) used:
  - `PATCH_LOCAL_ENCODER=pointmae_conv`
  - `PATCH_FPS_RANDOM_START=1`
- this control arm uses:
  - `PATCH_LOCAL_ENCODER=mlp`
  - `PATCH_FPS_RANDOM_START=1`
- all other settings are matched to section 111 (`content_tokens + rfps + point-only + split_sep`).

Submitted jobs:

- pretrain:
  - `101472.qjcm` (`pnmlpc_pt`)
- dependent FT:
  - `101473.qjcm` (`pnmlpc_obj_bg`, `afterok:101472`)
  - `101474.qjcm` (`pnmlpc_obj_only`, `afterok:101472`)
  - `101475.qjcm` (`pnmlpc_pb_t50_rs`, `afterok:101472`)

Queue snapshot right after submit:

- `101472`: `R`
- `101473`/`101474`/`101475`: `H` (dependency wait)

Submission record:

- `logs/sanity/patchnepa_submit/patchnepa_backbone_localenc_ctrl_20260303_000834/submit.sh`
- `logs/sanity/patchnepa_submit/patchnepa_backbone_localenc_ctrl_20260303_000834/submitted_jobs.txt`

## TODO. Point-MAE-parity path handling policy (2026-03-03)

Decision note:

- `backbone_mode=pointmae` / Point-MAE-parity path is currently treated as an experimental control route only.
- It should not be used as the main PatchNEPA recipe for headline conclusions, because it drifts from PatchNEPA's original design intent.

TODO (owner: user-side follow-up):

- review and refine/rollback Point-MAE-parity wiring as needed.
- keep core comparisons and mainline reporting on PatchNEPA-native settings.

## 112. Point-order sample diversity compare submit (6-view vs 12-view, point-order only) (2026-03-03)

User request:

- keep `serial_order` axis out of scope for this compare.
- compare only `point_order_mode` diversity against existing best-recipe family.
- two requested settings:
  - 6-view Morton axis permutations
  - 12-view (6-view + reverse variants)

Comparator baseline (existing best in this recipe family):

- from Section 103 (`content_tokens`, point-only, `pt_sample_mode=rfps`, `point_order_mode=morton`)
  - `obj_bg=0.8262`
  - `obj_only=0.8417`
  - `pb_t50_rs=0.7845`

Design submitted here:

- fixed recipe (same family as baseline):
  - pretrain: `content_tokens`, point-only, `pt_sample_mode=rfps`, `use_ema=1`
  - `patch_embed=fps_knn`
  - `patch_order_mode=none` (held fixed; not a comparison axis)
  - `qa_layout=split_sep`, `dual_mask=(0.5,0.1,w=32,type_aware=1)`
  - single-node DDP4 with `grad_accum=4` (effective global batch 128)
- only changed axis: `point_order_mode`

Arms:

1) `po6`
- `point_order_mode=sample:morton_xyz,morton_xzy,morton_yxz,morton_yzx,morton_zxy,morton_zyx`

2) `po12`
- `point_order_mode=sample:morton_xyz,morton_xzy,morton_yxz,morton_yzx,morton_zxy,morton_zyx,rev_morton_xyz,rev_morton_xzy,rev_morton_yxz,rev_morton_yzx,rev_morton_zxy,rev_morton_zyx`

Submitted jobs:

- pretrain:
  - `101484.qjcm` (`po6`)
  - `101488.qjcm` (`po12`)
- dependent FT (`afterok`, direct PatchNEPA FT, 3 variants each):
  - from `101484`: `101485` (`obj_bg`), `101486` (`obj_only`), `101487` (`pb_t50_rs`)
  - from `101488`: `101489` (`obj_bg`), `101490` (`obj_only`), `101491` (`pb_t50_rs`)

Submission root:

- `logs/sanity/patchnepa_submit/patchnepa_pointorder_samplecmp_20260303_011204`
  - `submit.sh`
  - `submitted_jobs.txt`
  - `qsub_logs/`

Runtime snapshot right after submit:

- pretrain `101484`/`101488`: `R`
- dependent FT `101485`-`101487`, `101489`-`101491`: `H` (`afterok` wait)

## 113. `101444`-`101451` completion audit (2026-03-03)

Checked jobs:

- pretrain: `101444` (`f_poM`), `101448` (`f_poR`)
- dependent FT: `101445`-`101447`, `101449`-`101451`

Final status:

- `101444`: `job_state=F`, `Exit_status=2`
- `101448`: `job_state=F`, `Exit_status=2`
- `101445`-`101447`, `101449`-`101451`: `job_state=F`, `Hold_Types=s` (dependency unsatisfied, not executed)

Important nuance:

- Both pretrain logs reached epoch end and wrote checkpoints:
  - `runs/patchnepa_rayqa/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950_pt_f_poM/ckpt_latest.pt`
  - `runs/patchnepa_rayqa/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950_pt_f_poR/ckpt_latest.pt`
- W&B summaries were also emitted:
  - `f_poM`: `diag/cos_prev=0.97037`, `diag/gap=0.02968`, `copy_win=0.05859`
  - `f_poR`: `diag/cos_prev=0.97682`, `diag/gap=0.02250`, `copy_win=0.07617`

Failure cause recorded in stderr (`pretrain_f_poM.err`, `pretrain_f_poR.err`):

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh: line 420: unexpected EOF while looking for matching \"`

Impact:

- This batch produced valid pretrain checkpoints but no dependent FT results.
- No new `obj_bg/obj_only/pb_t50_rs` test metrics were generated for jobs `101445`-`101451`.

## 114. FT-only relaunch from recovered checkpoints (`101509`-`101514`) (2026-03-03)

Why relaunch:

- `101444`/`101448` pretrains produced valid `ckpt_latest.pt`, but `Exit_status=2` blocked all `afterok` FT jobs.
- To avoid re-running expensive pretrain, FT was relaunched directly from those recovered checkpoints.

Relaunch root:

- `logs/sanity/patchnepa_submit/patchnepa_rfps_patchorder_2x2_retry2_ftonly_relaunch_20260303_033658`
  - `submitted_jobs.txt`
  - `qsub_logs/`

Submitted FT jobs (no dependency hold):

- from `f_poM` checkpoint:
  - `101509.qjcm` (`obj_bg`)
  - `101510.qjcm` (`obj_only`)
  - `101511.qjcm` (`pb_t50_rs`)
- from `f_poR` checkpoint:
  - `101512.qjcm` (`obj_bg`)
  - `101513.qjcm` (`obj_only`)
  - `101514.qjcm` (`pb_t50_rs`)

Checkpoint sources:

- `runs/patchnepa_rayqa/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950_pt_f_poM/ckpt_latest.pt`
- `runs/patchnepa_rayqa/patchnepa_rfps_patchorder_2x2_retry2_20260302_224950_pt_f_poR/ckpt_latest.pt`

## 115. Data-v2 token pipeline integration note (`ans_feat` fixed-dim vs additive) (2026-03-03)

Decision recorded:

- The "fixed `ans_feat` dimension" requirement and "additive modality embedding" are not contradictory.
- In the current pipeline, fixed dim is mandatory for batch collation (`[B,Nq,C]`), so `answer_in_dim` was introduced as the model-side contract.
- Additive modality embedding remains a next-step model improvement and should be implemented on top of the same fixed-schema data contract.

Applied code baseline in this repo:

- `nepa3d/models/patch_nepa.py`
  - `forward_tokens()` token stream path
  - `answer_in_dim` support (for v2 packed answer features)
- `nepa3d/data/mixed_pretrain.py`
  - `dataset_version=v2` routing to `V2SurfaceQueryDataset`
- New v2 stack files:
  - `nepa3d/data/answer_feature_pack.py`
  - `nepa3d/data/dataset_v2.py`
  - `nepa3d/data/preprocess_shapenet_v2.py`
  - `nepa3d/data/convert_npz_to_v2.py`
  - `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens.yaml`

Operational rule:

- Keep fixed-schema v2 training as mainline bring-up.
- Add additive embedding as an isolated ablation/upgrade after data-v2 run is stable.

## 116. ShapeNet v2 rebuild (multi-shard qsub) started (`101714`-`101721`) (2026-03-03)

Objective:

- Rebuild ShapeNet cache with v2 schema using parallel shard jobs (multi-node via multi-job sharding).

Code/runtime updates applied before submit:

- `nepa3d/data/preprocess_shapenet_v2.py`
  - Added `--num_shards/--shard_id` shard filtering.
  - Added `--skip_existing` for restart-safe reruns.
  - Changed manifest output for shard runs to:
    - `v2_manifest.shard{shard_id}of{num_shards}.json`
    - (prevents overwrite races in concurrent shard runs).
- Added launch wrappers:
  - `scripts/preprocess/preprocess_shapenet_v2.sh`
  - `scripts/preprocess/submit_preprocess_shapenet_v2_qf.sh`

Submitted command profile:

- `OUT_ROOT=data/shapenet_cache_v2_20260303`
- `NUM_SHARDS=8`
- `WORKERS=24` (per shard)
- `RT_QF=1`, `WALLTIME=72:00:00`
- logs root:
  - `logs/preprocess/shapenet_v2/shapenet_v2_rebuild_20260303_151147`

Submitted jobs:

- `101714.qjcm` (`shpv2_s00`)
- `101715.qjcm` (`shpv2_s01`)
- `101716.qjcm` (`shpv2_s02`)
- `101717.qjcm` (`shpv2_s03`)
- `101718.qjcm` (`shpv2_s04`)
- `101719.qjcm` (`shpv2_s05`)
- `101720.qjcm` (`shpv2_s06`)
- `101721.qjcm` (`shpv2_s07`)

Initial status at submit check:

- all 8 jobs in `R` state on `abciq`.

## 117. Auto dependent submit for unpaired split/materialize (`101730`,`101731`) (2026-03-03)

Requested operation:

- Automatically run unpaired split/materialize only after all ShapeNet v2 shard jobs complete successfully.

Logic validation and fixes made:

- Existing helper scripts had hard-coded legacy paths/project assumptions (`/groups/gag...`) and were updated to dynamic `ROOT_DIR` + `.venv` activation.
- Added dependency submit wrapper:
  - `scripts/preprocess/submit_shapenet_unpaired_post_qf.sh`
  - Inputs:
    - `PREPROC_JOB_IDS` (required)
    - `CACHE_ROOT`, `OUT_JSON`, `OUT_ROOT`, `RATIOS`, `SPLITS`, etc.
  - Behavior:
    1. submit split job with `depend=afterok:<all preprocess shard jobs>`
    2. submit materialize job with `depend=afterok:<split job>`
- Corrected PBS `-v` transport edge case:
  - values with spaces/commas break environment passing;
  - switched to `:`-delimited strings:
    - `RATIOS=0.34:0.33:0.33`
    - `SPLITS=train_mesh:train_pc:train_udf:eval`
  - runner scripts convert `:` -> whitespace before Python argparse.

Submitted jobs:

- `101730.qjcm` (`shpv2_split`)
  - `depend=afterok:101714:101715:101716:101717:101718:101719:101720:101721`
- `101731.qjcm` (`shpv2_mat`)
  - `depend=afterok:101730`

Status at submit check:

- `101730`, `101731` are `H` (`Hold_Types=s`) as expected until dependencies are satisfied.
- submit logs:
  - `logs/preprocess/shapenet_unpaired/shapenet_v2_post_20260303_151731/`

## 118. Data-v2 revalidation scope reset (user-locked) (2026-03-03)

This runlog now follows the restart matrix in:

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` (Section 13)

Key lock-ins applied:

1. New ShapeNet v2 cache only.
2. `loss_target_mode=content_tokens` only.
3. ScanObjectNN FT protocol fixed to Point-MAE-style:
- `n_point=2048`
- `PointcloudScaleAndTranslate` augmentation for `obj_bg/obj_only/pb_t50_rs/hardest`.
 - patch local encoder default `pointmae_conv`, with `patch_fps_random_start=1`.
4. Main comparison axes for this round:
- `qa_layout`: `split_sep` vs `interleave`
- `point_order_mode`: `morton` vs `random` vs `sample6` vs `sample12`
- `pt_sample_mode`: `rfps` vs `random` (non-cached)

De-prioritized in this round:

- `full_z` target arm.
- patch-order schedule (`epoch/batch/sample`) sweep.
- ModelNet FT.

## 119. Data-distribution arm definition (v2) (2026-03-03)

Distribution baseline decision:

- Baseline is `pc-only (100%)` on rebuilt ShapeNet v2.

Additional required composition arms:

1. `mesh+udf (50/50, no pc)`
2. `pc+mesh+udf (33/33/33)`

Implementation notes:

- Composition is encoded by split cardinality/materialized cache, not by runtime sampler weights.
- For these runs, use `replacement=false` one-pass sampling.
- `shapenet_unpaired_split.py` was updated with:
  - `--allow_empty_splits {0,1}`
  - strict composition runs use `--allow_empty_splits 1` to permit true zero-allocation splits.

## 120. ShapeNet v2 missing-only backfill + forced post-process (2026-03-03)

Context:

- After 16-shard rebuild/backfill, generation entered long-tail on a small set of meshes.
- To avoid waiting for the last hard cases, remaining preprocess jobs were stopped and post-process was advanced.

Code/tooling update:

- Added `--missing_only` to:
  - `nepa3d/data/preprocess_shapenet_v2.py`
  - `scripts/preprocess/preprocess_shapenet_v2.sh`
  - `scripts/preprocess/submit_preprocess_shapenet_v2_qf.sh`
- Purpose: rerun only currently missing NPZ outputs with standard shard parallelism and logs.

Missing-only submit:

- run tag:
  - `shapenet_v2_missingonly_20260303_183342`
- jobs:
  - `101933`..`101948` (`shpv2m16_s00`..`shpv2m16_s15`)
- logs:
  - `logs/preprocess/shapenet_v2/shapenet_v2_missingonly_20260303_183342/`
- startup summary from shard logs:
  - `total_missing_now=106`, shard tasks were `7` or `6`.

Forced stop decision:

- Remaining running long-tail jobs were force-stopped:
  - `101934,101935,101936,101937,101939,101941,101942,101944`
- cache count at stop time:
  - `train=47212`, `test=5246`, `total=52458` (vs expected `52472`, missing `14`).

Post-process submit (split/materialize):

- First attempt (failed due wrong `PBS_O_WORKDIR` selecting `/groups/.../VGI`):
  - split: `102025` (`shpv2_split_now`) failed with
    - `[error] python not found: /groups/qgah50055/ide/VGI/.venv/bin/python`
  - materialize dependent job did not run.

- Retry from project root (successful):
  - split: `102028` (`shpv2_split_now2`)
  - materialize: `102029` (`shpv2_mat_now2`, depend=afterok:102028)
  - logs:
    - `logs/preprocess/shapenet_unpaired/shapenet_v2_post_forced_retry_20260303_191513/`

Final outputs:

- split json:
  - `data/shapenet_unpaired_splits_v2_20260303.json`
- split counts (from split log):
  - `train_mesh=16050`, `train_pc=15579`, `train_udf=15583`, `eval=5246`
- materialize summary (symlink mode, overwrite on):
  - `created=52458`, `skipped=0`, `missing=0`
- materialized cache root:
  - `data/shapenet_unpaired_cache_v2_20260303`

## 121. Data-v2 sanity check + point-only token pretrain start (2026-03-03)

Preflight sanity check on materialized cache:

- target root:
  - `data/shapenet_unpaired_cache_v2_20260303`
- split counts:
  - `train_mesh=16050`
  - `train_pc=15579`
  - `train_udf=15583`
  - `eval=5246`
- sampled NPZ checks:
  - required keys present for each split (`pc_qry_*`, `mesh_qry_*`, `udf_qry_*`)
  - `np.isfinite` passed on sampled tensors (no NaN/Inf in sampled files)

Point-only first-run (smoke) submission:

- job: `102038.qjcm` (`pntok_pc100_s1`)
- run set:
  - `patchnepa_v2tok_pc100_smoke_20260303_193224`
- run tag:
  - `pt_pc100_smoke_20260303_193224`
- save dir:
  - `runs/patchnepa_tokens/patchnepa_v2tok_pc100_smoke_20260303_193224`
- logs:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_v2tok_pc100_smoke_20260303_193224/pt_pc100_smoke_20260303_193224.pbs.log`

Recipe snapshot:

- mix config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_pc100.yaml`
- `token_qa_layout=split_sep`
- `max_steps=3000`, `batch=8`, `n_surf=2048`, `n_qry=1024`
- Point-MAE compatibility enabled (`pc_norm=1`, `scale_translate=1`)

W&B status:

- enabled (`use_wandb=1`)
- project: `patchnepa-pretrain`
- group: `v2tok_pc100_smoke`
- run URL (from log):
  - `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/dakvcpz2`

## 122. Composition smoke bring-up (mesh+udf / 3mix) (2026-03-03)

Initial submit and failure:

- first submit jobs:
  - `102039` (`pntok_m50u50_s1`)
  - `102040` (`pntok_mix33_s1`)
- failure cause (both):
  - config cache roots did not exist yet:
    - `data/shapenet_unpaired_cache_v2_mesh50_udf50`
    - `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33`
  - error in logs:
    - `FileNotFoundError: no npz found: cache_root=...`

Recovery action:

1. Created strict-ratio split/materialized roots from v2 base cache.
- source cache:
  - `data/shapenet_cache_v2_20260303`
- mesh+udf split json:
  - `data/shapenet_unpaired_splits_v2_mesh50_udf50_20260303.json`
  - counts: `train_mesh=23605`, `train_udf=23607`, `eval=5246`
- 3mix split json:
  - `data/shapenet_unpaired_splits_v2_pc33_mesh33_udf33_20260303.json`
  - counts: `train_pc=15736`, `train_mesh=15736`, `train_udf=15740`, `eval=5246`
- materialized roots (symlink, overwrite):
  - `data/shapenet_unpaired_cache_v2_mesh50_udf50` (`created=52458`, `missing=0`)
  - `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33` (`created=52458`, `missing=0`)

2. Resubmitted composition smokes.
- `102041` (`pntok_m50u50_s1`)
  - run set: `patchnepa_v2tok_mesh50udf50_smoke_20260303_193742`
  - W&B run: `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/kb1687a3`
- `102042` (`pntok_mix33_s1`)
  - run set: `patchnepa_v2tok_pc33mesh33udf33_smoke_20260303_193742`
  - W&B run: `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/g58gjkxr`

Current status:

- `102038` (`pc100`), `102041` (`mesh50+udf50`), `102042` (`pc33+mesh33+udf33`) are all running.
- startup checks passed in all active logs:
  - `answer_in_dim=9`
  - expected `mix_info` sizes/counts
  - W&B online sync
  - training loss decreasing from step 1.

## 123. Always-on diag logging for token pretrain (2026-03-03)

Request:

- Ensure copy-probe diagnostics are always emitted in token-pretrain runs:
  - `diag/gap`
  - `diag/cos_tgt`
  - `diag/cos_prev` (plus typo alias `diag/cos_precv`)
  - `diag/copy_win` (plus alias `diag/copy-win`)

Applied code changes:

1. `nepa3d/train/pretrain_patch_nepa_tokens.py`
- Added copy diagnostics computation (`cos_tgt`, `cos_prev`, `gap`, `copy_win`) using the same mask semantics as pretrain_patch_nepa.
- Added always-on periodic stdout diagnostic line:
  - `[step ...] ... cos_tgt=... cos_prev=... gap=... copy_win=...`
- Added W&B logging for diagnostic keys each log interval.
- Added CLI arg:
  - `--diag_every` (default `50`)

2. Launcher wiring:
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - env `DIAG_EVERY` (default `50`)
  - forwards `--diag_every`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - passes `DIAG_EVERY` via `qsub -v`

Validation run:

- job: `102043` (`pntok_diagchk`)
- run set:
  - `patchnepa_v2tok_diagcheck_20260303_194438`
- observed in log:
  - step diagnostics printed at 20-step interval:
    - step 20/40/60/80/100/120
    - includes `cos_tgt`, `cos_prev`, `gap`, `copy_win`
- W&B summary confirms metric keys:
  - `diag/cos_tgt`
  - `diag/cos_prev`
  - `diag/cos_precv`
  - `diag/gap`
  - `diag/copy_win`
  - `diag/copy-win`

Note:

- Runs started before this patch do not backfill these keys; they need restart/re-submit to carry `diag/*`.

Re-submitted smoke jobs with diag-enabled code:

- `102044` (`pntok_pc100_d`)
  - `patchnepa_v2tok_pc100_smoke_diag_20260303_194716`
- `102045` (`pntok_mix33_d`)
  - `patchnepa_v2tok_mix33_smoke_diag_20260303_194716`
- `102046` (`pntok_m50u50_d`)
  - `patchnepa_v2tok_mesh50udf50_smoke_diag_20260303_194716`

## 119. Strict Surf-Answer Dataset Regeneration (Append, UDF-Grid Sphere Tracing, No-Ray) (2026-03-03)

Purpose:

- Move to strict surf-aligned Answer features for queryless v1-style PatchNEPA pretrain.
- Avoid full cache rewrite by appending required keys into existing v2 NPZs.

Applied code paths:

- `nepa3d/data/preprocess_shapenet_v2.py`
  - New surf-aligned keys:
    - `mesh_surf_n`, `mesh_surf_curv`
    - `udf_surf_t_in`, `udf_surf_t_out`, `udf_surf_hit_out`, `udf_surf_thickness`
    - `pc_n`, `pc_density`
  - Added append/update mode (`--augment_existing`) with skip-if-already-enriched.
  - Strict UDF-grid controls:
    - `--strict_udf_surface`
    - `--surf_udf_grid`, `--surf_udf_dilate`
    - `--surf_udf_max_t`, `--surf_udf_eps`
    - `--surf_udf_steps`, `--surf_udf_tol`, `--surf_udf_min_step`
  - UDF strict features are produced by sphere tracing on occupancy-derived UDF grid.
  - `n_rays=0` safe path.
- `nepa3d/data/dataset_v2.py`
  - Added `pt_xyz` / `pt_ans_feat` aligned output path.
- `nepa3d/data/mixed_pretrain.py`
  - Added `return_pt_ans`, `pt_answer_prefix`, `pt_answer_key` routing.
- preprocess launcher wiring:
  - `scripts/preprocess/preprocess_shapenet_v2.sh`
  - `scripts/preprocess/submit_preprocess_shapenet_v2_qf.sh`

Submitted jobs (active chain):

- preprocess (16 shards, append to `data/shapenet_cache_v2_20260303`, no-ray):
  - `102118` `102119` `102120` `102121` `102122` `102123` `102124` `102125`
  - `102126` `102127` `102128` `102129` `102130` `102131` `102132` `102133`
- dependent post-process:
  - split job: `102134` (`afterok` on all preprocess shards)
  - materialize job: `102135` (`afterok:102134`)
  - outputs:
    - split json: `data/shapenet_unpaired_splits_v2_20260303_strictgrid.json`
    - materialized root: `data/shapenet_unpaired_cache_v2_20260303_strictgrid`

## 124. v2 token pretrain (drop1, morton, split_sep) + short FT result capture (2026-03-04)

Completed pretrain runs (`max_steps=10000`, `loss_mask_mode=answer_and_point_context`):

- run set:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_drop1_20260304_001831/`
- `pc100` (`tok_drop1_pc100.log`)
  - final step: `loss=0.753113`, `cos_tgt=0.4521`, `cos_prev=0.4539`, `gap=-0.0018`, `copy_win=0.4965`
  - last100 mean: `loss=0.746067`, `cos_tgt=0.4663`, `cos_prev=0.4682`, `gap=-0.0019`, `copy_win=0.5016`
- `mesh50udf50` (`tok_drop1_mesh50udf50.log`)
  - final step: `loss=0.514475`, `cos_tgt=0.9418`, `cos_prev=0.9411`, `gap=+0.0007`, `copy_win=0.5006`
  - last100 mean: `loss=0.515127`, `cos_tgt=0.9410`, `cos_prev=0.9403`, `gap=+0.0007`, `copy_win=0.5078`
- `pc33mesh33udf33` (`tok_drop1_pc33m33u33.log`)
  - final step: `loss=0.556022`, `cos_tgt=0.8577`, `cos_prev=0.8573`, `gap=+0.0005`, `copy_win=0.5035`
  - last100 mean: `loss=0.567859`, `cos_tgt=0.8333`, `cos_prev=0.8333`, `gap=+0.0001`, `copy_win=0.5046`

Short FT results (obj_only, Point-MAE aug on, 60 epochs):

- run set:
  - `logs/sanity/patchnepa_ft/patchnepa_tokens_drop1_ftshort_20260304_002808/`
- `tokpc100.out`: `TEST acc=0.8365`
- `tokm50u50.out`: `TEST acc=0.8365`
- `tokp33.out`: `TEST acc=0.8365`

Note:

- The three FT logs are line-by-line identical in training trace and final score.
- This block is treated as a reproducibility/control signal; it is not used as evidence of composition gain.

## 125. mask+primitive fix rerun started (morton, split_sep) (2026-03-04)

Code changes applied in `nepa3d/train/pretrain_patch_nepa_tokens.py`:

- `diag` mask now follows the same `loss_mask_mode` logic as training loss.
- tokens route now assigns primitive-specific point type IDs (`mesh`/`udf`/`pc`) per sample.

Submitted runs:

- `102506` (`tok_m100f`) using `shapenet_unpaired_mix_v2_tokens_drop1_mesh100.yaml`
- `102507` (`tok_u100f`) using `shapenet_unpaired_mix_v2_tokens_drop1_udf100.yaml`
- `102508` (`tok_m50f`) using `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- run set:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_drop1_tokens_maskprimfix_morton_20260304_014828/`

Current status at logging time:

- all three are `R` and reached around step `~2000` (not final yet).

## 126. Cross-run diag trend / spike comparison (unequal-step aware) (2026-03-04)

Scope:

- compared by same diag family: `loss`, `cos_tgt`, `cos_prev`, `gap`, `copy_win`
- windows are index-based (`early/mid/late` = first/center/last 10% of available diag records)
- some runs are partial or running; treated as interim trend only

### A) v2 tokens branch (morton/sample6 + single-primitive checks)

| run | coverage | loss trend (early->mid->late) | cos_tgt_late | cos_prev_late | gap_late | copy_win_late | max loss spike |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tok_drop1_pc100.log` | step `1..10000` | `0.6061 -> 0.7293 -> 0.7463` | `0.4659` | `0.4684` | `-0.0025` | `0.5020` | `0.0828` @`3836->3837` |
| `tok_drop1_mesh50udf50.log` | step `1..10000` | `0.5505 -> 0.5161 -> 0.5151` | `0.9412` | `0.9404` | `+0.0008` | `0.5067` | `0.0255` @`401->402` |
| `tok_drop1_pc33m33u33.log` | step `1..10000` | `0.5565 -> 0.5665 -> 0.5687` | `0.8316` | `0.8315` | `+0.0001` | `0.5049` | `0.0707` @`6704->6705` |
| `tok_drop1_mesh100.log` *(partial)* | step `1..7391` | `0.5513 -> 0.5700 -> 0.5735` | `0.8218` | `0.8227` | `-0.0009` | `0.5137` | `0.0593` @`7332->7333` |
| `tok_drop1_udf100.log` *(partial)* | step `1..7408` | `0.5473 -> 0.5090 -> 0.5588` | `0.8519` | `0.8517` | `+0.0002` | `0.5010` | `0.0327` @`7391->7392` |
| `tok_m50u50_sample6.log` *(partial; patch-order sample6)* | step `1..6103` | `0.5691 -> 0.5174 -> 0.5182` | `0.9364` | `0.9359` | `+0.0005` | `0.5079` | `0.0309` @`338->339` |

### B) v2 mask+primitive-fix rerun (running; early diagnosis only)

| run | coverage | loss trend (early->mid->late) | cos_tgt_late | cos_prev_late | gap_late | copy_win_late | max loss spike |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tok_m100f.log` | step `1..4217` | `0.5753 -> 0.5581 -> 0.5670` | `0.4330` | `0.4336` | `-0.0006` | `0.7536` | `0.0430` @`2700->2701` |
| `tok_u100f.log` | step `1..4211` | `0.5744 -> 0.5371 -> 0.5842` | `0.4158` | `0.4161` | `-0.0002` | `0.7466` | `0.0355` @`3361->3362` |
| `tok_m50f.log` | step `1..4193` | `0.5835 -> 0.5761 -> 0.6090` | `0.3910` | `0.3919` | `-0.0009` | `0.7503` | `0.0724` @`3848->3849` |

### C) historical v1 reference (content-target family)

| run | coverage | loss trend (early->mid->late) | cos_tgt_late | cos_prev_late | gap_late | copy_win_late |
|---|---:|---:|---:|---:|---:|---:|
| `...po6.mr0.log` | step `0..107900` | `0.0191 -> 0.0007 -> 0.0006` | `0.9994` | `0.9699` | `+0.0296` | `0.0654` |
| `...po12.mr0.log` | step `0..107900` | `0.0182 -> 0.0014 -> 0.0003` | `0.9997` | `0.9738` | `+0.0259` | `0.0800` |
| `...content_2x2...rfps.mr0.log` | step `0..36700` | `0.0390 -> 0.0008 -> 0.0008` | `0.9992` | `0.9719` | `+0.0273` | `0.0563` |

### D) direct interpretation from above

- Current v2 tokens branch: `cos_tgt` and `cos_prev` are almost identical in late phase (`gap` around `-0.003 .. +0.001`, `copy_win` around `0.50`).
  - This matches the concern: effective lift over prev-copy is near zero.
- Mask+primitive-fix rerun (current early phase): the overlap remains and `copy_win` is even higher (`~0.75` around step `~4k`).
  - At current depth this still looks copy-dominated.
- Historical v1 reference is qualitatively different: stable positive gap (`~+0.026..+0.030`) and low `copy_win` (`~0.06..0.08`).
  - So the current v2 token behavior is not just a scale shift; the diagnostic regime itself differs.
- `m50u50` morton vs `m50u50` sample6 (axis-swapped set) are close at current horizon.
  - Sample6 does not yet show a decisive separation in `gap/copy_win` trend.

### E) patch-order 2x2 note (historical jobs)

- `f_poM` and `r_poR` multi-node attempts failed at rendezvous (`DistStoreError`, timeout before effective train progression), so excluded from trend comparison.
- `f_poR` single-node retry completed, and W&B summary reports:
  - `diag/cos_tgt=0.99932`, `diag/cos_prev=0.97682`, `diag/gap=0.0225`, `diag/copy_win=0.07617`
- This completed `f_poR` profile is consistent with the v1 reference family (non-zero positive gap, low copy_win), and inconsistent with current v2 token profiles.

## 127. patch_order morton -> random switch (all running jobs terminated and relaunched) (2026-03-04)

Request handling:

- terminated currently running `morton` mask+primitive-fix jobs:
  - `102506` (`tok_m100f`), `102507` (`tok_u100f`), `102508` (`tok_m50f`)
  - final state: `job_state=F`, `Exit_status=271` (user-terminated)

Code/default changes applied for tokens pretrain path:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - argparse default `--patch_order_mode`: `morton -> random`
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - `PATCH_ORDER_MODE` default: `morton -> random`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - `PATCH_ORDER_MODE` fallback in `qsub -v`: `morton -> random`

Relaunched random-order runs (same drop1/mask+primitive-fix family):

- run set:
  - `patchnepa_tokens_drop1_tokens_maskprimfix_random_20260304_021637`
- jobs:
  - `102512` (`tok_m100r`) with `shapenet_unpaired_mix_v2_tokens_drop1_mesh100.yaml`
  - `102513` (`tok_u100r`) with `shapenet_unpaired_mix_v2_tokens_drop1_udf100.yaml`
  - `102514` (`tok_m50r`) with `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`

Submission/runtime parameters confirmed in `qstat -fx`:

- `PATCH_ORDER_MODE=random`
- `TOKEN_QA_LAYOUT=split_sep`
- `ANSWER_IN_DIM=9`
- `LOSS_TARGET_MODE=content_tokens`
- `LOSS_MASK_MODE=answer_and_point_context`
- W&B group: `drop1_tokens_maskprimfix_random`

## 128. `content_plus_center` target added + relaunch (`mesh50+udf50`, fps) (2026-03-04)

Request handling:

- user requested to stop all currently running token-pretrain jobs before next trial.
- terminated jobs:
  - `102512` (`tok_m100r`)
  - `102513` (`tok_u100r`)
  - `102514` (`tok_m50r`)
  - `102515` (`tok_m50fps`)
- queue state after termination check: no running jobs remained, then new run submitted.

Code changes applied (tokens pretrain path):

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - added new loss target mode: `content_plus_center`
  - added CLI arg: `--center_target_alpha` (default `0.5`)
  - target construction logic:
    - `full_z` -> `out.z`
    - `content_tokens` -> `out.tokens`
    - `content_plus_center` -> `out.tokens + alpha * center_mlp(out.centers_xyz)`
  - startup summary now logs:
    - `loss_target_mode`
    - `center_target_alpha`

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - added env var passthrough:
    - `CENTER_TARGET_ALPHA` (default `0.5`)
  - logs now include `center_target_alpha`
  - forwards `--center_target_alpha` to python entrypoint

- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - added `CENTER_TARGET_ALPHA` to `qsub -v` environment list.

Validation:

- python syntax check passed:
  - `python -m py_compile nepa3d/train/pretrain_patch_nepa_tokens.py`
- shell syntax checks passed:
  - `bash -n scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - `bash -n scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`

Submitted run (`mesh50+udf50`, `fps`, `content_plus_center`):

- job: `102516` (`tok_m50fps_cpc`)
- run set:
  - `patchnepa_tokens_drop1_m50u50_fps_cpc_20260304_023729`
- config:
  - `MIX_CONFIG=nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
  - `PATCH_ORDER_MODE=fps`
  - `TOKEN_QA_LAYOUT=split_sep`
  - `ANSWER_IN_DIM=9`
  - `LOSS_TARGET_MODE=content_plus_center`
  - `CENTER_TARGET_ALPHA=0.5`
  - `LOSS_MASK_MODE=answer_and_point_context`
- logs:
  - pbs: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_drop1_m50u50_fps_cpc_20260304_023729/tok_m50fps_cpc.pbs.log`

Initial online signal (very early, not final):

- step `1`: `cos_tgt=-0.0177`, `cos_prev=-0.0195`, `gap=+0.0017`, `copy_win=0.4769`
- step `20`: `cos_tgt=0.0430`, `cos_prev=0.0418`, `gap=+0.0011`, `copy_win=0.4824`
- unlike the prior maskprimfix runs (`copy_win~0.74-0.75`), this run starts near `~0.48-0.50` band.

## 129. Latent objective branch recap (`mesh50+udf50`) (2026-03-04 04:00 JST)

This block consolidates the latest latent-diagnostic runs before reconstruction-objective wiring.

| job | run/log | step | late loss | late cos_tgt | late cos_prev | late gap | late copy_win | note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `102515` | `tok_m50fps.log` | 1394 | `0.5307` | `0.4693` | `0.4719` | `-0.0025` | `0.7538` | copy-dominant |
| `102516` | `tok_m50fps_cpc.log` | 3805 | `0.1904` | `0.8096` | `0.8111` | `-0.0015` | `0.5044` | overlap remains |
| `102517` | `mesh50udf50_fps_splitsep_varcov.log` | 1860 | `0.1275` | `0.8752` | `0.8800` | `-0.0048` | `0.5181` | terminated |
| `102518` | `mesh50udf50_fps_splitsep_varcov_intra.log` | 10000 | `0.4229` | `0.5788` | `0.5788` | `-0.0000` | `0.5009` | completed |
| `102519` | `mesh50udf50_fps_splitsep_varcov_intra_dualcol.log` | 8735* | `0.4224` | `0.5793` | `0.5793` | `-0.0000` | `0.5006` | running* |
| `102521` | `mesh50udf50_fps_splitsep_varcov_intra_hreg.log` | 7365* | `0.3765` | `0.6249` | `0.6111` | `+0.0138` | `0.4739` | best gap so far* |
| `102522` | `mesh50udf50_fps_splitsep_hreg_dcol_nf03.log` | 3970* | `0.4519` | `0.5509` | `0.5521` | `-0.0011` | `0.5026` | running* |

`*` running at snapshot time.

Interpretation:

- Order-only and target-only adjustments still tended to `cos_tgt ≈ cos_prev` (`gap≈0`).
- Hidden-source regularization (`102521`) is currently the only branch showing sustained positive gap and sub-0.5 copy-win.
- Dual-column mask on top of hidden-reg (`102522`) has not yet preserved that gain.

## 130. Reconstruction objective smoke (`recon_mse`) submitted (2026-03-04)

Request: wire-check reconstruction objective path before optional chamfer.

Submitted:

- `102523.qjcm` (`pntok_rms`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_smoke_20260304_040234/`
- key settings:
  - `PRETRAIN_OBJECTIVE=recon_mse`
  - `MIX_CONFIG=...drop1_mesh50_udf50.yaml`
  - `PATCH_ORDER_MODE=morton`
  - `MAX_STEPS=2000` (smoke)

Startup confirmation from pbs log:

- objective banner shows `pretrain_objective=recon_mse recon_w=(ctx=1.0,q=1.0,a=1.0)`.

## 131. Recheck: `recon_mse` wiring and full latent snapshot (2026-03-04 04:50 JST)

`qstat` status at recheck:

- `102521` (`pntok_hreg`) running
- `102522` (`pntok_dcol3`) running
- `102523` (`pntok_rms`) running

`102523` (`recon_mse`) runtime check:

- pbs log:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_smoke_20260304_040234/recon_m50u50_smoke.pbs.log`
- observed progress: step `954 / 2000`
- no `Traceback`/`RuntimeError`/`NaN` in log
- reconstruction channels confirmed active (`loss_recon_ctx`, `loss_recon_q`, `loss_recon_a` all present each step)

All latent diagnostic runs re-indexed into one TSV for full comparison:

- `nepa3d/docs/patch_nepa/latent_diag_snapshot_20260304.tsv`

## 132. Full matrix launch from best latent recipe (`morton` fixed) (2026-03-04 04:28 JST)

Selected pretrain recipe (from latent branch trend):

- `PRETRAIN_OBJECTIVE=nepa_cosine`
- `LOSS_TARGET_MODE=content_plus_center`
- `CENTER_TARGET_ALPHA=0.5`
- `LOSS_MASK_MODE=answer_and_point_context`
- `REG_VAR_WEIGHT=0.05`
- `REG_COV_WEIGHT=0.01`
- `REG_SCOPE=intra_shape`
- `REG_SOURCE=hidden`
- `TOKEN_QA_LAYOUT=split_sep`
- `PATCH_ORDER_MODE=morton`
- `ANSWER_IN_DIM=9`
- `PATCH_LOCAL_ENCODER=pointmae_conv`
- `PATCH_FPS_RANDOM_START=1`
- `MAX_STEPS=10000`

Pretrain x3:

- `102527.qjcm` (`ptkpc`) `pc100`
- `102531.qjcm` (`ptkmu`) `mesh50_udf50`
- `102532.qjcm` (`ptkpm`) `pc33_mesh33_udf33`

FT x9 (`afterok` chained):

- from `102527`: `102528` (`obj_bg`), `102529` (`obj_only`), `102530` (`pb_t50_rs`)
- from `102531`: `102533` (`obj_bg`), `102534` (`obj_only`), `102535` (`pb_t50_rs`)
- from `102532`: `102536` (`obj_bg`), `102537` (`obj_only`), `102538` (`pb_t50_rs`)

Submission note:

- `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh` failed on this host with
  `qsub: cannot send environment with the job`.
- FT jobs were submitted via direct `qsub` to the same run script with minimal env payload.
- job list record:
  `logs/sanity/patchnepa_submit/patchnepa_tokens_fullbest_morton_20260304_042155/submitted_jobs.txt`

## 133. FT chain switched to short protocol (`EPOCHS=60`) for quick TEST check (2026-03-04 04:37 JST)

Pretrain jobs unchanged:

- `102527` (`pc100`)
- `102531` (`mesh50_udf50`)
- `102532` (`pc33_mesh33_udf33`)

Old queued FT jobs were cancelled and replaced by short FT (`E60`):

- from `102527`: `102539` (`obj_bg`), `102540` (`obj_only`), `102541` (`pb_t50_rs`)
- from `102531`: `102542` (`obj_bg`), `102543` (`obj_only`), `102544` (`pb_t50_rs`)
- from `102532`: `102545` (`obj_bg`), `102546` (`obj_only`), `102547` (`pb_t50_rs`)

Record:

- `logs/sanity/patchnepa_submit/patchnepa_tokens_fullbest_morton_20260304_042155/submitted_jobs_shortft.txt`
  - generated from all token-pretrain `*.pbs.log` files containing `[step ...]` diagnostic lines
  - includes tail-window metrics and max loss spike per run

## 132. `recon_chamfer` smoke started (2026-03-04)

Submitted after `recon_mse` wiring verification:

- job: `102524.qjcm` (`pntok_rcf`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041040/`
- pbs log:
  - `.../reconch_m50u50_smoke.pbs.log`

Key args:

- `PRETRAIN_OBJECTIVE=recon_chamfer`
- `RECON_CHAMFER_METRIC=l2`
- `MIX_CONFIG=...drop1_mesh50_udf50.yaml`
- `TOKEN_QA_LAYOUT=split_sep`
- `PATCH_ORDER_MODE=morton`
- `MAX_STEPS=2000`

Startup log confirms objective routing:

- `objective: pretrain_objective=recon_chamfer recon_w=(ctx=1.0,q=1.0,a=1.0) chamfer_metric=l2`

## 133. `recon_chamfer` bring-up fix and re-run (2026-03-04)

Initial submit:

- `102524.qjcm` (`pntok_rcf`)
- failed at startup:
  - `ModuleNotFoundError: No module named 'chamfer'`

Fixes applied:

1. Built extension in-place with multi-arch targets:
   - `Point-MAE/extensions/chamfer_dist`
   - `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0`
   - `python setup.py build_ext --inplace`
2. Updated loader path handling:
   - file: `nepa3d/train/pretrain_patch_nepa_tokens.py`
   - function: `_load_chamfer_module`
   - added `Point-MAE/extensions/chamfer_dist` to `sys.path` before import.

Re-submitted:

- `102526.qjcm` (`pntok_rcf`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041727/`

Early runtime verification:

- startup succeeded, W&B online
- step logs progressing (`[step 000001..]`)
- `loss_recon_ctx` is nonzero and active
- previous kernel-arch error (`no kernel image is available`) is not observed in `102526`

## 134. Default fix applied: ratio changes keep full ShapeNet total (2026-03-04)

User lock-in: composition ratio can vary, but per-epoch corpus size must remain full ShapeNet.

Applied to drop1 token configs:

- `shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh100.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_udf100.yaml`

Each now fixed to:

- `replacement: true`
- `mix_num_samples: 47445` (drop1 train total)

Compute-matched controls left unchanged by design:

- `*_cm15655.yaml` remain `replacement=false`, `mix_num_samples=15655`.

Launcher defaults aligned:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - default `MIX_CONFIG` -> `...drop1_pc33_mesh33_udf33.yaml`
  - default `TOKEN_QA_LAYOUT` -> `split_sep`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - same default values in qsub env payload

## 135. Reconstruction objective outcomes (recorded) (2026-03-04)

`recon_mse` (`102523`) completed:

- log: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_smoke_20260304_040234/recon_m50u50_smoke.log`
- final step `2000`:
  - `loss_total=0.296183`
  - `loss_recon_ctx=0.003766`
  - `loss_recon_q=0.173864`
  - `loss_recon_a=0.118553`
  - `cos_tgt=-0.0183`, `cos_prev=-0.0186`, `gap=+0.0004`, `copy_win=0.7397`

`recon_chamfer` (`102526`) progressed but no completion marker:

- log: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041727/reconch_m50u50_smoke.log`
- last observed step `969`:
  - `loss_total=0.287651`
  - `loss_recon_ctx=0.003372`
  - `loss_recon_q=0.166045`
  - `loss_recon_a=0.118234`
  - `cos_tgt=-0.0053`, `cos_prev=-0.0058`, `gap=+0.0005`, `copy_win=0.7381`
- `[done]` marker absent in this file.

## 136. Full300 FT outputs currently available (historical, pre-default-fix context)

Source:

- `logs/sanity/patchnepa_submit/patchnepa_tokens_full300_morton_fixddp2_20260304_044851/ft_*.out`

Observed TEST lines:

- `pc100`:
  - `obj_bg=0.8244`
  - `obj_only=0.8296`
  - `pb_t50_rs=0.7946`
- `mesh50udf50`:
  - `obj_bg=0.8090`
  - `obj_only=0.8296`
  - `pb_t50_rs` not yet observed in current snapshot

Interpretation constraint:

- these came from the run set before "full total per ratio arm" was enforced as default.
- keep as diagnostic history; re-run under new default for final ratio comparison.

## 137. Recon objective diagnostics corrected (2026-03-04)

Problem fixed:

- `recon_mse/recon_chamfer` runs were previously displaying cosine-space diagnostics
  (`cos_tgt/cos_prev/copy_win`) that are not objective-aligned.

Current behavior:

- `nepa_cosine`:
  - logs `diag/cos_tgt`, `diag/cos_prev`, `diag/gap`, `diag/copy_win` (unchanged).
- `recon_mse` / `recon_chamfer`:
  - logs reconstruction-space diagnostics:
    - `diag/recon_ctx_err`, `diag/recon_q_err`, `diag/recon_a_err`
    - `diag/copy_ctx_err`, `diag/copy_q_err`, `diag/copy_a_err`
    - `diag/recon_lift_ctx`, `diag/recon_lift_q`, `diag/recon_lift_a`

Interpretation rule:

- `recon_lift_* > 0` means model prediction beats copy-baseline in the same loss space.

## 138. Short re-run after recon path fixes (`105287`, `105288`) (2026-03-04)

Purpose:

- re-run short `mesh50+udf50` for `recon_mse` and `recon_chamfer`
- verify objective-aligned trend via `recon_lift_q/a`.

Fixes included before rerun:

- `pretrain_patch_nepa_tokens.py`
  - fixed DDP access bug in recon heads (`recon_heads(pred_h)` path).
  - fixed chamfer import collision (`Point-MAE/datasets` shadowing HF `datasets`)
    by loading chamfer extension from `Point-MAE/extensions/chamfer_dist` only.

Runs:

- `105287.qjcm` -> `reconmse_m50u50_short_fix`
- `105288.qjcm` -> `reconchamfer_m50u50_short_fix`

Both:

- completed with `[done]` marker
- `max_steps=2000`
- logs:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_diagfix_20260304_1835/reconmse_m50u50_short_fix.mr0.log`
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_diagfix_20260304_1835/reconchamfer_m50u50_short_fix.mr0.log`

Key final diagnostics:

- `recon_mse`:
  - `diag/recon_lift_q=0.2626`, `diag/recon_lift_a=0.1196`
  - `diag/recon_q_err=0.2600`, `diag/recon_a_err=0.1200`
  - `diag/copy_q_err=0.5226`, `diag/copy_a_err=0.2396`
- `recon_chamfer`:
  - `diag/recon_lift_q=0.2626`, `diag/recon_lift_a=0.1196`
  - `diag/recon_q_err=0.2599`, `diag/recon_a_err=0.1200`
  - `diag/copy_q_err=0.5226`, `diag/copy_a_err=0.2396`

Trend (first100 -> last100):

- `recon_mse`:
  - `lift_q: 0.1280 -> 0.2028`
  - `lift_a: 0.0261 -> 0.1180`
- `recon_chamfer`:
  - `lift_q: 0.1273 -> 0.2028`
  - `lift_a: 0.0260 -> 0.1180`

Conclusion:

- both objectives now show stable positive lift against copy baseline.
- under this setup, `recon_mse` and `recon_chamfer` behave almost identically.

## 139. Full FT launched from converged short recon checkpoints (2026-03-04)

Decision:

- short recon runs are in late-stage plateau (tail averages stable), so proceed to full FT as final criterion.

Source checkpoints:

- `runs/patchnepa_tokens/patchnepa_tokens_reconmse_diagfix_20260304_1835/ckpt_final.pt`
- `runs/patchnepa_tokens/patchnepa_tokens_reconchamfer_diagfix_20260304_1835/ckpt_final.pt`

Submission batch:

- `105297.qjcm` : `reconmse -> obj_bg`
- `105298.qjcm` : `reconmse -> obj_only`
- `105299.qjcm` : `reconmse -> pb_t50_rs`
- `105300.qjcm` : `reconchamfer -> obj_bg`
- `105301.qjcm` : `reconchamfer -> obj_only`
- `105302.qjcm` : `reconchamfer -> pb_t50_rs`

FT conditions:

- `EPOCHS=300`
- `N_POINT=2048`
- `BATCH=64` (`global`)
- `patchnepa_ft_mode=qa_zeroa`
- `pooling=cls_max`, `head_mode=pointmae_mlp`
- `AUG_PRESET=pointmae`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
- W&B on (`patchnepa-finetune`)

Submit log root:

- `logs/sanity/patchnepa_ft/ft_recon_full_20260304_190412/`

## 140. Centered-cosine objective path added (2026-03-04)

Purpose:

- address the "shared common-mode vector" failure mode under `nepa_cosine`
- keep NEPA structure (AR + stop-grad + content target) unchanged

Implemented in:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`

Added CLI options:

- `--nepa_center_mode {none,shape,segment}`
- `--nepa_center_warmup_frac <float>`

Behavior:

- `none`: legacy cosine path (no centering)
- `shape`: shape-wise centering before cosine objective
- `segment`: centering by segment (Q/A boundary-aware) before cosine objective
- warmup: centering contribution ramps from 0 to 1 over `nepa_center_warmup_frac` of total steps

Additional logs/W&B:

- `train/nepa_center_ramp`
- `diag/center_alpha`
- `diag/center_mode_id`
- centered diagnostics remain on standard keys:
  - `diag/cos_tgt`, `diag/cos_prev`, `diag/gap`, `diag/copy_win`
- raw reference diagnostics also emitted when centering is active:
  - `diag/cos_tgt_raw`, `diag/cos_prev_raw`, `diag/gap_raw`, `diag/copy_win_raw`

Launcher propagation added:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_multinode_pbsdsh.sh`

all now pass:

- `NEPA_CENTER_MODE`
- `NEPA_CENTER_WARMUP_FRAC`

## 141. Centered-cosine smoke runs (baseline/segment/shape) (2026-03-04)

Common settings:

- objective: `PRETRAIN_OBJECTIVE=nepa_cosine`
- `TOKEN_QA_LAYOUT=split_sep`
- `LOSS_MASK_MODE=answer_and_point_context`
- `PATCH_ORDER_MODE=morton`
- `SKIP_K=1`
- `BATCH=8` per GPU (`global=128`), `EPOCHS=10`, no ray (`N_RAY=0`)
- full diag logging (`DIAG_EVERY=1`, `WANDB_LOG_EVERY=1`)

### 141.1 Baseline (no centering) `105310`

- job: `105310.qjcm` (`ptoksmk`)
- log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_192014/`
- status: `F`, `Exit_status=0`
- final (`step 3700`):
  - `loss_total=0.573958`
  - `cos_tgt=0.4260`
  - `cos_prev=0.4274`
  - `gap=-0.0013`
  - `copy_win=0.7496`

### 141.2 Segment-centering `105319`

- job: `105319.qjcm` (`ptokcsmk`)
- centered options:
  - `NEPA_CENTER_MODE=segment`
  - `NEPA_CENTER_WARMUP_FRAC=0.05`
- log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_193732/`
- status: `F`, `Exit_status=0`
- final (`step 3700`):
  - `loss_total=0.473478`
  - `cos_tgt=0.5265`
  - `cos_prev=0.5470`
  - `gap=-0.0205`
  - `copy_win=0.7623`

### 141.3 Shape-centering `105320`

- job: `105320.qjcm` (`ptokhsmk`)
- centered options:
  - `NEPA_CENTER_MODE=shape`
  - `NEPA_CENTER_WARMUP_FRAC=0.05`
- log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_193922/`
- status: `F`, `Exit_status=0`
- final (`step 3700`):
  - `loss_total=0.024591`
  - `cos_tgt=0.9754`
  - `cos_prev=0.9750`
  - `gap=+0.0004`
  - `copy_win=0.7494`

## 142. Recording policy update for later analysis (2026-03-04)

- For each new branch, record:
  1. exact objective/mask/order/center settings,
  2. final (or latest) `loss_total/cos_tgt/cos_prev/gap/copy_win`,
  3. job id + log root + completion marker status.
- This runlog remains the canonical timeline; the restart plan keeps the condensed execution view.

## 143. `skip_k` short sweep launched (`k=1,2,4`) (2026-03-04)

Goal:

- quick A/B/C check whether increasing prediction horizon (`skip_k`) improves
  `gap = cos_tgt - cos_prev` and reduces copy-dominant behavior.

Fixed settings (identical across runs):

- `PRETRAIN_OBJECTIVE=nepa_cosine`
- `LOSS_MASK_MODE=answer_and_point_context`
- `TOKEN_QA_LAYOUT=split_sep`
- `PATCH_ORDER_MODE=morton`
- `NEPA_CENTER_MODE=none`
- `MIX_CONFIG=nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `MAX_STEPS=2000` (short smoke), `EPOCHS=0`
- `BATCH=8` per GPU (`global=128`)
- full diagnostics enabled (`DIAG_EVERY=1`, `WANDB_LOG_EVERY=1`)

Run set:

- `patchnepa_tokens_skipk124_smoke_20260304_200306`

Jobs:

- `105324.qjcm` (`ptoksk1`) -> `RUN_TAG=skipk1_m50u50_smoke`, `SKIP_K=1`
- `105325.qjcm` (`ptoksk2`) -> `RUN_TAG=skipk2_m50u50_smoke`, `SKIP_K=2`
- `105326.qjcm` (`ptoksk4`) -> `RUN_TAG=skipk4_m50u50_smoke`, `SKIP_K=4`

Log root:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_skipk124_smoke_20260304_200306/`

Final status:

- `105324`: `F`, `Exit_status=0`
- `105325`: `F`, `Exit_status=0`
- `105326`: `F`, `Exit_status=0`

Final (`step 2000`, `mr0`) comparison:

- `skip_k=1`
  - `loss_total=0.499003`
  - `cos_tgt=0.5010`
  - `cos_prev=0.5005`
  - `gap=+0.0005`
  - `copy_win=0.7504`
- `skip_k=2`
  - `loss_total=0.499757`
  - `cos_tgt=0.5002`
  - `cos_prev=0.4997`
  - `gap=+0.0005`
  - `copy_win=0.7475`
- `skip_k=4`
  - `loss_total=0.500622`
  - `cos_tgt=0.4994`
  - `cos_prev=0.4989`
  - `gap=+0.0005`
  - `copy_win=0.7453`

Observed takeaway:

- `k=1/2/4` all converge to essentially identical gap (`~+0.0005`) and near-identical loss; skip horizon alone does not break the `cos_tgt ≈ cos_prev` regime in this setup.

## 144. 200-step isolate for center warmup spike (`segment`, no warmup) (2026-03-04)

Goal:

- isolate whether the early spike around step ~100 in centerseg run is tied to warmup ramps.

Job:

- `105328.qjcm` (`ptokcs0`) -> `RUN_TAG=ptok_centerseg_nowarmup200`
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_centerseg_nowarmup200_20260304_201009/`
- settings:
  - `NEPA_CENTER_MODE=segment`
  - `NEPA_CENTER_WARMUP_FRAC=0.0`
  - `MAX_STEPS=200`, `EPOCHS=0`
  - other knobs fixed to the centerseg smoke branch (`nepa_cosine`, `split_sep`, `answer_and_point_context`, `morton`, `SKIP_K=1`)
- status: `F`, `Exit_status=0`

Window stats from `mr0`:

- no-warmup (`105328`)
  - `step 1-80`: `loss=0.5618`, `cos_tgt=0.4382`, `cos_prev=0.4439`, `gap=-0.0057`
  - `step 81-120`: `loss=0.4806`, `cos_tgt=0.5194`, `cos_prev=0.5414`, `gap=-0.0220`
  - `step 121-200`: `loss=0.4795`, `cos_tgt=0.5205`, `cos_prev=0.5553`, `gap=-0.0348`
- warmup-0.05 reference (`105319`, same branch)
  - `step 1-80`: `loss=0.5547`, `cos_tgt=0.4453`, `cos_prev=0.4452`, `gap=+0.0001`
  - `step 81-120`: `loss=0.0826`, `cos_tgt=0.9174`, `cos_prev=0.9165`, `gap=+0.0009`
  - `step 121-185`: `loss=0.2561`, `cos_tgt=0.7438`, `cos_prev=0.7428`, `gap=+0.0011`
  - `step 186-260`: `loss=0.4825`, `cos_tgt=0.5175`, `cos_prev=0.5176`, `gap=-0.0002`

Interpretation:

- yes, the sharp early surge/drop pattern aligns with warmup schedules (LR + center ramp completion boundaries); removing center warmup removes that transient.
- however, removing warmup does not improve the core issue: the run settles into stronger negative gap (`cos_prev > cos_tgt`).

## 145. Backfill: previously unreferenced token run roots (2026-03-04)

Purpose:

- ensure all generated run roots are explicitly classified (kept / superseded / invalid), so later analysis does not confuse orphan logs with valid evidence.

### 145.1 Empty roots (no files produced; non-materialized)

- `patchnepa_tokens_20260304_005229`
- `patchnepa_tokens_20260304_005248`
- `patchnepa_tokens_20260304_032446`
- `patchnepa_tokens_drop1_20260304_001817`
- `patchnepa_tokens_drop1_answercheck_20260304_005258`
- `patchnepa_tokens_drop1_m50u50_fps_20260304_022852`
- `patchnepa_tokens_drop1_tokens_maskprimfix_random_20260304_021624`
- `patchnepa_tokens_dualcol_20260304_032435`
- `patchnepa_tokens_reconmse_smoke_20260304_040226`
- `patchnepa_v2tok_pc100_smoke_20260303_193214`

Interpretation:

- launcher/submit side roots were created, but no actual per-rank logs were materialized.
- these are non-evaluable and excluded from comparison.

### 145.2 PBS-only roots (no `mr0`, early launch failure / replaced)

- `patchnepa_tokens_drop1_answercheck_20260304_005330`
- `patchnepa_tokens_drop1_m50u50_fps_20260304_022859`
- `patchnepa_tokens_m50u50_ordercmp_20260304_012339`
- `patchnepa_tokens_varcov_20260304_025805`
- `patchnepa_tokens_varcov_intra_20260304_030620`
- `patchnepa_tokens_hreg_20260304_033023`
- `patchnepa_tokens_hreg_dcol_nf03_20260304_034404`
- `patchnepa_tokens_dualcol_20260304_032501`
- `patchnepa_tokens_reconchamfer_smoke_20260304_041358`
- `patchnepa_v2tok_mesh50udf50_smoke_20260303_193516`
- `patchnepa_v2tok_pc33mesh33udf33_smoke_20260303_193516`

Policy:

- kept as traceability artifacts only; all were superseded by later valid reruns already tracked in sections 124-144.

### 145.3 Explicit failed chains with logs (now superseded)

- Full300 first try (pre-DDP fixes):
  - root: `patchnepa_tokens_full300_morton_20260304_044214`
  - jobs: `102560/102561/102562`, all failed with missing pretrain marker (`Exit_status=97`).
- Full300 second try (partial DDP fix):
  - root: `patchnepa_tokens_full300_morton_fixddp_20260304_044633`
  - jobs: `102572/102573/102574`, same early DDP failure (`Exit_status=97`).
- Full300 short-lived submit batch root:
  - root: `patchnepa_tokens_full300_morton_20260304_044022`
  - jobs: `102548/102549/102550`, launch-level failure path (replaced immediately).
- cm15655 branch (partial training, not cardinality-locked):
  - root: `patchnepa_tokens_full300_morton_cm15655_20260304_102805`
  - job: `102828`, stopped at ~`step 2916-2950` (`Exit_status=271`), not used for final fair comparison.
- recon short diag first attempt (before diagfix):
  - roots: `patchnepa_tokens_reconmse_diag_20260304_181957`, `patchnepa_tokens_reconchamfer_diag_20260304_181957`
  - jobs: `105285`, `105286` (`Exit_status=97`), superseded by `105287/105288` (section 138).
- umbrella token run:
  - root: `patchnepa_tokens_20260304_181941`
  - job: `105284` (`Exit_status=265`), terminated before step logging.

Takeaway:

- these roots are now fully classified; downstream analysis should use validated branches only (sections 138-144 and subsequent full reruns).

## 146. InfoNCE vs Residual (centered-cosine) status snapshot (2026-03-04)

InfoNCE branch status:

- no completed token-pretrain run has been launched with `pretrain_objective=nepa_infonce` yet.
- in all observed recent runs, `loss_infonce` is exactly `0.0` throughout (objective path inactive).

Residual (centered-cosine) branch status:

- baseline (`center_mode=none`, `105310`, step 3700):
  - `loss_total=0.573958`, `cos_tgt=0.4260`, `cos_prev=0.4274`, `gap=-0.0013`, `copy_win=0.7496`
- segment (`center_mode=segment`, warmup `0.05`, `105319`, step 3700):
  - `loss_total=0.473478`, `cos_tgt=0.5265`, `cos_prev=0.5470`, `gap=-0.0205`, `copy_win=0.7623`
- shape (`center_mode=shape`, warmup `0.05`, `105320`, step 3700):
  - `loss_total=0.024591`, `cos_tgt=0.9754`, `cos_prev=0.9750`, `gap=+0.0004`, `copy_win=0.7494`
- segment no-warmup isolate (`105328`, step 200):
  - `loss_total=0.480784`, `cos_tgt=0.5192`, `cos_prev=0.5579`, `gap=-0.0387`, `copy_win=0.7832`

Current interpretation:

- residual centering changes optimization dynamics, but does not robustly open `gap=cos_tgt-cos_prev`.
- `shape` mode reaches very high cosine with near-zero gap (copy-risk unresolved).
- `segment` mode tends to negative gap in stable region.

## 147. InfoNCE short smoke submitted (`m50u50`, morton, split_sep) (2026-03-04)

Submission:

- job: `105332.qjcm` (`pntok_nce`)
- run set: `patchnepa_tokens_infonce_smoke_20260304_205108`
- run tag: `infonce_m50u50_smoke`

Fixed settings:

- `pretrain_objective=infonce`
- `infonce_tau=0.07`
- `mix_config=nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `patch_order_mode=morton`
- `token_qa_layout=split_sep`
- `loss_mask_mode=answer_and_point_context`
- `skip_k=1`
- `nepa_center_mode=none`
- `max_steps=2000`, `epochs=0`
- full diagnostics on: `diag_every=1`, `wandb_log_every=1`

Paths:

- pbs log:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_infonce_smoke_20260304_205108/infonce_m50u50_smoke.pbs.log`
- ddp run dir:
  - `logs/ddp_patch_nepa_tokens_pretrain/ddp_patchnepa_tokens_105332.qjcm_infonce_m50u50_smoke/`

Initial state:

- job entered `R` and node-entry logs confirm objective is `infonce` (not cosine fallback).

## 148. InfoNCE interim snapshot + preliminary insight (`105332`, in progress) (2026-03-04)

Source:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_infonce_smoke_20260304_205108/infonce_m50u50_smoke.mr0.log`

Interim readout (latest observed `step=1168`):

- `loss_total=6.828708`
- `loss_infonce=6.828708`
- `loss_nepa=0.690645` (diagnostic reference only; not optimization target)
- `pos_cos=0.3094`, `neg_cos=-0.0325`, `margin=0.3419`
- `r1=0.0046`
- `cos_tgt=0.3094`, `cos_prev=0.3090`, `gap=+0.0003`
- `copy_win=0.7319`

Window summaries:

- `step 1-80`: `loss=7.0012`, `cos_tgt=0.1722`, `cos_prev=0.1722`, `gap≈0`
- `step 81-120`: `loss=6.8470`, `cos_tgt=0.2747`, `cos_prev=0.2751`, `gap≈0`
- `tail100 (1069-1168)`:  
  `loss=6.8292`, `cos_tgt=0.3058`, `cos_prev=0.3057`, `gap=+0.0002`,  
  `pos_cos=0.3058`, `neg_cos=-0.0330`, `margin=0.3388`

Preliminary interpretation (not final until step 2000):

- InfoNCE is active and separates positives from negatives (`margin > 0`, `neg_cos < 0`).
- however, at this interim point the copy-risk proxy remains unresolved (`cos_tgt ≈ cos_prev`, near-zero gap), similar to prior cosine-family behavior.

## 149. `reconbest` full300 finalized (`g0`, 2026-03-05/06)

Lineage:

- submit root:
  - `logs/sanity/patchnepa_submit/patchnepa_reconbest_full300_20260305_224714`
- pretrain log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_reconbest_full300_20260305_224714`
- FT log root:
  - `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714`

Pretrain jobs:

- `105868` : `pc100`
- `105869` : `mesh50udf50`
- `105870` : `pc33mesh33udf33`

Dependent FT jobs:

- from `105868`: `105872/105873/105874`
- from `105869`: `105875/105876/105877`
- from `105870`: `105878/105879/105880`

Save roots:

- `runs/patchnepa_tokens/patchnepa_reconbest_full300_20260305_224714/pt_pc100_reconch_g0_e300`
- `runs/patchnepa_tokens/patchnepa_reconbest_full300_20260305_224714/pt_mesh50udf50_reconch_g0_e300`
- `runs/patchnepa_tokens/patchnepa_reconbest_full300_20260305_224714/pt_pc33mesh33udf33_reconch_g0_e300`

Policy note (2026-03-14):

- these FT rows were produced under the earlier ScanObjectNN
  `val_split_mode=file` policy
- keep the numbers for provenance and internal comparison only
- they are not the current canonical ScanObjectNN benchmark headline

Final readout:

| source | `recon_lift_q` | `recon_lift_a` | `obj_bg` | `obj_only` | `pb_t50_rs` |
|---|---:|---:|---:|---:|---:|
| `pc100` | `0.1349` | `0.1096` | `0.8348` | `0.8107` | `0.7998` |
| `mesh50udf50` | `0.1896` | `0.1466` | `0.8399` | `0.8227` | `0.8001` |
| `pc33mesh33udf33` | `0.1640` | `0.1357` | `0.8365` | `0.8348` | `0.8102` |

Historical headline:

- best-of-three headline = `0.8399 / 0.8348 / 0.8102`
- this beats the historical v1 reference on `obj_bg` and `pb_t50_rs`, but not
  on `obj_only`.

Canonical sources:

- pretrain diag:
  - `wandb/run-20260305_224819-qdxxc2vn/files/wandb-summary.json`
  - `wandb/run-20260305_224815-lv9e9hce/files/wandb-summary.json`
  - `wandb/run-20260305_224815-c27h2zlv/files/wandb-summary.json`
- FT logs:
  - `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714/*.out`

## 150. `recong2` full300 finalized (`g2`, 2026-03-06)

Lineage:

- submit root:
  - `logs/sanity/patchnepa_submit/patchnepa_recong2_full300_20260306_072643`
- pretrain log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_recong2_full300_20260306_072643`
- FT log root:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643`

Pretrain jobs:

- `105911` : `pc100`
- `105912` : `mesh50udf50`
- `105913` : `pc33mesh33udf33`

Dependent FT jobs:

- from `105911`: `105914/105915/105916`
- from `105912`: `105917/105918/105919`
- from `105913`: `105920/105921/105922`

Save roots:

- `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc100_reconch_g2_e300`
- `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_mesh50udf50_reconch_g2_e300`
- `runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc33mesh33udf33_reconch_g2_e300`

Policy note (2026-03-14):

- these FT rows were produced under the earlier ScanObjectNN
  `val_split_mode=file` policy
- keep the numbers for provenance and internal comparison only
- they are not the current canonical ScanObjectNN benchmark headline

Final readout:

| source | `recon_lift_q` | `recon_lift_a` | `obj_bg` | `obj_only` | `pb_t50_rs` |
|---|---:|---:|---:|---:|---:|
| `pc100` | `0.1520` | `0.1031` | `0.8279` | `0.8399` | `0.7932` |
| `mesh50udf50` | `0.1666` | `0.1507` | `0.8485` | `0.8434` | `0.8053` |
| `pc33mesh33udf33` | `0.1821` | `0.1421` | `0.8417` | `0.8589` | `0.8140` |

Historical headline:

- best-of-three headline = `0.8485 / 0.8589 / 0.8140`
- this historical file-split readout beats the historical v1 reference and the
  `g0` historical file-split best-of-three headline on all three
  ScanObjectNN variants.

Canonical sources:

- pretrain diag:
  - `wandb/run-20260306_072734-aeoo6m8d/files/wandb-summary.json`
  - `wandb/run-20260306_073118-h1hud0eo/files/wandb-summary.json`
  - `wandb/run-20260306_073118-ocn054xz/files/wandb-summary.json`
- FT logs:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643/*.out`

## 151. Translation-loss short sweep + mini-CPAC rerun completed (2026-03-06)

Short pretrain sweep lineage:

- run set:
  - `patchnepa_tokens_translationloss_pc33_g0_20260306_114629_{cmp,ans,cpa}`
- pretrain jobs:
  - `106029` (`composite`)
  - `106031` (`answer_only`)
  - `106033` (`context_plus_answer`)
- save roots:
  - `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_cmp/ckpt_final.pt`
  - `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_ans/ckpt_final.pt`
  - `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_cpa/ckpt_final.pt`

Initial dependent CPAC jobs:

- `106030/106032/106034`
- all invalid as canonical evidence:
  - requested checkpoint path was one directory too deep,
  - logs terminate with `[error] ckpt not found`.

Corrected CPAC rerun lineage:

- canonical log root:
  - `logs/patch_nepa_cpac/patchnepa_tokens_translationloss_pc33_g0_20260306_1900_rerun2`
- canonical result root:
  - `results/patch_nepa_cpac/patchnepa_tokens_translationloss_pc33_g0_20260306_1900_rerun2`
- corrected jobs:
  - `106571` (`context_plus_answer`)
  - `106572` (`answer_only`)
  - `106573` (`composite`)

Short pretrain readout:

| mode | pretrain job | `recon_lift_q` | `recon_lift_a` | `target_std_mean` |
|---|---:|---:|---:|---:|
| `composite` | `106029` | `+0.1371` | `0.1132` | `0.1423` |
| `answer_only` | `106031` | `-0.2415` | `0.1133` | `0.3654` |
| `context_plus_answer` | `106033` | `-0.1777` | `0.1130` | `0.1443` |

Canonical mini-CPAC readout:

| mode | corrected job | JSON | `iou@0.01` | `mae` | `rmse` |
|---|---:|---|---:|---:|---:|
| `composite` | `106573` | `cpac_pc2udf_cmp_fix.json` | `0.0948` | `0.07585` | `0.09929` |
| `answer_only` | `106572` | `cpac_pc2udf_ans_fix.json` | `0.1033` | `0.07584` | `0.09991` |
| `context_plus_answer` | `106571` | `cpac_pc2udf_cpa_fix.json` | `0.0954` | `0.07653` | `0.10043` |

Ledger decision:

- keep `composite` as the reconstruction baseline,
- keep translation-centric modes as screening controls until a mode wins on both
  pretrain and downstream criteria.

Canonical source note:

- use `logs/patch_nepa_cpac/...` and `results/patch_nepa_cpac/...json`,
  not flat `pntok_cpac.o*`, as the persistent source of truth.

## 152. `fps_then_sample` FT ablation recorded, not benchmark-eligible (2026-03-06)

Lineage:

- job:
  - `106582`
- run root:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_pc33_objonly_fpsparity_20260306_1810`
- compared against:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643/pc33mesh33udf33_obj_only.out`

Result:

| run | train sampler | best `val_acc` | `TEST acc` |
|---|---|---:|---:|
| baseline `g2` | `random` | `0.8969` | `0.8589` |
| parity screen | `fps_then_sample` | `0.8924` | `0.8193` |

Why this row is not benchmark-eligible:

- the current `scanobjectnn_*_v3_nonorm` cache already stores exactly `2048`
  input points per sample,
- therefore `fps_then_sample` on this cache degenerates to a full-set
  permutation, not a true `point_all > npoints` crop like Point-MAE,
- this run is retained as an inconclusive ablation, not as headline evidence.

## 153. Visibility-first branch launched (`L000A/L000B`, 2026-03-07)

Lineage:

- branch script:
  - `scripts/local/patchnepa_visocc_branch.sh`
- runtime root:
  - `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260306`
- decision artifact:
  - `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260306/decision.json`

Data lineage prepared by this branch:

- source cache:
  - `data/shapenet_cache_v2_20260306_visocc`
- split JSON:
  - `data/shapenet_unpaired_splits_v2_20260306_visocc.json`
  - `data/shapenet_unpaired_splits_v2_pc33_mesh33_udf33_visocc.json`
  - `data/shapenet_unpaired_splits_v2_mesh50_udf50_visocc.json`
- unpaired caches:
  - `data/shapenet_unpaired_cache_v2_20260306_visocc`
  - `data/shapenet_unpaired_cache_v2_20260306_visocc_drop1`
  - `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33_visocc`
  - `data/shapenet_unpaired_cache_v2_mesh50_udf50_visocc`

Schema change under test:

- added mesh-query-aligned keys:
  - `mesh_qry_vis_hit`
  - `mesh_qry_vis_t`
- active visibility answer schema:
  - `[n, curv, dist, grad, density, vis_hit, vis_t]`

Config paths used by the short screen:

- baseline:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc_base.yaml`
- visocc:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc.yaml`

Pre-launch validation artifacts:

- smoke NPZ:
  - `results/_tmp_visocc_smoke/sample_visocc.npz`
- validation result:
  - `mesh_qry_vis_hit.shape = (64, 16)`
  - `mesh_qry_vis_t.shape = (64, 16)`
  - values are nontrivial and bounded by `max_t = 0.25`

Operational correction applied before the live run:

- first launch failed immediately because the preprocess root pointed at
  `data/ShapeNetCore.v2` while the current source layout lives under
  `data/ShapeNetCore.v2/synsets`
- fixed defaults:
  - `scripts/preprocess/preprocess_shapenet_v2.sh`
  - `scripts/local/patchnepa_visocc_branch.sh`

Status at record time:

- source-cache rebuild is running
- no canonical metrics exist yet
- `L001-L004` remain gated until this branch resolves

## 154. Worldvis cache accepted at `52311` shapes; split/materialize completed from existing source cache (2026-03-12)

Lineage:

- source cache:
  - `data/shapenet_cache_v2_20260311_worldvis`
- accepted source count:
  - `52311 / 52472`
- missing-shape records:
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_by_synset.tsv`
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_synset_summary.tsv`
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_by_synset.json`

Interpretation:

- the unresolved `161` source misses are recorded as **shape dropout**,
- they are not reinterpreted as modality-dropout or `pc-only` fallback samples,
- pipeline proceeds with the existing `52311` source NPZs.

Observed missing concentration:

- top synsets by missing count:
  - `02691156`: `42 / 4045`
  - `03790512`: `22 / 337`
  - `04530566`: `22 / 1939`
  - `02958343`: `18 / 3514`
  - `04468005`: `9 / 389`
- global rate remains `0.3068%`.

Immediate post-cache continuation on `rt_QC` without preprocess dependency:

- base:
  - split job `112178`
  - materialize job `112179`
  - log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_base`
  - output:
    - `created=52311`, `missing=0`
    - split counts `train_mesh=16004`, `train_pc=15533`, `train_udf=15533`, `eval=5241`
- drop1:
  - split job `112180`
  - materialize job `112181`
  - log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_drop1`
  - output:
    - `created=52311`, `missing=0`
    - split counts `train_mesh=16004`, `train_pc=15533`, `train_udf=15533`, `eval=5241`
- pc33mesh33udf33:
  - split job `112182`
  - materialize job `112183`
  - log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_pc33`
  - output:
    - `created=52311`, `missing=0`
    - split counts `train_mesh=15533`, `train_pc=15533`, `train_udf=16004`, `eval=5241`
- mesh50udf50:
  - split job `112184`
  - materialize job `112185`
  - log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_mesh50udf50`
  - output:
    - `created=52311`, `missing=0`
    - split counts `train_mesh=23534`, `train_udf=23536`, `eval=5241`

Operational note:

- directory-level counts can show `52312` because `_meta/split_source.json` is
  included,
- canonical cache size is the materializer's `created=52311`.

## 155. CQA vis/thickness 2k smoke recorded; current tasks are majority-code dominated (2026-03-15)

Lineage:

- ABCI job:
  - `113012`
- save root:
  - `runs/cqa/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_smoke_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_smoke_20260315`
  - run `cqa_visthick_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000/eval_controls_128.json`
- target audit:
  - `results/cqa_audit/patchnepa_cqa_target_audit_20260315/summary.json`

Smoke config and scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml`
- tasks:
  - `mesh_visibility`
  - `udf_thickness`
- context:
  - both use `context_source=surf`
- meaning:
  - this is a branch-wiring smoke, not cross-primitive headline evidence.

Teacher-forced token eval (`128` eval shapes / task):

| control | overall `ce` | overall acc |
|---|---:|---:|
| `correct` | `1.623744` | `0.627747` |
| `no_context` | `1.805216` | `0.626038` |
| `shuffled_context` | `1.638101` | `0.628540` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `1.726435` | `0.628235` |

Per-task `correct` control:

| task | `ce` | acc | predicted mode |
|---|---:|---:|---|
| `mesh_visibility` | `1.986238` | `0.587524` | token `128` on `8191 / 8192` predictions |
| `udf_thickness` | `1.261251` | `0.667969` | token `448` on `8016 / 8192` predictions |

Majority-baseline comparison from the same evaluation slice:

- `mesh_visibility` target top-1 share:
  - token `128` on `4814 / 8192` targets (`0.587646`)
- `udf_thickness` target top-1 share:
  - token `448` on `5484 / 8192` targets (`0.669434`)
- current read:
  - the model is still extremely close to the majority-code baseline on both
    tasks.

Target-audit summary (`256` shapes / task, current worldvis drop1 cache):

| task | top-1 share | entropy (bits) | note |
|---|---:|---:|---|
| `udf_distance` | `0.060730` | `5.622965` | best next target candidate |
| `udf_clearance` | `0.696594` | `2.209615` | front-only in current code; acceptable next diagnostic |
| `udf_thickness` | `0.663574` | `1.839354` | too imbalanced for the next headline smoke |

Next-step configs staged from this read:

- `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfsurf.yaml`
  - next mainline smoke candidate: `surf -> udf_distance + udf_clearance`
- `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`
  - paired diagnostic: `pc_bank -> udf_distance + udf_clearance`

Operational interpretation:

- keep the explicit-query CQA branch alive,
- do not claim promptable cross-primitive answering from `113012`,
- move next experiments to target/data audit plus UDF-centric smokes before any
  further model changes.

## 156. CQA UDF-surf smoke recorded; `udf_distance` beats majority but context use is still weak (2026-03-15)

Lineage:

- ABCI job:
  - `113068`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfsurf.yaml`
- save root:
  - `runs/cqa/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_udfsurf_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_udfsurf_20260315`
  - run `patchnepa_cqa_udfsurf_20260315_cqa_udfdist_clearfront_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000/eval_controls_128.json`

Task scope:

- `udf_distance`
- `udf_clearance` (`front` only in current code)
- `context_source=surf` for both tasks

Teacher-forced token eval (`128` shapes / task):

| control | overall `ce` | overall acc |
|---|---:|---:|
| `correct` | `2.426786` | `0.405212` |
| `no_context` | `2.595490` | `0.392090` |
| `shuffled_context` | `2.435271` | `0.404236` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `3.446803` | `0.368103` |

Per-task `correct` read:

| task | `ce` | acc | predicted mode |
|---|---:|---:|---|
| `udf_distance` | `3.344186` | `0.109253` | distributed across many bins; top predictions `610/608/639/636/609` |
| `udf_clearance(front)` | `1.509385` | `0.701172` | token `512` on `8171 / 8192` predictions |

Same-slice majority baseline:

- `udf_distance`:
  - target top-1 share `0.058594`
  - result:
    - model beats majority (`0.109253 > 0.058594`)
- `udf_clearance(front)`:
  - target top-1 share `0.701538`
  - result:
    - model remains effectively tied to the majority baseline

Control read:

- `udf_distance`
  - `no_context`: `delta_ce=+0.205935`, `delta_acc=-2.66pt`
  - `shuffled_context`: `delta_ce=+0.008910`, `delta_acc=-0.22pt`
  - `shuffled_query`: `delta_ce=+2.039272`, `delta_acc=-7.42pt`
- `udf_clearance(front)`
  - `no_context`: `delta_ce=+0.131473`, `delta_acc=+0.04pt`
  - `shuffled_context`: `delta_ce=+0.008061`, `delta_acc=+0.02pt`
  - `shuffled_query`: `delta_ce=+0.000764`, `delta_acc=+0.00pt`

Operational interpretation:

- `udf_distance` is now the first explicit-query CQA task that clearly beats
  the majority baseline on the same eval slice,
- however, the branch is still not strongly context-conditioned:
  `shuffled_context` is almost inert and `no_context` moves `udf_distance` only
  modestly,
- `udf_clearance(front)` remains a secondary diagnostic task, not a headline
  one,
- next queue target should be the paired diagnostic config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`
  - to test whether `pc_bank` creates stronger prompt sensitivity on the same
    UDF tasks.

## 157. CQA `pc_bank` diagnostic recorded; heterogeneous context still does not help much (2026-03-15)

Lineage:

- ABCI job:
  - `113071`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`
- save root:
  - `runs/cqa/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_udfpcdiag_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_udfpcdiag_20260315`
  - run `patchnepa_cqa_udfpcdiag_20260315_cqa_udfdist_clearfront_pcdiag_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000/eval_controls_128.json`

Task scope:

- `udf_distance`
- `udf_clearance(front)`
- `context_source=pc_bank` for both tasks

Teacher-forced token eval (`128` shapes / task):

| control | overall `ce` | overall acc |
|---|---:|---:|
| `correct` | `2.375775` | `0.416748` |
| `no_context` | `2.507337` | `0.405762` |
| `shuffled_context` | `2.383283` | `0.416687` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `3.334098` | `0.378052` |

Per-task `correct` read:

| task | `ce` | acc | note |
|---|---:|---:|---|
| `udf_distance` | `3.338770` | `0.110962` | still beats the same-slice majority baseline (`0.058594`) |
| `udf_clearance(front)` | `1.412780` | `0.722534` | still near-majority (`0.701538`) |

Control read:

- `udf_distance`
  - `no_context`: `delta_ce=+0.139570`, `delta_acc=-2.20pt`
  - `shuffled_context`: `delta_ce=+0.004313`, `delta_acc=-0.01pt`
  - `shuffled_query`: `delta_ce=+1.914344`, `delta_acc=-7.74pt`
- `udf_clearance(front)`
  - `no_context`: `delta_ce=+0.123555`, `delta_acc=+0.00pt`
  - `shuffled_context`: `delta_ce=+0.010704`, `delta_acc=+0.00pt`
  - `shuffled_query`: `delta_ce=+0.002302`, `delta_acc=+0.00pt`

Comparison against the preceding `surf` CQA smoke:

- `udf_distance` remains the only clearly useful task,
- `pc_bank` does **not** improve context sensitivity relative to `surf`,
- `shuffled_query` remains the dominant control for `udf_distance`,
- `shuffled_context` remains nearly inert.

Operational interpretation:

- do not promote `pc_bank` as evidence that heterogeneous context is now
  helping,
- keep `udf_distance` as the current best explicit-query CQA task,
- keep `udf_clearance(front)` as a diagnostic only,
- next branch changes should focus on stronger context-conditioned task
  definitions rather than repeating the same `pc_bank` setup.

## 158. CQA single-task `udf_distance` smoke recorded; no-context sensitivity improves, wrong-shape remains modest (2026-03-15)

Lineage:

- ABCI job:
  - `113102`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_udfdist_20260315`
  - run `patchnepa_cqa_udfdist_20260315_cqa_udfdist_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000/eval_controls_128.json`

Task scope:

- `udf_distance` only
- `context_source=surf`

Teacher-forced token eval (`128` shapes):

| control | `ce` | acc |
|---|---:|---:|
| `correct` | `2.953239` | `0.155151` |
| `no_context` | `3.577064` | `0.100586` |
| `wrong_shape` | `3.063285` | `0.139282` |
| `shuffled_context` | `3.065694` | `0.136108` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `7.061647` | `0.031372` |

Per-task `correct` read:

- `udf_distance`
  - majority baseline: `0.058594`
  - model acc: `0.155151`
  - `pred_top1_share=0.109253`
  - `pred_unique_codes=31`

Control read:

- `no_context`
  - `delta_ce=+0.623825`
  - `delta_acc=-5.46pt`
- `wrong_shape`
  - `delta_ce=+0.110045`
  - `delta_acc=-1.59pt`
- `shuffled_context`
  - `delta_ce=+0.112455`
  - `delta_acc=-1.90pt`
- `shuffled_query`
  - `delta_ce=+4.108408`
  - `delta_acc=-12.38pt`

Operational interpretation:

- this is the strongest explicit-query CQA run so far,
- `udf_distance` remains well above the majority baseline and shows a much
  broader prediction spread than the earlier mixed-task runs,
- `no_context` now hurts substantially more than before, so context use is
  stronger,
- but `wrong_shape` is still modest, so the branch is not yet strongly
  shape-conditioned,
- the next gate should be a harder `udf_distance` query subset
  (`near_surface`) rather than another immediate `pc_bank` repeat.

## 159. CQA near-surface `udf_distance` hard-query smoke recorded; the task collapses back below majority (2026-03-15)

Lineage:

- ABCI job:
  - `113125`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_near.yaml`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_udfdist_near_20260315`
  - run `patchnepa_cqa_udfdist_near_20260315_cqa_udfdist_near_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000/eval_controls_128.json`

Task scope:

- `udf_distance` only
- `context_source=surf`
- `query_src_filter=near_surface`

Teacher-forced token eval (`128` shapes):

| control | `ce` | acc |
|---|---:|---:|
| `correct` | `3.637897` | `0.041748` |
| `no_context` | `3.638731` | `0.040894` |
| `wrong_shape` | `3.637672` | `0.042358` |
| `shuffled_context` | `3.637673` | `0.040894` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `3.637967` | `0.043701` |

Per-task `correct` read:

- `udf_distance`
  - majority baseline: `0.043945`
  - model acc: `0.041748`
  - `pred_top1_share=0.395874`
  - `pred_unique_codes=6`

Control read:

- `no_context`
  - `delta_ce=+0.000834`
  - `delta_acc=-0.09pt`
- `wrong_shape`
  - `delta_ce=-0.000225`
  - `delta_acc=+0.06pt`
- `shuffled_context`
  - `delta_ce=-0.000225`
  - `delta_acc=-0.09pt`
- `shuffled_query`
  - `delta_ce=+0.000069`
  - `delta_acc=+0.20pt`

Operational interpretation:

- the naive `near_surface` hard-query restriction is a strong negative result,
- `udf_distance` drops back below its same-slice majority baseline,
- the prediction spread collapses sharply,
- context and query controls become effectively inert.

## 160. CQA near-surface `pc_bank` retry recorded; heterogeneous context does not rescue the hard-query collapse (2026-03-15)

Lineage:

- ABCI job:
  - `113127`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_near_pcdiag.yaml`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000`
- log root:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_pcdiag_20260315`
- PBS log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000.pbs.log`
- W&B:
  - project `patchnepa-cqa-smoke`
  - group `patchnepa_cqa_udfdist_near_pcdiag_20260315`
  - run `patchnepa_cqa_udfdist_near_pcdiag_20260315_cqa_udfdist_near_pcdiag_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000/eval_controls_128.json`

Task scope:

- `udf_distance` only
- `context_source=pc_bank`
- `query_src_filter=near_surface`

Teacher-forced token eval (`128` shapes):

| control | `ce` | acc |
|---|---:|---:|
| `correct` | `3.642637` | `0.043335` |
| `no_context` | `3.643174` | `0.044189` |
| `wrong_shape` | `3.642695` | `0.043457` |
| `shuffled_context` | `3.642707` | `0.044189` |
| `wrong_type` | `999999552.0` | `0.0` |
| `shuffled_query` | `3.642593` | `0.043091` |

Per-task `correct` read:

- `udf_distance`
  - majority baseline: `0.044067`
  - model acc: `0.043335`
  - `pred_top1_share=0.468994`
  - `pred_unique_codes=7`

Control read:

- `no_context`
  - `delta_ce=+0.000537`
  - `delta_acc=+0.09pt`
- `wrong_shape`
  - `delta_ce=+0.000058`
  - `delta_acc=+0.01pt`
- `shuffled_context`
  - `delta_ce=+0.000070`
  - `delta_acc=+0.09pt`
- `shuffled_query`
  - `delta_ce=-0.000044`
  - `delta_acc=-0.02pt`

Operational interpretation:

- `pc_bank` does not rescue the hard-query collapse,
- the run remains below majority and even more concentrated than the surf
  hard-query run,
- heterogeneous context still has no positive evidence under the current
  `near_surface` recipe.

## 161. `world_v3` freeze sprint completed; existing `worldvis` cache is now the fixed base contract (2026-03-16)

Lineage:

- ABCI job:
  - `113209`
- worker:
  - `scripts/preprocess/freeze_shapenet_world_v3_qc.sh`
- submit wrapper:
  - `scripts/preprocess/submit_world_v3_freeze_qc.sh`
- source cache:
  - `data/shapenet_cache_v2_20260311_worldvis`
- schema spec:
  - `nepa3d/docs/patch_nepa/spec_world_v3_schema.md`
- outputs:
  - `results/data_freeze/world_v3_freeze_20260316_qc/augment_world_v3_summary.json`
  - `results/data_freeze/world_v3_freeze_20260316_qc/world_v3_audit_summary.json`
  - `results/data_freeze/world_v3_freeze_20260316_qc/subset_watertight_manifest.json`
  - `results/data_freeze/world_v3_freeze_20260316_qc/subset_watertight_summary.json`

Summary:

- augmentation:
  - `updated=39761`
  - `skipped=12550`
  - `errors=[]`
- frozen-cache quality:
  - `shape_count=52311`
  - `watertight_rate=0.002409`
  - `winding_consistent_rate=0.586263`
  - `visibility_fallback_used_rate=0.0`
- field-validity means:
  - `udf_surf_hit_out_rate_mean=0.744845`
  - `udf_clear_front_valid_rate_mean=0.744845`
  - `udf_clear_back_valid_rate_mean=0.773156`
  - `udf_probe_valid_rate_mean=1.0`
- strict clean subset:
  - `119 / 52311`
  - `train=108`
  - `test=11`

Operational interpretation:

- Phase 1 dataset freeze is complete without evidence that a full rebuild is
  necessary.
- the strict watertight subset is too small to replace the base corpus, so it
  remains a side manifest for future pivots rather than a new default cache.
- future CQA / recon reads should reference the fixed `world_v3` contract on
  top of the existing `worldvis` source cache.

## 162. Frozen-`world_v3` CQA rerun recorded; `udf_distance` reproduces at 2k and strengthens strongly by 10k (2026-03-16)

Lineage:

- `C008` ABCI job:
  - `113212`
- `C009` ABCI job:
  - `113213`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml`
- rerun save root:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_reval_20260316/cqa_udfdist_worldv3_g2_s2000`
- curve save root:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000`
- rerun eval:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_reval_20260316/cqa_udfdist_worldv3_g2_s2000/eval_controls_128.json`
- curve final eval:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/eval_controls_128.json`
- curve summary:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/eval_curve_128/curve_summary.json`

Summary:

- `C008`:
  - `acc=0.106445`
  - majority `=0.037598`
  - `pred_top1_share=0.211792`
  - `pred_unique_codes=33`
  - `delta_ce(no_context)=+1.059862`
  - `delta_ce(wrong_shape_same_synset)=+0.319331`
  - `delta_ce(wrong_shape_other_synset)=+1.152657`
- `C009` final:
  - `acc=0.317261`
  - majority `=0.037598`
  - `pred_top1_share=0.133911`
  - `pred_unique_codes=40`
  - `delta_ce(no_context)=+6.506371`
  - `delta_ce(wrong_shape_same_synset)=+4.143322`
  - `delta_ce(wrong_shape_other_synset)=+10.214767`
- source breakdown at `10k`:
  - `uniform acc=0.559713`
  - `near_surface acc=0.081809`

Operational interpretation:

- the post-freeze rerun confirms that `udf_distance` is a real surviving CQA
  task rather than a pre-freeze artifact.
- the 10k curve changes the branch read materially:
  shape-conditioned controls now move strongly, especially
  `wrong_shape_other_synset`.
- CQA is still not the paper mainline, but `udf_distance` is now a serious
  side branch rather than a purely provisional one.

## 163. Eval-only zero-shot `pc_bank -> udf_distance` transfer is positive on frozen `world_v3` (2026-03-16)

Lineage:

- `C010` ABCI job:
  - `113431`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml`
- save / result root:
  - `results/cqa_eval/patchnepa_cqa_udfdist_offdiag_20260316_171008`
- eval json:
  - `results/cqa_eval/patchnepa_cqa_udfdist_offdiag_20260316_171008/cqa_udfdist_offdiag_eval.json`

Summary:

- `correct` off-diagonal eval:
  - `acc=0.176636`
  - majority `=0.039673`
  - `pred_top1_share=0.075745`
  - `pred_unique_codes=39`
- controls:
  - `delta_ce(no_context)=+4.473030`
  - `delta_ce(wrong_shape_same_synset)=+1.857023`
  - `delta_ce(wrong_shape_other_synset)=+4.612063`
  - `delta_ce(shuffled_query)=+5.851544`
- source breakdown:
  - `uniform acc=0.316884`
  - `near_surface acc=0.033760`

Operational interpretation:

- the `surf`-trained `udf_distance` checkpoint transfers zero-shot to
  `pc_bank` at eval time only; this is the first clean off-diagonal CQA result.
- the signal remains concentrated in the easier broad query regime
  (`uniform` >> `near_surface`), but the transfer itself is clearly above
  majority and remains control-sensitive.
- paired `pc_bank -> udf_distance` training should stay diagnostic / upper
  bound only, because the zero-shot read is already positive.

## 164. Dense-grid `udf_distance` completion pilot is positive on held-out shapes (2026-03-16)

Lineage:

- `C011` ABCI job:
  - `113432`
- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml`
- result root:
  - `results/cqa_completion/patchnepa_cqa_udfdist_completion_20260316_171008`
- completion json:
  - `results/cqa_completion/patchnepa_cqa_udfdist_completion_20260316_171008/cqa_udfdist_completion_grid12.json`

Summary:

- pilot settings:
  - `16` held-out shapes
  - `grid_res=12`
- dense-grid metrics:
  - `MAE mean=0.047945 std=0.010271`
  - `RMSE mean=0.063767 std=0.013245`
  - `IoU@0.01 mean=0.081440`
  - `IoU@0.02 mean=0.184972`
  - `IoU@0.05 mean=0.571206`

Operational interpretation:

- `udf_distance` CQA now has method-native field completion evidence in
  addition to token/control metrics.
- this is still a modest pilot rather than a full mesh-level sweep, but it is
  already enough to show that the branch can decode into a meaningful dense UDF
  field on held-out shapes.

## 165. Seed-pack confirms `udf_distance` as a stable CQA branch and makes the paired `pc_bank` upper bound optional (2026-03-16)

Lineage:

- `C012` ABCI jobs:
  - mainline seed `1`: `113438`
  - off-diagonal seed `1`: `113439`
  - mainline seed `2`: `113440`
  - off-diagonal seed `2`: `113441`
- seed `1` mainline:
  - `runs/cqa/patchnepa_cqa_udfdist_seedpack_20260316_173222_curve_seed1/cqa_udfdist_worldv3_g2_s10000_seed1`
  - `runs/cqa/patchnepa_cqa_udfdist_seedpack_20260316_173222_curve_seed1/cqa_udfdist_worldv3_g2_s10000_seed1/eval_controls_128.json`
- seed `2` mainline:
  - `runs/cqa/patchnepa_cqa_udfdist_seedpack_20260316_173222_curve_seed2/cqa_udfdist_worldv3_g2_s10000_seed2`
  - `runs/cqa/patchnepa_cqa_udfdist_seedpack_20260316_173222_curve_seed2/cqa_udfdist_worldv3_g2_s10000_seed2/eval_controls_128.json`
- seed `1` off-diagonal:
  - `results/cqa_eval/patchnepa_cqa_udfdist_seedpack_20260316_173222_offdiag_seed1/cqa_udfdist_offdiag_eval_seed1.json`
- seed `2` off-diagonal:
  - `results/cqa_eval/patchnepa_cqa_udfdist_seedpack_20260316_173222_offdiag_seed2/cqa_udfdist_offdiag_eval_seed2.json`

Summary:

- mainline `surf -> udf_distance`:
  - seed `1`: `acc=0.333008`, majority `=0.038452`,
    `pred_top1_share=0.120972`, `pred_unique_codes=41`
  - seed `2`: `acc=0.344604`, majority `=0.038330`,
    `pred_top1_share=0.140381`, `pred_unique_codes=40`
- mainline controls:
  - seed `1`:
    - `delta_ce(no_context)=+8.345530`
    - `delta_ce(wrong_shape_same_synset)=+4.592689`
    - `delta_ce(wrong_shape_other_synset)=+10.969835`
  - seed `2`:
    - `delta_ce(no_context)=+6.250366`
    - `delta_ce(wrong_shape_same_synset)=+4.287287`
    - `delta_ce(wrong_shape_other_synset)=+10.770094`
- zero-shot off-diagonal `pc_bank -> udf_distance`:
  - seed `1`: `acc=0.158386`, majority `=0.040222`,
    `pred_top1_share=0.065369`, `pred_unique_codes=41`
  - seed `2`: `acc=0.161987`, majority `=0.037903`,
    `pred_top1_share=0.083374`, `pred_unique_codes=40`
- off-diagonal controls:
  - seed `1`:
    - `delta_ce(no_context)=+6.262180`
    - `delta_ce(wrong_shape_same_synset)=+1.763959`
    - `delta_ce(wrong_shape_other_synset)=+4.654159`
  - seed `2`:
    - `delta_ce(no_context)=+3.906184`
    - `delta_ce(wrong_shape_same_synset)=+1.681309`
    - `delta_ce(wrong_shape_other_synset)=+4.037548`

Operational interpretation:

- the seed pack confirms that the `udf_distance` read is stable rather than a
  seed-`0` artifact.
- zero-shot off-diagonal transfer to `pc_bank` is also stable enough that the
  paired `pc_bank` upper bound is no longer the next required job.
- paired `pc_bank -> udf_distance` remains a useful diagnostic if later
  reviewers ask for an upper bound, but it should not pre-empt the cleaner
  zero-shot story.

## 166. Paired `pc_bank -> udf_distance` upper bound improves over zero-shot but stays diagnostic (2026-03-16)

Lineage:

- `C013` ABCI job:
  - `113472`
- paired upper-bound root:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000`
- eval:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000/eval_controls_128.json`
- curve summary:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000/eval_curve_128/curve_summary.json`

Summary:

- paired `pc_bank -> udf_distance`:
  - `acc=0.253296`, majority `=0.037476`
  - `pred_top1_share=0.111206`, `pred_unique_codes=41`
- controls:
  - `delta_ce(no_context)=+5.598058`
  - `delta_ce(wrong_shape_same_synset)=+1.821988`
  - `delta_ce(wrong_shape_other_synset)=+4.571476`
  - `delta_ce(shuffled_query)=+7.894287`
- query-source breakdown:
  - `uniform acc=0.441712`
  - `near_surface acc=0.067620`

Operational interpretation:

- the task itself is clearly learnable with paired `pc_bank` training, and the
  result is stronger than the zero-shot off-diagonal checkpoint.
- this is useful as a diagnostic upper bound and reviewer-facing ceiling.
- the main scientific story should still center the zero-shot off-diagonal
  transfer, because that is the cleaner evidence for shared-context generality.

## 167. Ordered-query ablation does not rescue AR; shuffled queries were not the whole story (2026-03-21)

Lineage:

- ordered-AR train/eval chain:
  - train `114405`
  - offdiag `114406`
  - same completion `114407`
  - offdiag completion `114408`
- ordered-parallel train/eval chain:
  - train `114409`
  - offdiag `114410`
  - same completion `114411`
  - offdiag completion `114412`

Summary:

- ordered-AR controls:
  - same-context: `acc=0.327759`, `ce=2.216774`
  - off-diagonal: `acc=0.177246`, `ce=7.278724`
  - deltas:
    - same: `no_context=+20.404953`, `wrong_shape_same=+4.766537`,
      `wrong_shape_other=+11.878389`
    - offdiag: `no_context=+14.157351`, `wrong_shape_same=+3.208929`,
      `wrong_shape_other=+7.630381`
- ordered-parallel controls:
  - same-context: `acc=0.364502`, `ce=2.040040`
  - off-diagonal: `acc=0.216797`, `ce=7.217139`
  - deltas:
    - same: `no_context=+20.163294`, `wrong_shape_same=+6.045631`,
      `wrong_shape_other=+13.443542`
    - offdiag: `no_context=+14.439228`, `wrong_shape_same=+3.636963`,
      `wrong_shape_other=+8.093796`
- ordered completion means:
  - ordered-AR same/offdiag:
    - `MAE=0.060016 / 0.115406`
    - `RMSE=0.080399 / 0.163262`
    - `IoU@0.05=0.498508 / 0.331447`
    - `mesh_fscore=0.030321 / 0.028158`
  - ordered-parallel same/offdiag:
    - `MAE=0.029590 / 0.122251`
    - `RMSE=0.041486 / 0.193677`
    - `IoU@0.05=0.651563 / 0.392847`
    - `mesh_fscore=0.080819 / 0.058457`

Operational interpretation:

- turning off shuffling with a simple lexicographic XYZ order does **not**
  reverse the earlier `parallel > AR` read.
- ordered AR improves control sensitivity and slightly improves same-context
  token accuracy over shuffled AR, but it does not materially improve
  off-diagonal token accuracy and it worsens same-context completion.
- ordered parallel remains stronger than ordered AR, and the best overall CQA
  completion row still comes from shuffled-parallel.
- the current safe claim is therefore:
  - the main gain comes from the primitive-native **Q/A interface**,
  - AR is viable but not yet shown to be necessary,
  - shuffled queries are not the sole reason AR underperforms.

## 168. First `DISTANCE + NORMAL` shared checkpoint stays alive on both tasks; TYPE-switch assets land but are not yet paper-face (2026-03-23)

Lineage:

- shared train / suite:
  - train `115782`
  - suite `115783`
- TYPE-switch asset export:
  - `115784`

Summary:

- same-context:
  - `udf_distance`: `acc=0.376526`, majority `=0.037903`,
    `delta_ce(no_context)=+19.120546`
  - `mesh_normal`: `acc=0.318115`, majority `=0.160217`,
    `delta_ce(no_context)=+3.035598`
- off-diagonal (`pc_bank`):
  - `udf_distance`: `acc=0.195862`, majority `=0.039673`,
    `delta_ce(no_context)=+15.204853`
  - `mesh_normal`: `acc=0.211914`, majority `=0.156067`,
    `delta_ce(no_context)=+2.080836`
- TYPE-switch asset pack:
  - same frozen checkpoint and same `pc_bank` context can emit
    `context_points.ply`, `mesh_normal_{pred,gt}_points.ply`,
    `udf_distance_fields.npz`, and `udf_distance_{pred,gt_levelset}_mesh.obj`.

Operational interpretation:

- the first `DISTANCE + NORMAL` shared checkpoint passes the minimal scientific
  gate: both tasks are above majority and context-sensitive on same/offdiag.
- however, the qualitative `mesh_normal` outputs are still weak, so the
  TYPE-switch asset pack is technically valid but not yet a paper-face figure.

## 169. Continuous `udf_distance` remains viable without collapse, but discrete stays the canonical mainline (2026-03-23)

Lineage:

- train `115851`
- offdiag controls `115852`
- same completion `115853`
- offdiag completion `115854`

Summary:

- same controls:
  - `MAE=0.012248`
  - `RMSE=0.016697`
  - `IoU@0.05=0.879288`
  - `delta_mae(no_context)=+0.494854`
  - `delta_mae(wrong_shape_same)=+0.089588`
  - `delta_mae(wrong_shape_other)=+0.188842`
- same completion:
  - `MAE=0.013186`
  - `RMSE=0.017733`
  - `IoU@0.05=0.757485`
  - `mesh_chamfer_l2=0.004287`
  - `mesh_fscore=0.161704`
- offdiag completion:
  - `MAE=0.136251`
  - `RMSE=0.223132`
  - `IoU@0.05=0.404277`
  - `mesh_chamfer_l2=0.090904`
  - `mesh_fscore=0.086477`

Operational interpretation:

- continuous scalar regression for `udf_distance` does **not** collapse under
  the current independent CQA schema.
- same-context field metrics are strong, but off-diagonal and mesh-side reads
  are mixed relative to the discrete independent row.
- keep continuous target design as a serious ablation axis, not as the new
  canonical mainline.

## 170. Query-block ablation shows that full query-list conditioning helps only marginally (2026-03-23)

Lineage:

- `self_q` chain:
  - train `115858`
  - offdiag `115859`
  - same completion `115860`
  - offdiag completion `115861`
- `no_q` chain:
  - train `115862`
  - offdiag `115863`
  - same completion `115864`
  - offdiag completion `115865`
- baseline:
  - `full_q` = `C019`

Summary:

- same token accuracy:
  - `full_q=0.389771`
  - `self_q=0.377686`
  - `no_q=0.377197`
- offdiag token accuracy:
  - `full_q=0.223511`
  - `self_q=0.228577`
  - `no_q=0.215698`
- same completion (`MAE / IoU@0.05 / mesh_fscore`):
  - `full_q=0.020696 / 0.753730 / 0.170162`
  - `self_q=0.021140 / 0.734338 / 0.162727`
  - `no_q=0.021348 / 0.731425 / 0.160111`

Operational interpretation:

- `full_q` remains the best single row, but the gap to `self_q` and `no_q` is
  small.
- strong controls survive all three settings.
- the main gain comes from shared context plus per-query anchoring; the full
  query list is helpful but not the dominant source of performance.

## 171. Shared continuous `DISTANCE + NORMAL` keeps distance alive but fails on normals (2026-03-23)

Lineage:

- train `115972`
- same/offdiag suite `115973`

Summary:

- same-context:
  - `udf_distance`: `MAE=0.027435`, `RMSE=0.037976`, `IoU@0.05=0.740323`,
    `code_acc_proxy=0.234741` vs majority `0.037781`
  - `mesh_normal`: `mean_cos=0.002258`, `angle_deg=89.849701`,
    `code_acc_proxy=0.004883` vs majority `0.149109`
- off-diagonal:
  - `udf_distance`: `MAE=0.137208`, `RMSE=0.222924`, `IoU@0.05=0.416787`,
    `code_acc_proxy=0.137634` vs majority `0.038330`
  - `mesh_normal`: `mean_cos=0.008406`, `angle_deg=89.479912`,
    `code_acc_proxy=0.004517` vs majority `0.151794`

Operational interpretation:

- shared typed continuous regression can preserve `udf_distance`,
- but under the current `DISTANCE + NORMAL` recipe it does not learn
  `mesh_normal` in a useful way,
- so continuous target design remains task-specific for now rather than a
  shared multi-type replacement.

## 172. `DISTANCE + NORMAL_UNSIGNED` rescues the shared mesh branch (2026-03-24)

Lineage:

- train `116068`
- same/offdiag suite `116069`
- TYPE-switch export `116070`

Summary:

- same-context:
  - `udf_distance`: `acc=0.387695` vs majority `0.037903`
  - `mesh_normal_unsigned`: `acc=0.577332` vs majority `0.306641`
- off-diagonal:
  - `udf_distance`: `acc=0.200500` vs majority `0.039673`
  - `mesh_normal_unsigned`: `acc=0.383179` vs majority `0.306763`
- same `mesh_normal_unsigned` controls:
  - `delta_ce(no_context)=+2.827228`
  - `delta_ce(wrong_shape_same)=+0.883803`
  - `delta_ce(wrong_shape_other)=+2.904422`
- offdiag `mesh_normal_unsigned` controls:
  - `delta_ce(no_context)=+1.801807`
  - `delta_ce(wrong_shape_same)=+0.475568`
  - `delta_ce(wrong_shape_other)=+1.685428`
- TYPE-switch pack:
  - first exported shapes now show unsigned-normal `mean_cos` roughly
    `0.46-0.68`, clearly stronger than the prior signed-normal pack.

Operational interpretation:

- canonical hemisphere folding is enough to materially improve the shared
  `DISTANCE + NORMAL` line.
- the signed-normal weakness was therefore mostly target pathology rather than
  evidence against mesh-family prompting itself.
- `udf_distance` remains stable while `mesh_normal_unsigned` becomes a viable
  shared mesh task.

## 173. Unsigned normals rescue the shared continuous branch as well (2026-03-24)

Lineage:

- train `116094`
- same/offdiag suite `116095`

Summary:

- same-context:
  - `udf_distance`: `MAE=0.030454`, `RMSE=0.043400`, `IoU@0.05=0.734362`
  - `mesh_normal_unsigned`: `mean_cos=0.795400`, `angle_deg=27.850449`
- off-diagonal:
  - `udf_distance`: `MAE=0.122904`, `RMSE=0.196361`, `IoU@0.05=0.451584`
  - `mesh_normal_unsigned`: `mean_cos=0.682940`, `angle_deg=39.644447`
- same `mesh_normal_unsigned` controls:
  - `delta_mean_cos(no_context)=-0.314132`
  - `delta_mean_cos(wrong_shape_same)=-0.160481`
  - `delta_mean_cos(wrong_shape_other)=-0.334846`
- offdiag `mesh_normal_unsigned` controls:
  - `delta_mean_cos(no_context)=-0.206259`
  - `delta_mean_cos(wrong_shape_same)=-0.097945`
  - `delta_mean_cos(wrong_shape_other)=-0.219145`

Operational interpretation:

- signed-vs-unsigned is the decisive factor for continuous normals:
  the signed shared continuous row was nearly random, while unsigned
  continuous normals are geometrically strong on both same-context and
  off-diagonal evaluation.
- `udf_distance` remains alive at the same time.
- keep this as strong support for continuous target design, but the discrete
  unsigned shared line remains the safer canonical multi-type mainline.
