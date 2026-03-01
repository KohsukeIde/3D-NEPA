# Patch-NEPA Runlog (2026-02)

Last updated: 2026-03-01

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
