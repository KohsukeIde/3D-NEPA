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
