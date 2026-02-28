# Patch-NEPA Gap Audit (Query->Patch, End-to-End)

Last updated: 2026-02-28

## 1. Scope

This audit tracks parity and safety from preprocessing to fine-tune:

- pretrain parity target: `query_nepa` recipe (excluding `split/dual_mask/QA` family for now)
- fine-tune/eval parity target: `patchcls` line (not `query_nepa` classifier line)

## 2. Preprocessing / Data Contracts

### Required keys

- PatchNEPA pretrain (point-only baseline):
  - `pc_xyz`
  - `pc_fps_order` when `pt_sample_mode=fps`
- PatchCLS fine-tune/eval:
  - `pc_xyz`
  - `pc_fps_order` for strict FPS eval

### Current checks

- `ModelNet40QueryDataset` is fail-fast for incompatible `pt_dist` unless `ablate_point_dist=1`.
  - reference: `nepa3d/data/dataset.py`
- `PatchClsPointDataset` warns once if FPS is requested but `pc_fps_order` is missing.
  - reference: `nepa3d/data/cls_patch_dataset.py`

### Verified cache sanity (2026-02-28)

- `scanobjectnn_obj_bg_v3_nonorm`: `pc_xyz`/`pc_fps_order` present, ray pools present
- `scanobjectnn_obj_only_v3_nonorm`: `pc_xyz`/`pc_fps_order` present, ray pools present
- `scanobjectnn_pb_t50_rs_v3_nonorm`: `pc_xyz`/`pc_fps_order` present, ray pools present
- `shapenet_cache_v0` sampled files contain `pc_fps_order` and `pt_fps_order`

## 3. Pretrain Parity (Query baseline, excluding split/dual-mask/QA)

### 3.1 Aligned items

- DDP launch path enabled in patch launcher (`accelerate launch`, multi-node wrapper).
- Global batch is explicitly logged (`batch * num_processes * grad_accum`).
- Resume policy parity:
  - `auto_resume=1`
  - `resume_optimizer` supported
  - checkpoint includes `step`.
- Augmentation knobs parity (`aug_*`) added and propagated to dataset builder.
- Transformer support defaults aligned for `nepa2d` path:
  - `qk_norm_affine=0`
  - `qk_norm_bias=0`
  - `layerscale_value=1e-5`
  - `rope_theta=100`
- Scheduler parity default: `lr_scheduler=none`.
- Sampling default aligned to Query-style fixed recipe:
  - `pt_sample_mode=rfps` (launcher defaults, `pt_rfps_m=4096`).

### 3.2 Intentional differences (not yet ported)

- `split/interleave`, `dual_mask`, `QA-token` features.
- Query auxiliary heads/losses (`B2/B3/C/D/E`, teacher branch, cycle branch).
- Query sequence-length (`max_len`) constraints are not directly used in patch-token sequence yet.

### 3.3 Stability fixes already applied

- NEPA target branch stop-grad in patch loss (`target = z[:,1:,:].detach()`).
- Worker RNG initialization for `numpy`/`random` to avoid worker-order drift.

## 4. Fine-tune / Eval Parity (PatchNEPA -> PatchCLS)

Target is strict compatibility with `patchcls` training/eval settings.

### 4.1 Verified compatibility

- State-dict overlap between `PatchTransformerNepa` and `PatchTransformerClassifier` (same backbone recipe):
  - shared same-shape keys: 225
  - full coverage on:
    - `patch_embed.*`
    - `backbone.*`
    - `norm.*`
    - `cls_token`
  - expected unmatched pretrain-only keys:
    - `pred_head.weight`, `pred_head.bias`

### 4.2 Eval guardrails

- Hard guard in patchcls:
  - reject `mc_eval_k_test>1 && pt_sample_mode_eval=fps && aug_eval=0`
  - prevents fake MC/TTA with deterministic repeated crops.

### 4.3 Policy alignment

- Mainline split policy is fixed to `val_split_mode=file` only.
- `group_*` and `pointmae(test-as-val)` are treated as historical/reference modes and are excluded from new mainline comparisons.
- Avoid legacy `scanobjectnn_*_v2` caches unless explicitly allowed (`ALLOW_SCAN_UNISCALE_V2=1`).

## 5. Active Run References

- current parity-fixed pretrain run:
  - job: `99643.qjcm`
  - run_set: `patchnepa_pointonly_ddp8_rfpsfix_20260228_154649`
  - pbs log: `logs/ddp_patch_nepa_pretrain/patchnepa_pointonly_ddp8_rfpsfix_20260228_154649.pbs.log`
  - node logs:
    - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh138.patchnepa.log`
    - `logs/ddp_patch_nepa_pretrain/ddp_patchnepa_99643.qjcm_run_pointonly_patchnepa_pointonly_ddp8_rfpsfix_20260228_154649/logs/qh142.patchnepa.log`

## 6. Mandatory Preflight Checklist

Before accepting any run for comparison tables:

1. Launch topology in log matches intent (`num_processes`, `num_machines`, global batch).
2. Patch config in log matches recipe (`patch_embed`, `group_size`, `num_groups`).
3. Backbone support config in log matches recipe (`qk_norm*`, `layerscale`, `rope_theta`).
4. Data key fallback warnings are absent (especially FPS fallback warnings).
5. Resume behavior is explicit (`[resume] loaded=...` or disabled with reason).
6. Fine-tune eval uses valid MC/TTA setting (guard should pass cleanly).

## 7. Query-Shared Error Ledger (carry-over risks)

These are not Patch-specific bugs; they are shared operational risks from the Query line.
All should be checked in Patch-NEPA runs as well.

1. DDP topology mismatch
- Risk: allocated GPUs do not match actual training processes.
- Guard: always verify `num_processes`, `num_machines`, and logged global batch.

2. Wrapper env propagation mismatch
- Risk: wrong `WORKDIR`/empty env variables silently change recipe.
- Guard: validate launcher header lines (`workdir`, `mix_config`, key overrides).

3. Cached-order missing fallback (FPS/RFPS)
- Risk: on-the-fly sampling path slows jobs and breaks reproducibility assumptions.
- Guard: precheck cache keys and reject runs with fallback warnings.

4. Mixed pretrain sample-count drift
- Risk: effective number of samples differs from Query baseline; fairness is broken.
- Guard: enforce one-pass mix policy for parity experiments and log the exact config path.

5. Retry-time recipe drift
- Risk: scheduler/resume/sampling defaults differ across retries.
- Guard: include full recipe dump in every log and compare before accepting metrics.
