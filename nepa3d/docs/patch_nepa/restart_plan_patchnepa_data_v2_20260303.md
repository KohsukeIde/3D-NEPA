# Patch-NEPA Restart Plan (Data-v2 + CPAC Alignment)

Last updated: 2026-03-12

## 1. Purpose

This memo freezes the restart policy after the data semantics audit:

- Rebuild ShapeNet cache with explicit surface/query separation.
- Keep Patch-NEPA as the primary model.
- Make CPAC evaluable with the same Patch-NEPA checkpoint.
- Minimize redundant reruns by fixing obvious defaults first.

## 2. Confirmed Code Facts (Current Repo)

1. ShapeNet preprocessing currently reuses the ModelNet preprocessor.
- `nepa3d/data/preprocess_shapenet.py:10`
- `nepa3d/data/preprocess_shapenet.py:94`

2. `pt_xyz_pool` is a mixed pool (uniform-in-cube + near-surface jitter).
- `nepa3d/data/preprocess_modelnet40.py:433`
- `nepa3d/data/preprocess_modelnet40.py:445`

3. PatchNEPA answer embedding is tied to the same sampled point set via `group_idx` gather.
- `nepa3d/models/patch_nepa.py:89`
- `nepa3d/models/patch_nepa.py:819`

4. Current Patch pretrain path does not pass normals (`pt_n=None`).
- `nepa3d/train/pretrain_patch_nepa.py:711`

5. UDF backend explicitly marks ray modality unavailable.
- `nepa3d/backends/udfgrid_backend.py:9`
- `nepa3d/backends/udfgrid_backend.py:55`

6. PatchNEPA now has a mixed-modality safety path (dummy rays if missing).
- `nepa3d/models/patch_nepa.py:828`
- `nepa3d/models/patch_nepa.py:847`

## 3. Restart Defaults (Fix These First)

1. Rebuild data before any new score comparison.
2. Default sampling policy for reruns:
- non-cached RFPS (`pt_sample_mode=rfps`) for train-time point sampling.
- avoid `rfps_cached` for mainline comparisons.
3. Keep Patch ordering experiments separate from data-v2 bring-up.
4. Keep serial-order experiments off for now unless explicitly targeted.

## 4. Data-v2 Contract

Add explicit fields so training/eval semantics are unambiguous:

- `surf_xyz`: surface point set for patch tokenization.
- `surf_n` (optional but recommended): surface normals.
- `qry_xyz`: space query points for completion probe/CPAC.
- `qry_udf_dist`: UDF target at `qry_xyz`.
- `ray_o/ray_d/ray_t/ray_hit/ray_n` when ray modality is enabled.

Keep backward compatibility:

- retain existing keys (`pc_xyz`, `pt_xyz_pool`, `pt_dist_*`) during transition.
- write adapter logic in dataset/backend instead of breaking all old caches at once.

## 5. Modeling Rule (Prevent Shortcut Regressions)

1. Split "where" vs "what":
- Where: positional path (`center_mlp` / position embedding).
- What: Answer content embedding (primitive-specific signals).

2. Do not inject raw XYZ strongly into Answer content path by default.
- This reintroduces copy shortcuts in content-only training.

3. Primitive-specific Answer targets:
- Mesh: normal/geometry attributes (surface teacher).
- UDF: distance (and optional gradient) at query points.
- PointCloud: observation-side attributes only (avoid mesh leakage by default).

## 6. CPAC Alignment Strategy (Same Checkpoint Principle)

Shortest path:

1. Add token-input path to PatchNEPA (e.g., `forward_tokens(feat, type_id, ...)`).
2. Reuse `completion_cpac_udf.py` pipeline with PatchNEPA checkpoint loading.
3. Keep existing ridge + optional mesh metrics path unchanged.

Goal:

- One model family, one checkpoint lineage, both classification and CPAC/completion evaluation.

## 7. Execution Phases

Phase A: Data-v2 build and validation
- Implement v2 writer.
- Add cache sanity checker: key existence, shape, finite checks, modality flags.

Phase B: PatchNEPA token-path support
- Add token forward path.
- Add a smoke test with tiny cache split.

Phase C: Single-primitive reruns (minimal)
- PC-only
- Mesh-only
- UDF-only

Phase D: Mixed unpaired mainline
- PC + Mesh + UDF unpaired training
- CPAC/UCPR + downstream classification together

## 8. Minimal Ablation Set (Post v2)

Run only major-change axes first:

1. Data semantics
- old cache vs data-v2 cache

2. Sampling
- `rfps` vs `random` (both non-cached)

3. Ordering
- baseline order vs Morton-family order schedule

4. Query masking
- off vs selected best mask ratio (already screened)

5. Ray binding (if enabled)
- proxy binding vs independent binding

Everything else stays fixed until these complete.

## 9. Immediate TODO (Repository)

1. `nepa3d/data`: implement data-v2 writer and validation utility.
2. `nepa3d/backends`: add v2-aware loading (`surf_xyz`, `qry_xyz`, `qry_udf_dist`).
3. `nepa3d/models/patch_nepa.py`: add token-input forward path for CPAC compatibility.
4. `nepa3d/analysis/completion_cpac_udf.py`: add PatchNEPA model-loading/forward branch.
5. `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`: log all reruns as "data-v2 only".

## 10. Governance Note

From this point, results produced on old mixed `pt_xyz_pool` semantics should be treated as historical diagnostics, not final evidence for the main claim.

## 11. Decision Log (2026-03-03, v2 tokens)

Decision summary:

1. `ans_feat` fixed dimension is required in the current v2 token pipeline.
- Reason: `dataset_v2` + `v2_collate_fn` stack tensors into `[B, Nq, C]`, so `C` must be batch-consistent.
- Implementation path: add `answer_in_dim` support in PatchNEPA and keep schema-fixed packing in data layer.

2. This does **not** conflict with the additive modality-embedding direction.
- Current fixed `answer_in_dim` is a compatibility foundation for stable mixed training.
- Additive projection (`sum of modality-specific projections`) is a model-side upgrade on top of the same fixed-schema data contract.

3. Practical migration order is fixed:
- Phase 1 (now): fixed-schema `ans_feat` + `answer_in_dim` + token pretrain path.
- Phase 2 (next): optional `answer_embed_mode=additive` while preserving the same schema contract.

Implementation status (applied):

- `patch_nepa.py`: token API (`forward_tokens`) + explicit `answer_in_dim`.
- `mixed_pretrain.py`: v2 dataset routing (`dataset_version=v2`, key/prefix overrides).
- Added v2 files:
  - `data/dataset_v2.py`
  - `data/answer_feature_pack.py`
  - `data/preprocess_shapenet_v2.py`
  - `data/convert_npz_to_v2.py`
  - `train/pretrain_patch_nepa_tokens.py`
  - `configs/shapenet_unpaired_mix_v2_tokens.yaml`

## 12. Point-MAE Compatibility Check (2026-03-03)

Verified against Point-MAE code:

1. ShapeNet pretrain uses per-iteration random subsampling + `pc_norm`.
- `Point-MAE/datasets/ShapeNet55Dataset.py`

2. Pretrain augmentation uses `PointcloudScaleAndTranslate`.
- scale range: `[2/3, 3/2]`
- translate range: `[-0.2, 0.2]`
- `Point-MAE/datasets/data_transforms.py`
- `Point-MAE/tools/runner_pretrain.py`

Applied in this repo:

1. `V2SurfaceQueryDataset` now supports train-mode random sampling (non-deterministic per call).
- `nepa3d/data/dataset_v2.py`

2. Added optional Point-MAE-compatible preprocessing flags:
- `pointmae_pc_norm`
- `pointmae_scale_translate`
- `pointmae_scale_low/high`
- `pointmae_translate`
- `transform_answers` (keeps `ans_feat` roughly consistent under transforms)
- `nepa3d/data/dataset_v2.py`

3. Wired these controls through mixed loader and token pretrain entry:
- `nepa3d/data/mixed_pretrain.py`
- `nepa3d/train/pretrain_patch_nepa_tokens.py`

Recommended default for v2 token pretrain reruns:

- `--pm_pc_norm 1`
- `--pm_scale_translate 1`
- `--pm_scale_low 0.6666667 --pm_scale_high 1.5`
- `--pm_translate 0.2`
- `--pm_transform_answers 1`

## 13. Re-baseline Matrix (user-locked, data-v2 only) (2026-03-03)

Scope lock:

- All new comparisons use rebuilt ShapeNet v2 cache only.
- Historical `pt_xyz_pool`-based scores are diagnostic history, not mainline evidence.

Major-change criteria used for revalidation:

1. Data distribution changes.
2. Learning target changes.
3. Sequence/ordering changes.
4. Modality changes (ray/no-ray, assignment strategy).
5. Evaluation protocol changes (FT head/pooling/augmentation).

## 14. Architecture Update (Queryless v1-style Path) (2026-03-03)

Decision:

- Keep PatchNEPA v1-style `forward()` as the primary pretrain path.
- Remove dependency on token-query stream for mainline pretrain.
- Preserve Q/A index alignment through shared `group_idx` gather.

Applied changes:

1. `PatchTransformerNepa.forward()` now accepts generic per-point Answer features.
- Added args: `pt_ans_feat` / `points_ans_feat`.
- If provided, `pt_ans_feat` is used as Answer source (must align to `pt_xyz` shape and `answer_in_dim`).
- If not provided, legacy path (`pt_dist` / `pt_grad`) remains unchanged.
- File: `nepa3d/models/patch_nepa.py`

2. `pretrain_patch_nepa.py` now supports this path without breaking legacy jobs.
- Added CLI arg: `--answer_in_dim` (`0` keeps legacy auto behavior).
- Auto-probes dataset sample for `ans_feat`/`pt_ans_feat` when `--answer_in_dim=0`.
- Training loop now forwards:
  - `pt_xyz` from `batch['pt_xyz']` or `batch['surf_xyz']`.
  - `pt_ans_feat` from `batch['pt_ans_feat']` or `batch['ans_feat']` when length matches `pt_xyz`.
- On length mismatch, fails fast (`RuntimeError`) to preserve reproducibility.
- File: `nepa3d/train/pretrain_patch_nepa.py`

Compatibility note:

- Existing runs/scripts using only `pt_dist`/`pt_grad` are unchanged.
- New surf-aligned primitive-specific Answer features can be adopted incrementally.

Out of scope for this round:

- infra-only fixes (launcher/env/dependency/DDP timeout).
- `loss_target_mode=full_z` arm.
- ModelNet FT (deferred; ScanObjectNN first).
- patch-order schedule (`epoch/batch/sample`) sweep.

Fixed defaults for this round:

1. Pretrain objective:
- `loss_target_mode=content_tokens` only.

2. Pretrain augmentation and normalization:
- Point-MAE style path fixed on:
  - `pc_norm` (centroid shift + max-radius normalize).
  - `PointcloudScaleAndTranslate` equivalent (`scale=[2/3, 3/2]`, `translate=0.2`).
  - answer feature transform correction enabled (approximate).

3. Sampling/capacity protocol:
- 300 epochs.
- effective global batch 128 (`16 GPU x per-proc batch 8 x grad_accum 1` or equivalent).
- `pt_sample_mode` non-cached only (`rfps`/`random` arms).
- local patch encoder default: `patch_local_encoder=pointmae_conv`.
- randomized patch FPS seed default: `patch_fps_random_start=1`.

4. Finetune protocol (ScanObjectNN):
- `n_point=2048` fixed.
- train-time augmentation: `PointcloudScaleAndTranslate` for:
  - `obj_bg`
  - `obj_only`
  - `pb_t50_rs`
  - `hardest`

5. Logging policy:
- `USE_WANDB=1` required.
- pretrain project: `patchnepa-pretrain`
- finetune project: `patchnepa-finetune`

Revalidation axes (required):

1. `qa_layout`:
- `split_sep` vs `interleave`.

2. `point_order_mode`:
- `morton` (fixed)
- `random`
- `sample:<6-view morton perms>`
- `sample:<12-view morton perms + reverse>`

3. `pt_sample_mode`:
- `rfps` vs `random` (both non-cached).

4. Data distribution composition:
- `pc-only (100%)`
- `mesh+udf (50/50, no pc)`
- `pc+mesh+udf (33/33/33)`

Composition implementation rule:

- For this axis, composition is controlled by split cardinality, not sampler weights.
- Keep `replacement=false` so each epoch is one-pass over the materialized set.
- Note: in current `MixtureSampler`, weights are ignored when `replacement=false`; therefore exact ratios must be encoded in split JSON/materialized counts.
- `shapenet_unpaired_split.py` now supports strict ratio mode via:
  - `--allow_empty_splits 1`
  - required for `pc-only` and `mesh+udf` (otherwise per-synset minimum-1 logic injects unwanted splits).

Recommended execution order:

1. Pretrain-only screening (same FT-off protocol, compare by pretrain metrics first):
- `loss`, `cos_prev`, `cos_pred`, `gap`, `copy_win`.

2. Promote only top arms to full ScanObjectNN FT (all 4 variants above).

3. Run modality/ray-axis checks after ordering/layout baseline is fixed on v2.

## 14. Applied Now: token_qa_layout + mini-CPAC Screening Hooks

Only the two requested operational hooks were added:

1. token_qa_layout experiment injection path
- New pretrain launcher (token path):
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
- New submit wrapper with env passthrough for `TOKEN_QA_LAYOUT`:
  - `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
- Example:
  - `TOKEN_QA_LAYOUT=interleave bash scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - `TOKEN_QA_LAYOUT=split_sep bash scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`

2. mini-CPAC integration for screening
- New PatchNEPA CPAC job script (supports mini defaults):
  - `scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh`
- New chain submitter:
  - `scripts/sanity/submit_patchnepa_tokens_layout_screen_cpac_qf.sh`
- The chain submitter does:
  - submit token pretrain per layout
  - submit mini-CPAC with `afterok:<pretrain_job>`
  - write a TSV manifest of pretrain/cpac job IDs and ckpt paths

Default mini-CPAC profile in the chain:
- `MAX_TRAIN_SHAPES=64`
- `MAX_EVAL_SHAPES=64`
- `N_CTX_POINTS=1024`
- `N_QUERY=1024`
- `CHUNK_N_QUERY=1024`

## 15. Applied Now: Strict Surf-Answer Regeneration (UDF-Grid Sphere Tracing, No-Ray, 16-shard)

Decision:

- Do **append update** on `shapenet_cache_v2_20260303` (not full rewrite).
- Keep existing v2 keys, add surf-aligned primitive-native answer keys.
- No ray recomputation (`N_RAYS=0` in this run); existing ray arrays remain but are not used.
- UDF strict features are computed by sphere tracing on an unsigned distance grid from normalized surface occupancy.

Code changes:

- `nepa3d/data/preprocess_shapenet_v2.py`
  - Added strict surf-answer features:
    - `mesh_surf_n`, `mesh_surf_curv`
    - `udf_surf_t_in`, `udf_surf_t_out`, `udf_surf_hit_out`, `udf_surf_thickness`
    - `pc_n`, `pc_density` (full context-aligned arrays)
  - Added append mode:
    - `--augment_existing` updates existing NPZ in place.
    - `--skip_existing` now skips only when required surf keys already exist.
  - Added strict UDF-grid controls:
    - `--strict_udf_surface`
    - `--surf_udf_grid`, `--surf_udf_dilate`
    - `--surf_udf_max_t`, `--surf_udf_eps`
    - `--surf_udf_steps`, `--surf_udf_tol`, `--surf_udf_min_step`
  - Ray generation now safely supports `n_rays=0`.

- `nepa3d/data/dataset_v2.py`
  - Added queryless-aligned answer path:
    - `return_pt_ans`, `pt_answer_prefix`, `pt_answer_key`
    - Emits `pt_xyz` and `pt_ans_feat` aligned by the same sampled indices.

- `nepa3d/data/mixed_pretrain.py`
  - Added v2 routing knobs:
    - `return_pt_ans`, `pt_answer_prefix`, `pt_answer_key`
    - per-dataset override support via mix config extras.

- `scripts/preprocess/preprocess_shapenet_v2.sh`
- `scripts/preprocess/submit_preprocess_shapenet_v2_qf.sh`
  - Added env passthrough for strict surf-answer append run.

Submitted jobs (active chain, 2026-03-03):

- preprocess 16-shard append run:
  - `102118` .. `102133` (`shpv2g_s00` .. `shpv2g_s15`)
- dependency chain:
  - split: `102134` (afterok on all 16 preprocess jobs)
  - materialize: `102135` (afterok:`102134`)

## 16. Applied Now: Primitive-Conditioned Q/A Type IDs (default on)

Motivation:

- In mixed pretrain, `mesh` and `udf` can share similar surface context (`surf_xyz`) but require different answer spaces.
- Added explicit primitive signal in `type_id` to avoid ambiguous supervision.

Changes:

- New token types appended in `nepa3d/token/tokenizer.py`:
  - `TYPE_Q_POINT_MESH=10`, `TYPE_A_POINT_MESH=11`
  - `TYPE_Q_POINT_UDF=12`, `TYPE_A_POINT_UDF=13`
  - `TYPE_Q_POINT_PC=14`, `TYPE_A_POINT_PC=15`
  - `TYPE_VOCAB_SIZE=16`
- `PatchTransformerNepa.forward(..., primitive=...)` now maps per-sample primitive label to Q/A point types.
- `build_mixed_pretrain()` now sets `V2SurfaceQueryDataset(primitive_label=<backend>)`, so batch primitive labels are reliable (`pc/mesh/udf`).
- `pretrain_patch_nepa.py` now passes `primitive=batch['primitive']` into model forward.
- Masks were extended so new types are treated consistently:
  - query masks (`q_mask`, pooling/query selection),
  - NEPA loss answer/context masks,
  - causal dual-mask type-aware query-like mask.

Compatibility:

- Legacy ids (`TYPE_Q_POINT`, `TYPE_A_POINT`, etc.) remain unchanged.
- Existing checkpoints remain loadable with `strict=False` adaptation path already used in this codebase.

## 17. Applied Now: Split/Materialize from Current Cache (Drop 1 Missing Sample)

Reason:

- One pathological mesh remained missing in strict regen (`04530566/c79c87dcf17a40f698501ff072a5cd78`).
- To unblock training, split/materialize were generated from current cache contents (existing files only).

Execution (2026-03-03):

- Stopped remaining strict single-item retry job (`102320`, then finished as `F`).
- Built split JSON:
  - `data/shapenet_unpaired_splits_v2_20260303_drop1.json`
- Materialized unpaired cache:
  - `data/shapenet_unpaired_cache_v2_20260303_drop1`
  - link mode: `symlink`, overwrite: enabled

Observed output:

- split counts:
  - `train_mesh=16127`
  - `train_pc=15655`
  - `train_udf=15663`
  - `eval=5357`
- materialize:
  - `created=52802`
  - `missing=0` (the missing sample is excluded at split stage because source NPZ does not exist)

## 18. Applied Now: `content_plus_center` Target (order-only fix was insufficient)

Background:

- In v2 token runs (`morton`/`random`/`fps`), `cos_tgt` and `cos_prev` remained nearly overlapped and `copy_win` stayed high.
- This suggested order change alone is not enough; target needs additional non-trivial spatial signal without reintroducing `pos/type` leakage.

Changes:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - Added `loss_target_mode=content_plus_center`.
  - Added `--center_target_alpha` (default `0.5`).
  - Target definitions now:
    - `full_z`: `target = out.z`
    - `content_tokens`: `target = out.tokens`
    - `content_plus_center`: `target = out.tokens + alpha * center_mlp(out.centers_xyz)`
  - Added startup print for `loss_target_mode` and `center_target_alpha`.

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - Added `CENTER_TARGET_ALPHA` env passthrough and CLI forwarding.
  - Startup log now records `center_target_alpha`.

- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - Added `CENTER_TARGET_ALPHA` to qsub environment forwarding.

Job handling:

- user-requested termination of all currently running token-pretrain jobs:
  - `102512`, `102513`, `102514`, `102515`
- submitted first validation run under new target:
  - `102516` (`tok_m50fps_cpc`)
  - config: `mesh50_udf50`, `patch_order_mode=fps`, `token_qa_layout=split_sep`,
    `answer_in_dim=9`, `loss_target_mode=content_plus_center`, `center_target_alpha=0.5`

## 19. Applied Now: Intra-shape Var/Cov Regularization Scope (Fix scope mismatch)

Issue confirmed from logs:

- Batch-scope var/cov (`(B*T,D)` flatten) improved global dispersion but did not reduce
  intra-shape copy tendency (`cos_prev > cos_tgt` remained).
- This is a scope mismatch: diagnosis and failure mode are intra-shape, but regularizer was batch-global.

Changes:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - Added `--reg_scope {batch,intra_shape}` (default: `intra_shape`).
  - Added intra-shape regularizer:
    - per sample, over token axis (`(T,D)`),
    - var floor + off-diagonal covariance penalty,
    - averaged across valid samples in batch.
  - Added `_regularization_loss(...)` dispatcher for `batch` / `intra_shape`.
  - Startup log now prints regularization scope.

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - Added env passthrough: `REG_SCOPE` (default `intra_shape`).

- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - Added qsub env passthrough for `REG_SCOPE`.

Job status:

- Prior batch-scope run `102517` (`mesh50udf50_fps_splitsep_varcov`) was terminated.
- New run submitted with intra-shape scope:
  - `102518` (`mesh50udf50_fps_splitsep_varcov_intra`)
  - key params: `mesh50_udf50`, `fps`, `split_sep`, `content_plus_center(alpha=0.5)`,
    `reg_var_weight=0.05`, `reg_cov_weight=0.01`, `reg_scope=intra_shape`.

## 20. Applied Now: PointGPT-like Dual Mask Options (tokens path)

Added options for dual masking behavior parity experiments:

- `dual_mask_mode`:
  - `element` (existing behavior; per-edge Bernoulli drop)
  - `column` (PointGPT-like; sampled key columns are dropped for all rows)
- `dual_mask_keep_prefix`:
  - in `column` mode, first K key positions are always visible
- `dual_mask_warmup_frac`:
  - linear ramp of `dual_mask_near/far` from 0 to target over the first frac of steps

Implementation updates:

- `nepa3d/models/causal_transformer.py`
  - forward signature now accepts `dual_mask_mode`, `dual_mask_keep_prefix`
  - `column` mode implemented inside dual-mask branch
- `nepa3d/models/patch_nepa.py`
  - `forward` / `forward_tokens` now pass new dual-mask args through backbone
- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - new CLI args: `--dual_mask_mode`, `--dual_mask_keep_prefix`, `--dual_mask_warmup_frac`
  - warmup schedule applied per step to `dual_mask_near/far`
- qsub scripts:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - env passthrough added for the new args

## 21. Latent Branch Consolidated Snapshot (2026-03-04 04:00 JST)

Scope:

- `mesh50+udf50` tokens path diagnostics around copy tendency:
  - `loss`, `cos_tgt`, `cos_prev`, `gap`, `copy_win`
- all metrics below are late-window means (last 10% of logged steps for each run)
- mixed horizons are retained as-is; this is trend-level comparison, not final ranking

### 21.1 Key runs and latest trends

| job | run/log | step_reached | late loss | late cos_tgt | late cos_prev | late gap | late copy_win | max spike |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `102515` | `tok_m50fps.log` | 1394 | `0.5307` | `0.4693` | `0.4719` | `-0.0025` | `0.7538` | `0.0260` |
| `102516` | `tok_m50fps_cpc.log` | 3805 | `0.1904` | `0.8096` | `0.8111` | `-0.0015` | `0.5044` | `0.0691` |
| `102517` | `mesh50udf50_fps_splitsep_varcov.log` | 1860 | `0.1275` | `0.8752` | `0.8800` | `-0.0048` | `0.5181` | `0.0675` |
| `102518` | `mesh50udf50_fps_splitsep_varcov_intra.log` | 10000 | `0.4229` | `0.5788` | `0.5788` | `-0.0000` | `0.5009` | `0.1107` |
| `102519` | `mesh50udf50_fps_splitsep_varcov_intra_dualcol.log` | 8735* | `0.4224` | `0.5793` | `0.5793` | `-0.0000` | `0.5006` | `0.1107` |
| `102521` | `mesh50udf50_fps_splitsep_varcov_intra_hreg.log` | 7365* | `0.3765` | `0.6249` | `0.6111` | `+0.0138` | `0.4739` | `0.1298` |
| `102522` | `mesh50udf50_fps_splitsep_hreg_dcol_nf03.log` | 3970* | `0.4519` | `0.5509` | `0.5521` | `-0.0011` | `0.5026` | `0.1189` |

`*` running at snapshot time (`qstat`): `102519`, `102521`, `102522`.

### 21.2 Interpretation at this checkpoint

1. `cpc + intra var/cov` without hidden source (`102516/102518/102519`) did not create stable positive gap.
   - `gap` is effectively `~0` and `copy_win ~ 0.50`.
2. Hidden-source regularization (`102521`) is the first branch with a clearly positive late `gap` (`+0.0138`) and `copy_win < 0.50`.
   - This is the only currently running branch that breaks the `cos_tgt ≈ cos_prev` pattern in a useful direction.
3. Adding `dual_mask(column, near/far=0.3)` on top of hidden reg (`102522`) currently weakens that improvement.
   - `gap` returned near zero and `copy_win` went back to `~0.50`.
4. Earlier morton/random primitive-fix-only runs (`102508` and `102512` families) stayed copy-dominant (`copy_win ~0.74-0.75`) despite order changes.

### 21.3 Practical decision from latent branch

- Keep `morton/fps/random` order search as secondary until objective-side lift is secured.
- Prioritize hidden-source regularization branch for short FT sanity before adding more masking complexity.

## 22. Applied Now: Reconstruction Objective Path (`recon_mse` first, `recon_chamfer` optional)

Requested policy:

1. First verify wiring with `recon_mse` (no Chamfer dependency).
2. Then enable `recon_chamfer` when needed.

Code updates:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - Added objective switch:
    - `--pretrain_objective nepa_cosine|recon_mse|recon_chamfer`
  - Added reconstruction heads:
    - context patch head (`D -> group_size*3`)
    - query xyz head (`D -> 3`)
    - answer feature head (`D -> answer_in_dim`)
  - Added reconstruction loss weights:
    - `--recon_ctx_weight`, `--recon_q_weight`, `--recon_a_weight`
  - Added Chamfer selector:
    - `--recon_chamfer_metric l1|l2`
  - Added sequence metadata path (`ctx_group_idx`, layout-aware index mapping) so reconstruction targets align with causal next-token positions.
  - Checkpoint payload now stores `recon_heads` weights.

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - Added env/CLI forwarding for:
    - `PRETRAIN_OBJECTIVE`
    - `RECON_CTX_WEIGHT`
    - `RECON_Q_WEIGHT`
    - `RECON_A_WEIGHT`
    - `RECON_CHAMFER_METRIC`

- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - Added qsub env passthrough for all reconstruction objective knobs above.

Compatibility:

- Default remains `--pretrain_objective=nepa_cosine` (existing latent runs unchanged).
- `recon_chamfer` uses Point-MAE extension loader (`Point-MAE/extensions/chamfer_dist`) and fails fast if extension is unavailable.

Smoke submission (wiring check):

- submitted: `102523` (`pntok_rms`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_smoke_20260304_040234/`
- startup log confirms objective routing:
  - `pretrain_objective=recon_mse`
  - `recon_w=(ctx=1.0,q=1.0,a=1.0)`

## 23. Recheck (2026-03-04 04:50 JST): `recon_mse` wiring + latent all-log snapshot

### 23.1 `recon_mse` wiring status (`102523`)

- job is running (`qstat`): `102523` (`pntok_rms`)
- current step observed: `954 / 2000`
- no runtime errors found in pbs log (`Traceback/RuntimeError/NaN` none).
- new reconstruction terms are actively logged every step:
  - `loss_main`
  - `loss_recon_ctx`
  - `loss_recon_q`
  - `loss_recon_a`

Latest sample around step `954`:

- `loss_total=0.2728`
- `loss_recon_ctx~0.00x`, `loss_recon_q~0.1-0.3`, `loss_recon_a~0.11-0.12`
- confirms all three reconstruction branches are connected and contributing.

### 23.2 Latent diagnostics: full snapshot recorded

To avoid cherry-picking, all token-pretrain pbs logs with step diagnostics were re-parsed into one table:

- `nepa3d/docs/patch_nepa/latent_diag_snapshot_20260304.tsv`
  - rows: `24` logs
  - columns:
    - `log_path`
    - `step_reached`
    - `tail_loss`
    - `tail_cos_tgt`
    - `tail_cos_prev`
    - `tail_gap`
    - `tail_copy_win`
    - `loss_spike_max_abs`

This file is now the canonical “all latent runs” checkpoint for comparison.

## 24. `recon_chamfer` smoke submitted (2026-03-04 04:11 JST)

User-approved next step (`recon_mse` wiring verified -> enable `recon_chamfer`) has been submitted.

- job: `102524` (`pntok_rcf`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041040/`
- pbs log:
  - `.../reconch_m50u50_smoke.pbs.log`

Submission settings:

- `PRETRAIN_OBJECTIVE=recon_chamfer`
- `RECON_CHAMFER_METRIC=l2`
- `MIX_CONFIG=nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `TOKEN_QA_LAYOUT=split_sep`
- `PATCH_ORDER_MODE=morton`
- `MAX_STEPS=2000` (smoke)
- `ANSWER_IN_DIM=9`

Startup banner confirms:

- `objective: pretrain_objective=recon_chamfer ... chamfer_metric=l2`

### 24.1 Fix applied after first attempt

First `recon_chamfer` submission (`102524`) failed immediately because `extensions/chamfer_dist` imported `chamfer` as absolute module while the extension was not built/visible in runtime path.

Applied fixes:

1. Built Point-MAE chamfer extension in-place with multi-arch CUDA codegen:
   - `TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0`
   - `Point-MAE/extensions/chamfer_dist/setup.py build_ext --inplace`
2. Added robust path injection in loader:
   - `nepa3d/train/pretrain_patch_nepa_tokens.py::_load_chamfer_module`
   - now prepends both:
     - `Point-MAE/`
     - `Point-MAE/extensions/chamfer_dist/`

Resubmitted:

- job: `102526` (`pntok_rcf`)
- run root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041727/`

Runtime check:

- no `Traceback` / `ModuleNotFoundError`
- no `no kernel image is available` error
- step logs progressing with nonzero `loss_recon_ctx` (Chamfer branch active)

## 25. Full run launch from best latent branch (morton fixed, drop1 x3 + FT x9) (2026-03-04 04:28 JST)

Goal:

- Pick one strongest latent recipe from completed diagnostics, then run full pretrain+FT matrix on:
  - `pc100`
  - `mesh50+udf50`
  - `pc33+mesh33+udf33`
- with fixed order: `PATCH_ORDER_MODE=morton`.

Selected recipe (from latent branch trend):

- base branch: hidden-source var/cov branch (same direction as `102521` improvements)
- fixed knobs:
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
  - `N_SURF=2048`, `N_QRY=1024`, `N_RAY=0`
  - `MAX_STEPS=10000`
  - Point-MAE compatible transform: on (`pm_pc_norm=1`, `pm_scale_translate=1`)

Submitted pretrain jobs:

- `102527.qjcm` (`ptkpc`): `pc100`
- `102531.qjcm` (`ptkmu`): `mesh50+udf50`
- `102532.qjcm` (`ptkpm`): `pc33+mesh33+udf33`

Submitted FT jobs (afterok dependency from each pretrain, 3 variants each):

- from `102527` (`pc100`):
  - `102528` (`obj_bg`), `102529` (`obj_only`), `102530` (`pb_t50_rs`)
- from `102531` (`mesh50+udf50`):
  - `102533` (`obj_bg`), `102534` (`obj_only`), `102535` (`pb_t50_rs`)
- from `102532` (`pc33+mesh33+udf33`):
  - `102536` (`obj_bg`), `102537` (`obj_only`), `102538` (`pb_t50_rs`)

Notes:

- Total chain size is `12 jobs` (`pretrain 3 + ft 9`).
- FT submission used direct `qsub` with minimal `-v` env payload because
  `scripts/sanity/submit_patchnepa_finetune_variants_qf.sh` hit
  `qsub: cannot send environment with the job` on this host.
- canonical submission record:
  - `logs/sanity/patchnepa_submit/patchnepa_tokens_fullbest_morton_20260304_042155/submitted_jobs.txt`

## 26. FT switched to short protocol for quick TEST readout (2026-03-04 04:37 JST)

Reason:

- Goal is to get first TEST accuracy quickly under the same v1-style screening flow.
- Keep pretrain as submitted (`max_steps=10000`), shorten FT to `EPOCHS=60`.

Actions:

- cancelled old queued FT jobs (`102528`-`102530`, `102533`-`102538`).
- resubmitted FT chains with `EPOCHS=60` (same checkpoints / same dependencies):
  - from `102527` (`pc100`): `102539`, `102540`, `102541`
  - from `102531` (`mesh50+udf50`): `102542`, `102543`, `102544`
  - from `102532` (`pc33+mesh33+udf33`): `102545`, `102546`, `102547`

Record:

- `logs/sanity/patchnepa_submit/patchnepa_tokens_fullbest_morton_20260304_042155/submitted_jobs_shortft.txt`

## 27. v1-aligned full300 relaunch (16GPU pretrain, 300ep FT) and DDP fix (2026-03-04 04:50 JST)

### 27.1 First full300 chain (`patchnepa_tokens_full300_morton_20260304_044214`) failed immediately

Pretrain jobs:

- `102560` (`pc100`)
- `102561` (`mesh50udf50`)
- `102562` (`pc33mesh33udf33`)

Failure:

- all pretrains finished with `Exit_status=97`.
- per-node traceback showed DDP wrapper attribute access bug:
  - `AttributeError: 'DistributedDataParallel' object has no attribute 'encode_patches'`

FT jobs (`102563`-`102571`) stayed in dependency-fail hold (`Hold_Types=s`).

### 27.2 Code fix #1 (tokens pretrain)

File:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`

Change:

- in train loop, `build_qa_sequence(...)` now receives `model_raw = accelerator.unwrap_model(model)` instead of wrapped `model`.

### 27.3 Second full300 chain (`patchnepa_tokens_full300_morton_fixddp_20260304_044633`) exposed next DDP access bug

Pretrain jobs:

- `102572`, `102573`, `102574`

New failure traceback:

- `AttributeError: 'DistributedDataParallel' object has no attribute 'forward_tokens'`

### 27.4 Code fix #2 (tokens pretrain)

File:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`

Changes:

- added `_forward_tokens_wrapped(...)` helper.
- helper dispatches `forward_tokens` for both wrapped and unwrapped models.
- train loop now calls `_forward_tokens_wrapped(...)` instead of `model.forward_tokens(...)`.

### 27.5 Current active full300 chain (after both fixes)

Run set:

- `patchnepa_tokens_full300_morton_fixddp2_20260304_044851`

Submitted pretrain jobs (16GPU: 4 nodes x 4 procs, v1-style DDP):

- `102585` (`pc100`)
- `102586` (`mesh50udf50`)
- `102587` (`pc33mesh33udf33`)

Submitted FT jobs (300ep, afterok):

- `102588`-`102596` (3 variants x 3 arms)

Queue snapshot at submission:

- pretrain `R`: `102585`,`102586`,`102587`
- FT `H` (afterok): `102588`-`102596`

Records:

- `logs/sanity/patchnepa_submit/patchnepa_tokens_full300_morton_fixddp2_20260304_044851/submitted_jobs.txt`


### 27.6 Progress check (2026-03-04 10:2x JST)

Status snapshot:

- pretrain `102585` (`pc100`): `F/Exit_status=0` (done)
- pretrain `102586` (`mesh50udf50`): `F/Exit_status=0` (done)
- pretrain `102587` (`pc33mesh33udf33`): `R` (running)

Confirmed epoch math from logs:

- `102585`:
  - `steps_per_epoch=122`
  - `effective_max_steps=36600`
  - final reached `[step 036600]` and `[done] saved .../tok_pc100_morton`
- `102586`:
  - `steps_per_epoch=248`
  - `effective_max_steps=74400`
  - final reached `[step 074400]` and `[done] saved .../tok_mesh50udf50_morton`
- `102587`:
  - `steps_per_epoch=370`
  - `effective_max_steps=111000`
  - currently around `step ~81k` (about epoch ~220/300), so FT for this arm remains hold.

Partial FT results (completed so far):

- from `pc100`:
  - `obj_bg`: `TEST acc=0.8244`
  - `obj_only`: `TEST acc=0.8296`
  - `pb_t50_rs`: `TEST acc=0.7946`
- from `mesh50udf50`:
  - `obj_bg`: `TEST acc=0.8090`
  - `obj_only`: `TEST acc=0.8296`
  - `pb_t50_rs`: running (`102593`)
- from `pc33mesh33udf33`:
  - all FT jobs `Hold_Types=s` (waiting `afterok:102587`)

## 28. Default policy update: fixed full-ShapeNet cardinality across ratio arms (2026-03-04)

Decision (user-requested, now enforced):

- ratio (`pc100` / `mesh50udf50` / `pc33mesh33udf33`) may change,
- but per-epoch sample count must stay at full ShapeNet train cardinality.

Applied to drop1 token mix configs:

- `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml`
- `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33.yaml`
- `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh100.yaml`
- `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_udf100.yaml`

All above now use:

- `replacement: true`
- `mix_num_samples: 47445` (drop1 train total = `15655 + 16127 + 15663`)

Compute-matched controls remain separate (intentionally unchanged):

- `*_cm15655.yaml` (`replacement=false`, `mix_num_samples=15655`)

Launcher defaults also updated so accidental non-full runs are avoided:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - default `MIX_CONFIG` -> `...drop1_pc33_mesh33_udf33.yaml`
  - default `TOKEN_QA_LAYOUT` -> `split_sep`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - same default changes in qsub env payload

## 29. Reconstruction objective status (`recon_mse` / `recon_chamfer`) (2026-03-04)

### 29.1 `recon_mse` smoke (`102523`) — completed

Log:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_smoke_20260304_040234/recon_m50u50_smoke.log`

Final step (`2000/2000`):

- `loss_total=0.296183`
- `loss_recon_ctx=0.003766`
- `loss_recon_q=0.173864`
- `loss_recon_a=0.118553`
- `cos_tgt=-0.0183`
- `cos_prev=-0.0186`
- `gap=+0.0004`
- `copy_win=0.7397`

Run completed with `[done]` marker and checkpoint save.

### 29.2 `recon_chamfer` smoke (`102526`) — partial progress only

Log:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_smoke_20260304_041727/reconch_m50u50_smoke.log`

Last observed step:

- `step=969`
- `loss_total=0.287651`
- `loss_recon_ctx=0.003372`
- `loss_recon_q=0.166045`
- `loss_recon_a=0.118234`
- `cos_tgt=-0.0053`
- `cos_prev=-0.0058`
- `gap=+0.0005`
- `copy_win=0.7381`

No `[done]` marker in this log (smoke run not completed to max steps in the recorded file).

## 30. Full300+FT outputs recorded so far and interpretation boundary

From run set:

- `logs/sanity/patchnepa_submit/patchnepa_tokens_full300_morton_fixddp2_20260304_044851/`

Observed TEST outputs:

- `pc100`:
  - `obj_bg=0.8244`
  - `obj_only=0.8296`
  - `pb_t50_rs=0.7946`
- `mesh50udf50`:
  - `obj_bg=0.8090`
  - `obj_only=0.8296`
  - `pb_t50_rs`: no TEST line in current log snapshot

Important boundary:

- these were produced before enforcing "same total samples per ratio arm" as the default.
- therefore they are kept as historical diagnostics, not final fair-comparison evidence for ratio effects.

## 31. Recon diagnostics aligned with objective space (2026-03-04)

Issue:

- For `pretrain_objective=recon_mse/recon_chamfer`, legacy `cos_tgt/cos_prev/copy_win`
  are cosine-space probes and are not objective-aligned.

Applied fix (`nepa3d/train/pretrain_patch_nepa_tokens.py`):

- keep legacy diag only for `nepa_cosine`.
- for `recon_*`, log objective-aligned diagnostics:
  - `diag/recon_ctx_err`, `diag/recon_q_err`, `diag/recon_a_err`
  - `diag/copy_ctx_err`, `diag/copy_q_err`, `diag/copy_a_err`
  - `diag/recon_lift_ctx`, `diag/recon_lift_q`, `diag/recon_lift_a`

Definition:

- `copy_*_err` = error of "previous target copied as current prediction".
- `recon_lift_* = copy_*_err - recon_*_err` (positive means better than copy baseline).

## 32. Short re-run (`recon_mse` / `recon_chamfer`) with diagfix (2026-03-04)

Submitted:

- `105287.qjcm` (`reconmse_m50u50_short_fix`)
- `105288.qjcm` (`reconchamfer_m50u50_short_fix`)

Run roots:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconmse_diagfix_20260304_1835/`
- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_reconchamfer_diagfix_20260304_1835/`

Status:

- both completed to `step=2000` with `[done]` marker.

Result summary (from mr0 logs / W&B summary):

- `recon_mse`
  - final: `diag/recon_lift_q=0.2626`, `diag/recon_lift_a=0.1196`
  - final: `diag/recon_q_err=0.2600`, `diag/recon_a_err=0.1200`
  - final: `diag/copy_q_err=0.5226`, `diag/copy_a_err=0.2396`
  - window trend (first100 -> last100):
    - `lift_q: 0.1280 -> 0.2028`
    - `lift_a: 0.0261 -> 0.1180`
- `recon_chamfer`
  - final: `diag/recon_lift_q=0.2626`, `diag/recon_lift_a=0.1196`
  - final: `diag/recon_q_err=0.2599`, `diag/recon_a_err=0.1200`
  - final: `diag/copy_q_err=0.5226`, `diag/copy_a_err=0.2396`
  - window trend (first100 -> last100):
    - `lift_q: 0.1273 -> 0.2028`
    - `lift_a: 0.0260 -> 0.1180`

Interpretation:

- both objectives show the same qualitative behavior:
  - `recon_lift_q/a` move from near-zero (or negative early) to stable positive.
  - prediction is consistently better than copy-baseline at the end.
- in this setup, `recon_mse` and `recon_chamfer` are almost numerically equivalent
  on the tracked diagnostics.

## 33. Full FT launched from short recon checkpoints (2026-03-04)

Rationale:

- tail behavior at 2000 steps is plateau-like; proceed to downstream FT for decision.

Submitted jobs:

- `105297.qjcm` (`reconmse`, `obj_bg`)
- `105298.qjcm` (`reconmse`, `obj_only`)
- `105299.qjcm` (`reconmse`, `pb_t50_rs`)
- `105300.qjcm` (`reconchamfer`, `obj_bg`)
- `105301.qjcm` (`reconchamfer`, `obj_only`)
- `105302.qjcm` (`reconchamfer`, `pb_t50_rs`)

Common FT setup:

- `EPOCHS=300`, `N_POINT=2048`, `BATCH=64(global)`
- `patchnepa_ft_mode=qa_zeroa`
- `pooling=cls_max`, `head_mode=pointmae_mlp`
- `AUG_PRESET=pointmae`, `AUG_EVAL=1`, `MC_EVAL_K_TEST=10`
- W&B enabled (`patchnepa-finetune`)

Submit log:

- `logs/sanity/patchnepa_ft/ft_recon_full_20260304_190412/job_ids.txt`

## 34. Centered-cosine branch tracking added (2026-03-04)

Goal:

- keep `nepa_cosine` path but test residualized cosine via centering before cosine objective.

Added control knobs:

- `NEPA_CENTER_MODE`: `none | shape | segment`
- `NEPA_CENTER_WARMUP_FRAC`: ramp fraction for center contribution

Propagation confirmed in launchers:

- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
- `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_multinode_pbsdsh.sh`

Smoke jobs launched (same config except centering mode):

- `105310` (`none`, baseline) -> done (`Exit_status=0`)
- `105319` (`segment`) -> done (`Exit_status=0`)
- `105320` (`shape`) -> done (`Exit_status=0`)

Log roots:

- baseline: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_192014/`
- segment: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_193732/`
- shape: `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_smoke_20260304_193922/`

Current readout policy:

- keep `runlog_patch_nepa_202602.md` as canonical detailed record (final values + diagnostics).
- keep this restart plan focused on branch structure and job topology.

Observed branch shape:

- `segment` with warmup ends in `cos_prev > cos_tgt` (negative gap regime).
- `shape` converges to near-trivial high cosine (`cos_tgt≈cos_prev≈0.975`) and does not improve copy margin.

## 35. `skip_k` short sweep submitted (2026-03-04)

Purpose:

- minimal horizon test for `skip_k` effectiveness under the current token NEPA setup.

Configuration lock:

- objective/layout/order/mask fixed:
  - `nepa_cosine`, `split_sep`, `morton`, `answer_and_point_context`, `nepa_center_mode=none`
- data fixed:
  - `drop1_mesh50_udf50`
- budget fixed:
  - `MAX_STEPS=2000`

Sweep axis:

- `SKIP_K ∈ {1,2,4}`

Submitted jobs:

- `105324` (`skipk1_m50u50_smoke`)
- `105325` (`skipk2_m50u50_smoke`)
- `105326` (`skipk4_m50u50_smoke`)

Common logs:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_skipk124_smoke_20260304_200306/`

Completion:

- all three (`105324/105325/105326`) finished with `Exit_status=0`.
- end metrics are almost identical across `k=1,2,4` (`gap≈+0.0005`).
- decision: keep `skip_k=1` as default; no evidence that longer horizon fixes the core diag collapse.

## 36. Center warmup isolate (`segment`, 200-step no-warmup) (2026-03-04)

Purpose:

- isolate whether the step~100 spike in centerseg is caused by warmup schedules.

Job:

- `105328` (`ptokcs0`) with
  - `NEPA_CENTER_MODE=segment`
  - `NEPA_CENTER_WARMUP_FRAC=0.0`
  - `MAX_STEPS=200`, `EPOCHS=0`
  - other settings matched to centerseg smoke branch.

Outcome:

- job finished (`Exit_status=0`).
- removing center warmup removes the abrupt mid-early transient seen in warmup=0.05 run.
- but convergence quality is worse (`cos_prev` stays above `cos_tgt`, more negative gap).

Operational takeaway:

- the early spike was warmup-coupled.
- warmup is not the root cause of `cos_tgt≈cos_prev`; that issue persists with and without center warmup.

## 37. Backfill for previously unrecorded run roots (2026-03-04)

Objective:

- close documentation gaps so every generated token pretrain root is classified as:
  - valid evidence,
  - superseded failure,
  - or non-materialized artifact.

Backfilled classes:

- non-materialized empty roots:
  - examples: `patchnepa_tokens_20260304_005229`, `patchnepa_tokens_drop1_20260304_001817`, `patchnepa_tokens_dualcol_20260304_032435`
- pbs-only early failures (no `mr0`):
  - examples: `patchnepa_tokens_varcov_20260304_025805`, `patchnepa_tokens_hreg_20260304_033023`, `patchnepa_tokens_reconchamfer_smoke_20260304_041358`
- explicit failed chains with logs:
  - `patchnepa_tokens_full300_morton_20260304_044214` (`102560/102561/102562`, `Exit_status=97`)
  - `patchnepa_tokens_full300_morton_fixddp_20260304_044633` (`102572/102573/102574`, `Exit_status=97`)
  - `patchnepa_tokens_full300_morton_cm15655_20260304_102805` (`102828`, partial then stop, `Exit_status=271`)
  - `patchnepa_tokens_reconmse_diag_20260304_181957` (`105285`, fail) and `patchnepa_tokens_reconchamfer_diag_20260304_181957` (`105286`, fail), both superseded by diagfix runs (`105287/105288`).

Operational rule:

- for analysis and decision making, only use validated branches already recorded in sections 32-36 and the canonical runlog sections 138+.

## 38. InfoNCE / Residual summary (as of 2026-03-04)

InfoNCE:

- objective codepath exists in `pretrain_patch_nepa_tokens.py`, but no completed run in this branch has been executed with `pretrain_objective=nepa_infonce`.
- observed recent runs show `loss_infonce=0.0` across all steps (inactive).

Residual (centered-cosine):

- `center_mode=none` (`105310`): low cosine, near-zero negative gap.
- `center_mode=segment` (`105319`): better cosine than baseline, but stable `cos_prev > cos_tgt`.
- `center_mode=shape` (`105320`): very high cosine (`~0.975`) but gap remains near zero.
- `segment` no-warmup isolate (`105328`, 200 steps): removes early warmup transient but worsens negative gap.

Decision at this checkpoint:

- residual branch alone is not sufficient to solve the `cos_tgt≈cos_prev` issue.
- InfoNCE remains an untested branch and should be treated as pending, not failed.

## 39. InfoNCE short smoke launched (2026-03-04)

Job:

- `105332.qjcm` (`pntok_nce`)
- run set: `patchnepa_tokens_infonce_smoke_20260304_205108`
- run tag: `infonce_m50u50_smoke`

Locked configuration:

- `pretrain_objective=infonce`, `infonce_tau=0.07`
- `mix_config=drop1_mesh50_udf50`
- `morton`, `split_sep`, `answer_and_point_context`, `skip_k=1`
- `nepa_center_mode=none`
- `max_steps=2000`

Runtime check:

- state entered `R` and DDP node logs show `objective: pretrain_objective=infonce` (no fallback).

Logs:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_infonce_smoke_20260304_205108/infonce_m50u50_smoke.pbs.log`
- `logs/ddp_patch_nepa_tokens_pretrain/ddp_patchnepa_tokens_105332.qjcm_infonce_m50u50_smoke/`

## 40. Cross-branch insights (interim, 2026-03-04)

1. `skip_k` insight:
- `k=1/2/4` produced near-identical end state (`gap≈+0.0005`, similar loss/copy_win).
- implication: current collapse/copy issue is not horizon-length limited; increasing skip distance alone is insufficient.

2. residual-centering insight:
- `segment` improves absolute cosine but trends to negative gap (`cos_prev > cos_tgt`).
- `shape` reaches very high cosine (`~0.975`) with near-zero gap, indicating trivialized similarity scaling rather than improved predictiveness.
- implication: centering changes optimization geometry but does not reliably improve copy-risk margin.

3. warmup isolate insight:
- removing center warmup removes the early transient spike.
- but stable-region quality is not improved (gap worsens in segment no-warmup short run).
- implication: warmup explains transient behavior, not the core `cos_tgt≈cos_prev` pathology.

4. InfoNCE interim insight (`105332`, in progress):
- positives/negatives are separated (`pos_cos ~0.306`, `neg_cos ~-0.033`, `margin ~0.339` at tail100).
- yet `cos_tgt≈cos_prev` remains near-zero-gap at current progress.
- implication: discrimination in objective space is happening, but it has not yet translated into better copy-risk diagnostics.

## 41. InfoNCE smoke completed (`105332`) and final readout (2026-03-05)

Updated status:

- `105332` (`infonce_m50u50_smoke`) finished to `step=2000` with `[done]`.
- this supersedes section 38/40's "in progress" interpretation.

Paths:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_infonce_smoke_20260304_205108/infonce_m50u50_smoke.mr0.log`
- `logs/ddp_patch_nepa_tokens_pretrain/ddp_patchnepa_tokens_105332.qjcm_infonce_m50u50_smoke/pretrain_done.marker`

Final metrics (`step=2000`):

- `loss_total=6.825081`
- `loss_infonce=6.825081`
- `loss_nepa=0.694207` (diagnostic reference)
- `pos_cos=0.3058`
- `neg_cos=-0.0345`
- `margin=0.3403`
- `r1=0.0041`
- `cos_tgt=0.3058`
- `cos_prev=0.3051`
- `gap=+0.0007`
- `copy_win=0.7301`

Interpretation update:

- InfoNCE objective is active and stable (`margin > 0`, `neg_cos < 0`).
- but copy-risk proxy remains unresolved (`cos_tgt≈cos_prev`, high `copy_win`).
- therefore InfoNCE alone did not break the current near-copy regime in this short setup.

## 42. Recon full FT (`105297`-`105302`) completed and test summary (2026-03-05)

Updated status:

- all six FT jobs launched in section 33 completed with `TEST acc`.

Path:

- `logs/sanity/patchnepa_ft/ft_recon_full_20260304_190412/`

Final TEST:

- `recon_mse`
  - `obj_bg=0.7900` (`reconmse_obj_bg_fullft_20260304_190412.out`)
  - `obj_only=0.7814` (`reconmse_obj_only_fullft_20260304_190412.out`)
  - `pb_t50_rs=0.7620` (`reconmse_pb_t50_rs_fullft_20260304_190412.out`)
- `recon_chamfer`
  - `obj_bg=0.8124` (`reconchamfer_obj_bg_fullft_20260304_190412.out`)
  - `obj_only=0.8313` (`reconchamfer_obj_only_fullft_20260304_190412.out`)
  - `pb_t50_rs=0.7765` (`reconchamfer_pb_t50_rs_fullft_20260304_190412.out`)

Comparison against prior v1-family baseline (`obj_bg=0.8262`, `obj_only=0.8417`, `pb_t50_rs=0.7845`):

- `recon_mse`: `-0.0362 / -0.0603 / -0.0225`
- `recon_chamfer`: `-0.0138 / -0.0104 / -0.0080`

Interpretation update:

- `recon_chamfer` is consistently better than `recon_mse` on all three FT variants.
- however, both remain below the prior v1-family baseline.
- reconstruction-space lift is valid, but current downstream transfer is still insufficient.

## 43. PointMAE-style path vs PointGPT-style path: practical delta (2026-03-05)

Conclusion:

- high-level frame is similar (grouped point patches, causal transformer, geometric reconstruction),
  but implementation objective is not equivalent yet.

Main differences (current branch vs `PointGPT`):

1. Prediction target:
- current branch:
  - latent path: cosine/InfoNCE over token embeddings
  - recon path: `ctx` uses Chamfer (patch xyz), `q/a` use MSE
- PointGPT:
  - point-patch reconstruction in xyz space as primary objective (Chamfer family)

2. Sequence/task structure:
- current branch:
  - Q/A split and primitive-specific answer stream (`pc/mesh/udf`)
- PointGPT:
  - single point stream (no primitive-conditioned answer channel)

3. Dual-mask semantics:
- current branch:
  - dual mask is configurable (`dual_mask_mode=column`, `keep_prefix`, warmup),
  - but many recent runs used `dual_mask_near/far=0` (effectively no additional mask pressure)
- PointGPT:
  - causal mask + additional column-wise mask is part of the core path
  - first `keep_attend=10` tokens are protected, remainder masked by ratio (`mask_ratio` in config)

4. Ordering/grouping prior:
- current branch:
  - order controlled by `patch_order_mode` (morton/fps/random/sample)
- PointGPT:
  - grouping + order heuristics are tightly integrated in its `Group`/transformer path

Actionable implication:

- "dual mask + chamfer exists" is true, but not yet "same training regime as PointGPT".
- if moving to PointGPT-style reference, prioritize:
  - stronger effective column masking (non-zero mask pressure),
  - xyz reconstruction as the dominant objective on the main path,
  - then re-measure downstream FT under the same full-data protocol.

## 44. PointGPT pretrain diag alignment for PatchNEPA comparison (2026-03-05)

Purpose:

- align PointGPT logs with PatchNEPA recon diagnostics (`recon/copy/lift`) for fair trend comparison.

Code updates:

- `PointGPT/models/PointGPT.py`
  - `forward(..., return_metrics=True)` path added.
  - logs objective-aligned chamfer diagnostics (valid positions = tokens `1..G-1`):
    - `recon_cd_l1`, `recon_cd_l2`
    - `copy_cd_l1`, `copy_cd_l2`  (copy baseline: previous patch -> current patch)
    - `recon_lift_cd_l1`, `recon_lift_cd_l2` (`copy - recon`)
- `PointGPT/tools/runner_pretrain.py`
  - calls model with `return_metrics=True`.
  - sends `diag/*` metrics to W&B every train step.
- `PointGPT/main.py`
  - keep W&B active with explicit `wandb.log` path (`sync_tensorboard=False`) to avoid step conflicts.

Relaunched run:

- `105848.qjcm`
- run tag: `pointgpt_pretrain_scan_20260305_202253_diagcmp`
- W&B run: `https://wandb.ai/ide_koh/pointgpt-pretrain/runs/plstb5ht`
- log: `logs/sanity/pointgpt_pretrain_scan/pointgpt_pretrain_scan_20260305_202253_diagcmp.out`

Verification:

- local W&B run file includes:
  - `train/loss`
  - `diag/recon_cd_l1`, `diag/recon_cd_l2`
  - `diag/copy_cd_l1`, `diag/copy_cd_l2`
  - `diag/recon_lift_cd_l1`, `diag/recon_lift_cd_l2`

Note:

- `cos_tgt/cos_prev` are not native PointGPT objective metrics; primary comparison should be done in reconstruction-aligned space (`recon/copy/lift`).

## 45. Latent-space comparison axis added for PointGPT vs PatchNEPA (2026-03-05)

Motivation:

- user-requested latent comparison in addition to objective-space reconstruction diagnostics.

Added PointGPT latent diagnostics (`diag/*`):

- `diag/latent_z_cos_prev` : adjacent-token cosine on encoder tokens (`z_in`)
- `diag/latent_h_cos_prev` : adjacent-token cosine on transformer context tokens (`h_ctx`)
- `diag/latent_z_cos_far`  : far-offset token cosine on `z_in`
- `diag/latent_h_cos_far`  : far-offset token cosine on `h_ctx`
- `diag/latent_z_std_mean`, `diag/latent_z_std_min`
- `diag/latent_h_std_mean`, `diag/latent_h_std_min`

Interpretation mapping to PatchNEPA:

- PointGPT `latent_*_cos_prev` ↔ PatchNEPA `diag/cos_prev` (local copy risk / token similarity pressure)
- PointGPT `latent_*_std_*` ↔ PatchNEPA `diag/target_std_*` family (token spread / collapse risk)
- Primary objective comparison remains:
  - PointGPT: `diag/recon_cd_*`, `diag/copy_cd_*`, `diag/recon_lift_cd_*`
  - PatchNEPA: `diag/recon_*_err`, `diag/copy_*_err`, `diag/recon_lift_*`

Run with latent diagnostics enabled:

- `105849.qjcm`
- W&B: `https://wandb.ai/ide_koh/pointgpt-pretrain/runs/mayuu2rc`
- log: `logs/sanity/pointgpt_pretrain_scan/pointgpt_pretrain_scan_20260305_202739_diagcmp_latent.out`

Verification:

- local W&B run file contains `diag/latent_*` keys and `diag/recon_lift_cd_*` keys.

## 46. PatchNEPA dual-mask parity update vs PointGPT (2026-03-05)

Goal:

- make `dual_mask_mode=column` behavior match PointGPT core semantics:
  - keep first `keep_prefix` tokens always visible
  - apply column-wise drop by fixed `mask_ratio` even when `near/far=0`

Code changes:

- `nepa3d/models/causal_transformer.py`
  - added `dual_mask_column_ratio`.
  - `column` mode now supports fixed-count column drop (`round(num_eligible * ratio)`), matching PointGPT-style column masking.
  - backward-compatible path kept:
    - when `dual_mask_column_ratio=0`, column drop falls back to near/far Bernoulli logic.
- `nepa3d/models/patch_nepa.py`
  - propagated `dual_mask_column_ratio` through `forward` / `forward_tokens`.
- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - added CLI `--dual_mask_column_ratio` (default `0.7`).
  - warmup now ramps `dual_mask_near/far/column_ratio`.
  - launch log line now prints `column_ratio=...`.
- launchers:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_multinode_pbsdsh.sh`
  - all propagate/pass `DUAL_MASK_COLUMN_RATIO`.

Operational note:

- with `dual_mask_mode=column`, `keep_prefix=10`, `dual_mask_column_ratio=0.7`,
  additional column masking is active even under `dual_mask_near=0`, `dual_mask_far=0`.

## 47. PointGPT ShapeNet re-run (apples-to-apples pretrain domain) (2026-03-05)

Request addressed:

- rerun PointGPT pretrain on ShapeNet (not ScanObjectNN) with W&B tracking.

Prep updates:

- `PointGPT/datasets/io.py`
  - added `.npz` support (`pc_xyz` preferred key).
- `PointGPT/datasets/ShapeNet55Dataset.py`
  - robust path parsing for non-legacy list formats (`train/<tax>/<id>.npz`).
  - random sampler made point-count robust (works with 2048-point caches while sampling 1024).
- new submit path:
  - `scripts/sanity/pointgpt_pretrain_shapenet_qf.sh`
  - `scripts/sanity/submit_pointgpt_pretrain_shapenet_qf.sh`

Data wiring used:

- source cache: `data/shapenet_cache_v2_20260303`
- generated list files:
  - `PointGPT/data/ShapeNet55-34/ShapeNet-55/train.txt` (`47445`)
  - `PointGPT/data/ShapeNet55-34/ShapeNet-55/test.txt` (`5357`)
- point path symlink:
  - `PointGPT/data/ShapeNet55-34/shapenet_pc -> data/shapenet_cache_v2_20260303`

Submitted runs:

- `105851.qjcm`: PointGPT ShapeNet pretrain (300-epoch config)
  - run tag: `pointgpt_shapenet_300_20260305_213137`
  - W&B run started and syncing (`project=pointgpt-pretrain`, group `pointgpt_shapenet_pretrain`).
- `105852.qjcm`: PatchNEPA parity smoke (`column+keep_prefix=10+column_ratio=0.7`, `near/far=0`)
  - run tag: `ptok_column_pgpt_equiv_smoke_20260305_213144`
  - confirms new dual-mask settings are active in startup log.

## 48. Immediate outcome after parity patch (2026-03-05)

### 48.1 PatchNEPA parity smoke (`105852`) — completed

Log:

- `logs/patch_nepa_pretrain_tokens/patchnepa_tokens_20260305_213144/ptok_column_pgpt_equiv_smoke_20260305_213144.mr0.log`

Final step (`2000/2000`):

- `loss_total=0.499029`
- `loss_nepa=0.499029`
- `loss2d=-0.500971`
- `cos_tgt=0.5010`
- `cos_prev=0.5005`
- `gap=+0.0005`
- `copy_win=0.7495`

W&B:

- `https://wandb.ai/ide_koh/patchnepa-pretrain/runs/7c358uzw`

Interpretation:

- dual-mask parity patch works technically (column mask active with `near/far=0`).
- however, behavior remains in the same regime (`cos_tgt≈cos_prev`, high `copy_win`).
- therefore, dual-mask implementation mismatch was not the primary bottleneck.

### 48.2 PointGPT ShapeNet run (`105851`) — in progress

Log:

- `logs/sanity/pointgpt_pretrain_shapenet/pointgpt_shapenet_300_20260305_213137.out`

Current snapshot:

- ShapeNet train/test loaded as expected (`47445` / `5357`)
- run is progressing normally through epochs (no startup/runtime error).
- recent train summary around epoch 6:
  - `[Training] EPOCH: 6 ... Losses=['43.6205']`

W&B:

- `https://wandb.ai/ide_koh/pointgpt-pretrain/runs/3i0xp672`

Operational conclusion at this point:

- apples-to-apples pretrain domain is now aligned (both on ShapeNet family data).
- with mask parity applied, remaining major differences are objective/task design
  (`PointGPT: reconstruction`, `PatchNEPA smoke here: latent cosine with QA stream).

## 49. Minimal causal test: reconstruction + generator depth sweep (2026-03-05)

Goal:

- test the next causal hypothesis directly:
  - `fixed-target reconstruction` already removed cone/mean-solution bottleneck.
  - remaining gap should be tested by adding PointGPT-like generator depth.

Code changes:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - added `--recon_generator_depth` (default `0`).
  - for `pretrain_objective in {recon_mse, recon_chamfer}`:
    - `depth=0`: legacy path (no generator).
    - `depth>0`: inserts extra `CausalTransformer` before recon heads.
  - generator parameters are added to optimizer and clipping set.
  - checkpoint now saves optional `recon_generator` state dict.
- launchers updated to propagate new flag:
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - `scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh`
  - `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_multinode_pbsdsh.sh`

Submitted short A/B/C (same config except depth):

- run set: `patchnepa_tokens_reconch_genab_20260305_221814`
- objective: `recon_chamfer` (`l2`)
- data: `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50`
- budget: `MAX_STEPS=2000`, `EPOCHS=0`
- DDP: `rt_QF=4`, `NPROC_PER_NODE=4` (16 GPU)
- W&B: enabled (`project=patchnepa-pretrain`, group=`patchnepa_tokens_reconch_genab_20260305_221814`)

Jobs:

- `105857.qjcm` (`reconch_m50u50_gend0`)
- `105858.qjcm` (`reconch_m50u50_gend2`)
- `105859.qjcm` (`reconch_m50u50_gend4`)

Startup verification (mr0 logs):

- all three runs print expected `recon_generator_depth={0,2,4}`.
- all three runs are actively training; no startup/runtime error observed in current snapshot.

Primary readout for this sweep:

- objective-aligned diagnostics:
  - `diag/recon_lift_q`, `diag/recon_lift_a`
  - `diag/recon_q_err`, `diag/recon_a_err`
- downstream decision metric (next stage):
  - FT accuracy vs current `recon + no-generator` baseline.

## 50. Add-on compare run: `gend4` with PointGPT-style column dual-mask (2026-03-05)

Rationale:

- in the first `gend{0,2,4}` sweep, dual-mask was effectively off
  (`mode=element`, `near/far=0`).
- for direct compare, add one run with PointGPT-equivalent column mask.

Submitted:

- `105865.qjcm`
  - run set: `patchnepa_tokens_reconch_gend4_colmask_20260305_223447`
  - run tag: `reconch_m50u50_gend4_colmask`
  - objective: `recon_chamfer`
  - data: `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50`
  - generator: `recon_generator_depth=4`
  - dual-mask:
    - `DUAL_MASK_MODE=column`
    - `DUAL_MASK_KEEP_PREFIX=10`
    - `DUAL_MASK_COLUMN_RATIO=0.7`
    - `DUAL_MASK_NEAR=0.0`, `DUAL_MASK_FAR=0.0`, `DUAL_MASK_WINDOW=0`
  - budget: `MAX_STEPS=2000`, `EPOCHS=0`

Startup verification:

- run log confirms:
  - `objective: ... recon_generator_depth=4`
  - `dual_mask: ... mode=column keep_prefix=10 column_ratio=0.7 ...`

## 51. Reconstruction log cleanup (`cos_*` hidden) (2026-03-05)

Request addressed:

- hide `cos_tgt/cos_prev/gap/copy_win` from reconstruction runs to avoid metric-space mismatch confusion.

Applied in:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`

Behavior after patch:

- console print:
  - `objective in {recon_mse, recon_chamfer}` prints only objective-aligned recon diagnostics
    (`copy_*_err`, `recon_lift_*`) and losses.
  - `cos_*` print remains only for `nepa_cosine` / `infonce`.
- W&B logging:
  - `diag/cos_*` and `diag/*_raw` are logged only for `nepa_cosine` / `infonce`.
  - recon runs keep objective-aligned keys:
    - `diag/recon_ctx_err`, `diag/recon_q_err`, `diag/recon_a_err`
    - `diag/copy_ctx_err`, `diag/copy_q_err`, `diag/copy_a_err`
    - `diag/recon_lift_ctx`, `diag/recon_lift_q`, `diag/recon_lift_a`

Sanity confirmation on new running recon job (`105868`, ShapeNet full):

- log example (`.../qh003.patchnepa_tokens.log`) shows recon print without `cos_*` fields.

## 52. Best recon selection and full300+FT launch (2026-03-05)

Selection basis (completed short sweep, same setup except generator depth):

- run set: `patchnepa_tokens_reconch_genab_20260305_221814`
- final (`step=2000`) key metrics:
  - `gend0` (`105857`): `lift_q=0.2626`, `lift_a=0.1196`
  - `gend2` (`105858`): `lift_q=0.2593`, `lift_a=0.1208`
  - `gend4` (`105859`): `lift_q=0.2452`, `lift_a=0.1174`
  - add-on `gend4+column` (`105865`): `lift_q=0.2445`, `lift_a=0.1173`

Operational pick for full run:

- **`recon_chamfer + recon_generator_depth=0`** (best combined lift with simplest path).

Submitted full pretrains (`EPOCHS=300`, ShapeNet full):

- submit root:
  - `logs/sanity/patchnepa_submit/patchnepa_reconbest_full300_20260305_224714/`
- jobs:
  - `105868` : `pc100`
  - `105869` : `mesh50udf50`
  - `105870` : `pc33mesh33udf33`
- record:
  - `logs/sanity/patchnepa_submit/patchnepa_reconbest_full300_20260305_224714/pretrain_jobs.tsv`

Submitted dependent FT (3 variants × 3 pretrains):

- record:
  - `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714/job_ids.tsv`
- FT job IDs:
  - from `105868`: `105872`, `105873`, `105874`
  - from `105869`: `105875`, `105876`, `105877`
  - from `105870`: `105878`, `105879`, `105880`

Current scheduler state snapshot:

- pretrains `105868/105869/105870`: `R` (running)
- dependent FT `105872`–`105880`: `H` (dependency hold until pretrain success)

## 53. PointGPT-loss parity controls added for reconstruction branch (2026-03-05)

Issue raised:

- previous recon branch optimized `ctx(chamfer)+q(mse)+a(mse)` composite loss.
- this is not the same axis as PointGPT pretrain objective (`CD`-only patch reconstruction).

Implemented controls:

- `--recon_loss_mode`
  - `composite` (legacy): `ctx + q + a` weighted sum.
  - `pointgpt_ctx_only`: optimize only context reconstruction term (`loss_recon_ctx`).
- `--recon_chamfer_metric` now supports `l12` in addition to `l1/l2`.
  - `l12` = `ChamferL1 + ChamferL2` (PointGPT-style cdl12 axis).

Logging updates for analysis:

- recon runs now expose:
  - `train/loss_pointgpt_equiv` (= `loss_recon_ctx`)
  - `train/loss_pointgpt_equiv_x1k`
  - `diag/recon_loss_mode_id` (`0=composite`, `1=pointgpt_ctx_only`)

Q/A split confirmation:

- Q/A are computed on separate token positions and separate heads/losses:
  - token position split: `_sequence_absolute_positions(...)`
  - separate indices/losses: `q_pred_idx/q_tgt_idx`, `a_pred_idx/a_tgt_idx`,
    `loss_recon_q`, `loss_recon_a`.

Operational note:

- already-running jobs keep old launch-time settings.
- use the new knobs on the next submission for strict PointGPT-axis comparison.

## 54. Strict PointGPT-loss axis run re-submitted (`pc100` only) (2026-03-05)

Intent:

- keep the current 3 full runs as main track.
- add one strict comparison pretrain run on `pc100` with PointGPT-equivalent loss axis.

Submitted:

- `105884.qjcm`
- run set:
  - `patchnepa_recon_pgptctxonly_pc100_full300_20260305_231258`
- run tag:
  - `pt_pc100_reconch_ctxonly_l12_e300`

Key config:

- `MIX_CONFIG=nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml`
- `PRETRAIN_OBJECTIVE=recon_chamfer`
- `RECON_LOSS_MODE=pointgpt_ctx_only`
- `RECON_CHAMFER_METRIC=l12`  (PointGPT cdl12 axis)
- `RECON_GENERATOR_DEPTH=0`
- `EPOCHS=300`, `MAX_STEPS=0` (epoch-driven full run)

Queue status at submission check:

- `job_state=R`

## 55. Translation-centric minimal compare enabled + completed (2026-03-06)

Intent:

- add only comparison-enabling changes for the translation/CPAC direction.
- do **not** hard-code stronger research conclusions (`generator required`, staged training,
  `PC` removal, discrete answer language) into the mainline.

Code changes applied:

- `nepa3d/train/pretrain_patch_nepa_tokens.py`
  - added `recon_loss_mode={answer_only, context_plus_answer, query_plus_answer}`
  - added loss-mode-aware DDP handling:
    - `find_unused_parameters=True` is now enabled automatically for reconstruction modes
      that optimize only a strict subset of `{ctx, q, a}` heads
    - this is the required fix to avoid the same unused-parameter failure seen in
      `pointgpt_ctx_only` run `105884`
  - W&B `diag/recon_loss_mode_id` now distinguishes all active reconstruction modes
- `nepa3d/analysis/completion_cpac_udf_patchnepa.py`
  - added `--surf_xyz_key`, `--qry_xyz_key`, `--qry_dist_key`
  - added `--context_primitive`, `--query_primitive`
  - context/query token `type_id` now follows primitive-specific PatchNEPA point types
- `scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh`
  - forwards the new CPAC primitive/key arguments
- `scripts/pretrain/nepa3d_pretrain_patch_nepa_tokens_qf.sh`
  - updated `RECON_LOSS_MODE` comment to include the new translation-centric modes

Minimal experiment matrix (screening only):

1. short pretrain sweep (`recon_chamfer`, `g0`, `pc33_mesh33_udf33`, `MAX_STEPS=2000`)
2. dependent mini-CPAC on each resulting checkpoint
   - context: `pc`
   - query: `udf`
   - keys: `pc_xyz -> udf_qry_xyz / udf_qry_dist`
3. defer until after this screen:
   - `query_plus_answer` launch
   - `g0` vs `g2` comparison on the winning loss mode
   - ScanObjectNN FT on the winning loss mode only

Submitted jobs:

| arm | pretrain mode | pretrain job | cpac job | status at submit check | pretrain save root |
|---|---|---|---|---|---|
| `cmp` | `composite` | `106029.qjcm` | `106030.qjcm` | pretrain `R`, CPAC `H(afterok)` | `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_cmp/pt_pc33_reconch_g0_cmp_s2000` |
| `ans` | `answer_only` | `106031.qjcm` | `106032.qjcm` | pretrain `R`, CPAC `H(afterok)` | `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_ans/pt_pc33_reconch_g0_ans_s2000` |
| `cpa` | `context_plus_answer` | `106033.qjcm` | `106034.qjcm` | pretrain `R`, CPAC `H(afterok)` | `runs/patchnepa_tokens/patchnepa_tokens_translationloss_pc33_g0_20260306_114629_cpa/pt_pc33_reconch_g0_cpa_s2000` |

Operational settings shared by the three pretrains:

- `RT_QF=2`, `NPROC_PER_NODE=4`, `GRAD_ACCUM=2`, `BATCH=8`
- effective global batch remains `128`
- `PRETRAIN_OBJECTIVE=recon_chamfer`
- `RECON_GENERATOR_DEPTH=0`
- `SAVE_EVERY=1000`

Primary readout for this screen:

- pretrain:
  - `diag/recon_lift_q`
  - `diag/recon_lift_a`
  - `diag/target_std_mean`
  - `diag/target_std_min`
- dependent mini-CPAC:
  - `mae`
  - `rmse`
  - `iou@0.01`

Decision rule after completion:

- if `answer_only` or `context_plus_answer` beats `composite` on both
  `recon_lift_a` and mini-CPAC, promote that mode to the next `g0 vs g2` compare
- otherwise keep `composite` as the reconstruction baseline and treat the
  translation-centric modes as negative controls for now

Completed outcome:

- pretrain `106029/106031/106033` all finished successfully.
- dependent CPAC jobs `106030/106032/106034` are **invalid** as canonical
  evidence because they pointed one directory too deep and failed with
  `[error] ckpt not found`.
- the decision rule resolved to:
  - keep `composite` as the reconstruction baseline,
  - keep translation-centric modes as screening controls for now,
  - use mini-CPAC only as a secondary signal until a loss mode wins on both
    pretrain and downstream criteria.

## 56. `reconbest` full300 finalized (`g0`, 2026-03-05/06)

Canonical sources:

- pretrain log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_reconbest_full300_20260305_224714`
- FT log root:
  - `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714`
- pretrain job manifest:
  - `logs/sanity/patchnepa_submit/patchnepa_reconbest_full300_20260305_224714/pretrain_jobs.tsv`
- FT job manifest:
  - `logs/sanity/patchnepa_ft/patchnepa_reconbest_full300_20260305_224714/job_ids.tsv`

Policy note (2026-03-14):

- these ScanObjectNN FT readouts were produced under the earlier
  `val_split_mode=file` policy
- keep them as historical/internal reference only
- do not reuse them as the current canonical ScanObjectNN benchmark headline

Per-source final readout:

| source | pretrain job | `recon_lift_q` | `recon_lift_a` | `obj_bg` | `obj_only` | `pb_t50_rs` |
|---|---:|---:|---:|---:|---:|---:|
| `pc100` | `105868` | `0.1349` | `0.1096` | `0.8348` | `0.8107` | `0.7998` |
| `mesh50udf50` | `105869` | `0.1896` | `0.1466` | `0.8399` | `0.8227` | `0.8001` |
| `pc33mesh33udf33` | `105870` | `0.1640` | `0.1357` | `0.8365` | `0.8348` | `0.8102` |

Historical headline read:

- best `obj_bg`: `0.8399` (`mesh50udf50`)
- best `obj_only`: `0.8348` (`pc33mesh33udf33`)
- best `pb_t50_rs`: `0.8102` (`pc33mesh33udf33`)
- vs historical v1 reference (`0.8262 / 0.8417 / 0.7845`), `g0` beats v1 on
  `obj_bg` and `pb_t50_rs`, but remains below on `obj_only`.

## 57. `recong2` full300 finalized (`g2`, 2026-03-06)

Canonical sources:

- pretrain log root:
  - `logs/patch_nepa_pretrain_tokens/patchnepa_recong2_full300_20260306_072643`
- FT log root:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643`
- pretrain job manifest:
  - `logs/sanity/patchnepa_submit/patchnepa_recong2_full300_20260306_072643/pretrain_jobs.tsv`
- FT job manifest:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_full300_20260306_072643/job_ids.tsv`

Policy note (2026-03-14):

- these ScanObjectNN FT readouts were produced under the earlier
  `val_split_mode=file` policy
- keep them as historical/internal reference only
- do not reuse them as the current canonical ScanObjectNN benchmark headline

Per-source final readout:

| source | pretrain job | `recon_lift_q` | `recon_lift_a` | `obj_bg` | `obj_only` | `pb_t50_rs` |
|---|---:|---:|---:|---:|---:|---:|
| `pc100` | `105911` | `0.1520` | `0.1031` | `0.8279` | `0.8399` | `0.7932` |
| `mesh50udf50` | `105912` | `0.1666` | `0.1507` | `0.8485` | `0.8434` | `0.8053` |
| `pc33mesh33udf33` | `105913` | `0.1821` | `0.1421` | `0.8417` | `0.8589` | `0.8140` |

Historical headline read:

- best `obj_bg`: `0.8485` (`mesh50udf50`)
- best `obj_only`: `0.8589` (`pc33mesh33udf33`)
- best `pb_t50_rs`: `0.8140` (`pc33mesh33udf33`)
- this historical file-split readout exceeds both the historical v1 reference
  (`0.8262 / 0.8417 / 0.7845`) and the `g0` historical file-split headline
  (`0.8399 / 0.8348 / 0.8102`) on all three ScanObjectNN variants.

## 58. Translation-loss minimal screen completed (2026-03-06)

Canonical W&B sources:

- `wandb/run-20260306_114732-75xy5ej0/files/wandb-summary.json`
- `wandb/run-20260306_114732-awx50i19/files/wandb-summary.json`
- `wandb/run-20260306_114732-wgtsmqey/files/wandb-summary.json`

Short pretrain readout (`pc33mesh33udf33`, `g0`, `recon_chamfer`, `MAX_STEPS=2000`):

| arm | mode | pretrain job | `recon_lift_q` | `recon_lift_a` | `target_std_mean` | `target_std_min` |
|---|---|---:|---:|---:|---:|---:|
| `cmp` | `composite` | `106029` | `+0.1371` | `0.1132` | `0.1423` | `0.0356` |
| `ans` | `answer_only` | `106031` | `-0.2415` | `0.1133` | `0.3654` | `0.0703` |
| `cpa` | `context_plus_answer` | `106033` | `-0.1777` | `0.1130` | `0.1443` | `0.0364` |

Decision outcome:

- `answer_only` and `context_plus_answer` did not beat `composite` on the joint
  criterion because both degraded `recon_lift_q` sharply.
- `composite` remains the reconstruction baseline for the next `g0` vs `g2`
  comparisons.
- translation-centric modes remain useful screening controls, not mainline.

## 59. Mini-CPAC reevaluation completed (2026-03-06)

Invalid first dependency chain:

| arm | invalid job | reason |
|---|---:|---|
| `cmp` | `106030` | checkpoint path pointed one directory too deep |
| `ans` | `106032` | checkpoint path pointed one directory too deep |
| `cpa` | `106034` | checkpoint path pointed one directory too deep |

Canonical rerun sources:

- log root:
  - `logs/patch_nepa_cpac/patchnepa_tokens_translationloss_pc33_g0_20260306_1900_rerun2`
- result JSON root:
  - `results/patch_nepa_cpac/patchnepa_tokens_translationloss_pc33_g0_20260306_1900_rerun2`

Canonical rerun metrics (`PC context -> UDF query`, `64/64` shapes, `rep_source=h`):

| arm | corrected job | result JSON | `iou@0.01` | `mae` | `rmse` |
|---|---:|---|---:|---:|---:|
| `cmp` | `106573` | `cpac_pc2udf_cmp_fix.json` | `0.0948` | `0.07585` | `0.09929` |
| `ans` | `106572` | `cpac_pc2udf_ans_fix.json` | `0.1033` | `0.07584` | `0.09991` |
| `cpa` | `106571` | `cpac_pc2udf_cpa_fix.json` | `0.0954` | `0.07653` | `0.10043` |

Conclusion:

- `answer_only` is best on `iou@0.01`,
- `composite` is best on `rmse`,
- `context_plus_answer` is weakest overall,
- CPAC therefore gives a real but still split signal: translation-style loss
  helps a thresholded completion metric, but not enough to replace the current
  reconstruction baseline.

## 60. Sampling-parity screen (`fps_then_sample`) completed (2026-03-06)

Canonical source:

- FT log:
  - `logs/sanity/patchnepa_ft/patchnepa_recong2_pc33_objonly_fpsparity_20260306_1810/obj_only.out`

Result:

| run | train sampler | best `val_acc` | `TEST acc` |
|---|---|---:|---:|
| baseline `g2` (`pc33mesh33udf33`, `obj_only`) | `random` | `0.8969` | `0.8589` |
| parity screen `106582` | `fps_then_sample` | `0.8924` | `0.8193` |

Interpretation boundary:

- this is **not** strong causal evidence about Point-MAE-style train sampling,
  because the current `scanobjectnn_*_v3_nonorm` cache exposes
  `pc_xyz.shape[0] = 2048`,
- therefore `fps_then_sample` degenerates to a full-set permutation rather than
  a true `point_all > npoints` crop,
- keep this row as an inconclusive ablation only; do not promote it to a
  benchmark headline comparison.

Retention rule (effective immediately):

- do not treat flat PBS stdout/stderr such as `pntok_cpac.o*` as the sole
  storage location for final results,
- store final evidence in docs plus structured artifacts:
  - FT/pretrain metrics in the active docs,
  - machine-readable completion metrics in `results/patch_nepa_cpac/.../*.json`,
  - machine-readable pretrain diags in the corresponding W&B
    `wandb-summary.json`.

## 61. Visibility-first geometry-answer branch staged (`L000A/L000B`, 2026-03-07)

Purpose:

- test a first high-frequency geometry answer family before returning to
  `L001-L004`
- rebuild ShapeNet v2 derived caches from source with mesh-query-aligned
  visibility / occlusion answers
- compare same-lineage `baseline` vs `visocc` under one short
  `recon_chamfer + g2 + composite` screen and one fixed mini-CPAC protocol

Canonical structured paths:

- branch log root:
  - `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260306`
- branch decision JSON:
  - `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260306/decision.json`
- local launch script:
  - `scripts/local/patchnepa_visocc_branch.sh`
- source smoke artifact:
  - `results/_tmp_visocc_smoke/sample_visocc.npz`

New lineage for this exploratory branch:

- source cache:
  - `data/shapenet_cache_v2_20260306_visocc`
- split JSON:
  - `data/shapenet_unpaired_splits_v2_20260306_visocc.json`
  - `data/shapenet_unpaired_splits_v2_pc33_mesh33_udf33_visocc.json`
  - `data/shapenet_unpaired_splits_v2_mesh50_udf50_visocc.json`
- unpaired cache:
  - `data/shapenet_unpaired_cache_v2_20260306_visocc`
  - `data/shapenet_unpaired_cache_v2_20260306_visocc_drop1`
  - `data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33_visocc`
  - `data/shapenet_unpaired_cache_v2_mesh50_udf50_visocc`

Added answer family (first pass only):

- NPZ keys:
  - `mesh_qry_vis_hit`
  - `mesh_qry_vis_t`
- semantics:
  - local hemisphere visibility from each `mesh_qry_xyz`, aligned to
    `mesh_qry_n`
  - `16` deterministic hemisphere directions
  - `mesh_qry_vis_hit[d] = 1` if a ray hits within `max_t=0.25`
  - `mesh_qry_vis_t[d] = t_hit` if hit, else `0.25`

Public API changes staged for this branch:

- `nepa3d/data/preprocess_shapenet_v2.py`
  - new CLI:
    - `--mesh_vis_enable`
    - `--mesh_vis_dirs`
    - `--mesh_vis_max_t`
    - `--mesh_vis_eps`
  - new outputs:
    - `mesh_qry_vis_hit`
    - `mesh_qry_vis_t`
- `nepa3d/data/answer_feature_pack.py`
  - new schema entries:
    - `vis_hit`
    - `vis_t`
- `nepa3d/data/dataset_v2.py`
  - `vis_t` now follows the same geometric scaling rules as other distance-like
    answers
- `scripts/preprocess/preprocess_shapenet_v2.sh`
  - forwards the visibility knobs

Config lineage staged for the screen:

- baseline:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc_base.yaml`
- visibility:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc.yaml`
- side lineage prepared but not screened first:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50_visocc_base.yaml`
  - `nepa3d/configs/shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50_visocc.yaml`

Validation completed before launch:

- syntax / formatting:
  - `git diff --check`
  - `.venv/bin/python -m py_compile nepa3d/data/preprocess_shapenet_v2.py nepa3d/data/answer_feature_pack.py nepa3d/data/dataset_v2.py`
  - `bash -n scripts/preprocess/preprocess_shapenet_v2.sh scripts/local/patchnepa_visocc_branch.sh`
- data smoke:
  - `results/_tmp_visocc_smoke/sample_visocc.npz`
  - observed nontrivial keys:
    - `mesh_qry_vis_hit.shape == (64, 16)`
    - `mesh_qry_vis_t.shape == (64, 16)`
    - `vis_hit.mean = 0.1523`
    - `vis_t.mean = 0.2255`

Operational note:

- the first launch attempt exposed a source-root mismatch
  (`data/ShapeNetCore.v2` vs `data/ShapeNetCore.v2/synsets`)
- default local/preprocess roots were corrected before the current branch was
  relaunched

Current status at this memo update:

- `L000A` is running from
  `logs/local_patchnepa_visocc/patchnepa_visocc_l000ab_20260306`
- `L000B` is queued inside the same branch script and will only execute after
  the source-cache rebuild and the two short pretrain arms succeed
- no scientific conclusion changes are promoted yet

## 62. Worldvis source cache finalized at `52311` shapes; proceed with shape-dropout protocol (2026-03-12)

Canonical source-cache status:

- source cache root:
  - `data/shapenet_cache_v2_20260311_worldvis`
- finalized source count:
  - `52311` NPZ files
- unresolved source misses:
  - `161 / 52472` shapes (`0.3068%`)
- missing-shape artifacts:
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_by_synset.tsv`
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_synset_summary.tsv`
  - `logs/preprocess/shapenet_v2/shapenet_worldvis_20260311_abci_r3_missing_qc/missing_shapes_by_synset.json`

Decision:

- proceed with the existing `52311`-shape source cache,
- treat the unresolved `161` as **shape dropout / excluded shapes**,
- do **not** reinterpret them as modality-dropout or ad hoc `pc-only` samples.

Bias check summary:

- missing synsets with the largest counts:
  - `02691156`: `42 / 4045` (`1.04%`)
  - `03790512`: `22 / 337` (`6.53%`)
  - `04530566`: `22 / 1939` (`1.13%`)
  - `02958343`: `18 / 3514` (`0.51%`)
  - `04468005`: `9 / 389` (`2.31%`)
- interpretation:
  - the missing set is not fully uniform across synsets,
  - but the global missing rate remains low enough to proceed without blocking
    the pipeline.

Canonical post-materialization rerun from the existing source cache:

- base:
  - split log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_base`
  - split counts:
    - `train_mesh=16004`, `train_pc=15533`, `train_udf=15533`, `eval=5241`
  - materialize result:
    - `created=52311`, `missing=0`
- drop1:
  - split log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_drop1`
  - split counts:
    - `train_mesh=16004`, `train_pc=15533`, `train_udf=15533`, `eval=5241`
  - materialize result:
    - `created=52311`, `missing=0`
- pc33mesh33udf33:
  - split log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_pc33`
  - split counts:
    - `train_mesh=15533`, `train_pc=15533`, `train_udf=16004`, `eval=5241`
  - materialize result:
    - `created=52311`, `missing=0`
- mesh50udf50:
  - split log root:
    - `logs/preprocess/shapenet_unpaired/shapenet_worldvis_20260312_existing52311_qc_mesh50udf50`
  - split counts:
    - `train_mesh=23534`, `train_udf=23536`, `eval=5241`
  - materialize result:
    - `created=52311`, `missing=0`

Operational note:

- the directory-level file count may appear as `52312` because each
  materialized cache also contains `_meta/split_source.json`,
- the canonical dataset count is the `created=52311` value emitted by
  `preprocess_shapenet_unpaired.py`.

## 63. CQA vis/thickness smoke completed; branch wiring is valid but the current tasks are majority-code dominated (2026-03-15)

Canonical sources:

- smoke log:
  - `logs/cqa_pretrain/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_smoke_20260315/cqa_visthick_g2_s2000/eval_controls_128.json`
- target audit:
  - `results/cqa_audit/patchnepa_cqa_target_audit_20260315/summary.json`

Current branch interpretation:

- the explicit-query CQA branch trains and the mask / type-routing / vocab
  wiring are functioning,
- but the first `mesh_visibility + udf_thickness` 2k-step smoke is **not**
  evidence that promptable primitive-conditioned answering has been learned,
- it is a branch-level wiring confirmation plus a target-selection negative
  result.

Canonical 2k smoke readout (`128` eval shapes / task, current config:
`mesh_visibility + udf_thickness`, both with `context_source=surf`):

| control | overall `ce` | overall token acc | note |
|---|---:|---:|---|
| `correct` | `1.6237` | `0.6277` | train/eval path runs end-to-end |
| `no_context` | `1.8052` | `0.6260` | only small overall accuracy drop |
| `shuffled_context` | `1.6381` | `0.6285` | almost unchanged |
| `wrong_type` | `~1e9` | `0.0000` | type routing is active |
| `shuffled_query` | `1.7264` | `0.6282` | weak query sensitivity |

Per-task smoke readout:

| task | `ce` | token acc | dominant prediction |
|---|---:|---:|---|
| `mesh_visibility` | `1.9862` | `0.5875` | token `128` on `8191 / 8192` predictions |
| `udf_thickness` | `1.2613` | `0.6680` | token `448` on `8016 / 8192` predictions |

Majority-baseline check from the same evaluation slice:

- `mesh_visibility` target top-1 share:
  - token `128` on `4814 / 8192` targets (`0.5876`)
- `udf_thickness` target top-1 share:
  - token `448` on `5484 / 8192` targets (`0.6694`)
- interpretation:
  - the model is, at this stage, extremely close to a majority-code predictor
    on both tasks.

Target-audit summary (`256` shapes / task from
`data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1`):

| task | top-1 code share | entropy (bits) | practical read |
|---|---:|---:|---|
| `udf_distance` | `0.0607` | `5.6230` | healthiest next mainline candidate |
| `udf_clearance` (front-only in current code) | `0.6966` | `2.2096` | usable next diagnostic / secondary candidate |
| `udf_thickness` | `0.6636` | `1.8394` | conditional only; first-bin pressure is strong |

Additional raw-field audit notes:

- `udf_surf_thickness` quantiles show median `0.0`,
- `udf_surf_clear_front/back` both saturate at `2.0`,
- `mesh_visibility` still has a documented fallback path in
  `_visibility_signature(...)`, but fallback usage is **not** recorded per
  shape in the current cache, so the cache-level fallback rate remains unknown.

Decision boundary after this smoke:

- demote `mesh_visibility` from the next CQA headline smoke,
- move the next mainline CQA smoke to `surf -> udf_distance + udf_clearance`,
- run `pc_bank -> udf_distance + udf_clearance` only as a paired diagnostic,
- keep `udf_thickness` conditional on later entropy / control improvements.

## 64. CQA UDF-surf smoke completed; `udf_distance` is the first viable task, but context use is still weak (2026-03-15)

Canonical sources:

- ABCI log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfsurf_20260315/cqa_udfdist_clearfront_g2_s2000/eval_controls_128.json`
- target audit reference:
  - `results/cqa_audit/patchnepa_cqa_target_audit_20260315/summary.json`

Scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfsurf.yaml`
- tasks:
  - `udf_distance`
  - `udf_clearance` (`front` only in current code)
- context:
  - both use `context_source=surf`
- job:
  - `113068`
- W&B:
  - project `patchnepa-cqa-smoke`
  - run `patchnepa_cqa_udfsurf_20260315_cqa_udfdist_clearfront_g2_s2000`

Teacher-forced token eval (`128` eval shapes / task):

| control | overall `ce` | overall token acc |
|---|---:|---:|
| `correct` | `2.4268` | `0.4052` |
| `no_context` | `2.5955` | `0.3921` |
| `shuffled_context` | `2.4353` | `0.4042` |
| `wrong_type` | `~1e9` | `0.0000` |
| `shuffled_query` | `3.4468` | `0.3681` |

Per-task `correct` control:

| task | `ce` | token acc | same-slice majority baseline | read |
|---|---:|---:|---:|---|
| `udf_distance` | `3.3442` | `0.1093` | `0.0586` | beats majority and clearly uses query order / anchor |
| `udf_clearance(front)` | `1.5094` | `0.7012` | `0.7015` | still majority-code dominated |

Control deltas vs `correct`:

- `udf_distance`
  - `no_context`: `delta_ce=+0.2059`, `delta_acc=-2.66pt`
  - `shuffled_context`: `delta_ce=+0.0089`, `delta_acc=-0.22pt`
  - `shuffled_query`: `delta_ce=+2.0393`, `delta_acc=-7.42pt`
- `udf_clearance(front)`
  - `no_context`: `delta_ce=+0.1315`, `delta_acc=+0.04pt`
  - `shuffled_context`: `delta_ce=+0.0081`, `delta_acc=+0.02pt`
  - `shuffled_query`: `delta_ce=+0.0008`, `delta_acc=+0.00pt`

Interpretation:

- this is the first CQA smoke where one task (`udf_distance`) is clearly
  better than the same-slice majority baseline,
- `udf_distance` is also much more query-sensitive than the previous
  `mesh_visibility / udf_thickness` pair,
- but `surf` context is still not being used strongly enough:
  - `no_context` moves `udf_distance`, but not enough to call the task truly
    context-conditioned,
  - `shuffled_context` is almost a no-op,
- `udf_clearance(front)` remains majority-like and should not be promoted to a
  headline task yet.

Decision:

- keep `udf_distance` as the current best CQA task,
- keep `udf_clearance(front)` only as a secondary / diagnostic task,
- next run should be the paired diagnostic:
  - `pc_bank -> udf_distance + udf_clearance(front)`
- do **not** claim promptable cross-primitive answering from `C002`; this is
  still a task-selection / conditioning readout.

## 65. CQA `pc_bank` diagnostic completed; heterogeneous context did not strengthen conditioning (2026-03-15)

Canonical sources:

- ABCI log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfpcdiag_20260315/cqa_udfdist_clearfront_pcdiag_g2_s2000/eval_controls_128.json`

Scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`
- tasks:
  - `udf_distance`
  - `udf_clearance` (`front` only in current code)
- context:
  - both use `context_source=pc_bank`
- job:
  - `113071`

Teacher-forced token eval (`128` eval shapes / task):

| control | overall `ce` | overall token acc |
|---|---:|---:|
| `correct` | `2.3758` | `0.4167` |
| `no_context` | `2.5073` | `0.4058` |
| `shuffled_context` | `2.3833` | `0.4167` |
| `wrong_type` | `~1e9` | `0.0000` |
| `shuffled_query` | `3.3341` | `0.3781` |

Per-task `correct` control:

| task | `ce` | token acc | same-slice majority baseline | read |
|---|---:|---:|---:|---|
| `udf_distance` | `3.3388` | `0.1110` | `0.0586` | still beats majority and remains query-sensitive |
| `udf_clearance(front)` | `1.4128` | `0.7225` | `0.7015` | still majority-like despite slightly better fit |

Control deltas vs `correct`:

- `udf_distance`
  - `no_context`: `delta_ce=+0.1396`, `delta_acc=-2.20pt`
  - `shuffled_context`: `delta_ce=+0.0043`, `delta_acc=-0.01pt`
  - `shuffled_query`: `delta_ce=+1.9143`, `delta_acc=-7.74pt`
- `udf_clearance(front)`
  - `no_context`: `delta_ce=+0.1236`, `delta_acc=+0.00pt`
  - `shuffled_context`: `delta_ce=+0.0107`, `delta_acc=+0.00pt`
  - `shuffled_query`: `delta_ce=+0.0023`, `delta_acc=+0.00pt`

Comparison against the preceding `surf` diagnostic (`C002`):

- `udf_distance` remains the only viable task,
- but `pc_bank` did **not** increase context sensitivity:
  - `no_context delta_ce` weakened from `+0.2059` (`surf`) to `+0.1396`
  - `shuffled_context` remained near-zero in both settings
  - `shuffled_query` remained the dominant perturbation in both settings
- therefore the current branch still looks more **query-sensitive than
  context-sensitive**.

Decision:

- keep `udf_distance` as the branch's best task,
- keep `udf_clearance(front)` only as a secondary diagnostic,
- do not promote `pc_bank` as evidence that heterogeneous context has become
  useful,
- next changes should target stronger context-conditioned tasks or task
  definitions, not another immediate repeat of the same `pc_bank` diagnostic.

## 66. CQA `udf_distance` single-task smoke completed; context use strengthened but shape-conditioning remains weak (2026-03-15)

Canonical sources:

- ABCI log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_20260315/cqa_udfdist_g2_s2000/eval_controls_128.json`

Scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml`
- task:
  - `udf_distance` only
- context:
  - `context_source=surf`
- job:
  - `113102`

Teacher-forced token eval (`128` eval shapes):

| control | `ce` | token acc |
|---|---:|---:|
| `correct` | `2.9532` | `0.1552` |
| `no_context` | `3.5771` | `0.1006` |
| `wrong_shape` | `3.0633` | `0.1393` |
| `shuffled_context` | `3.0657` | `0.1361` |
| `wrong_type` | `~1e9` | `0.0000` |
| `shuffled_query` | `7.0616` | `0.0314` |

Per-task `correct` read:

| task | `ce` | token acc | majority baseline | prediction spread |
|---|---:|---:|---:|---|
| `udf_distance` | `2.9532` | `0.1552` | `0.0586` | `pred_top1_share=0.1093`, `pred_unique_codes=31` |

Control deltas vs `correct`:

- `no_context`: `delta_ce=+0.6238`, `delta_acc=-5.46pt`
- `wrong_shape`: `delta_ce=+0.1100`, `delta_acc=-1.59pt`
- `shuffled_context`: `delta_ce=+0.1125`, `delta_acc=-1.90pt`
- `shuffled_query`: `delta_ce=+4.1084`, `delta_acc=-12.38pt`

Interpretation:

- single-task `udf_distance` is the strongest CQA result so far:
  - it clearly beats the same-slice majority baseline,
  - it predicts across many bins rather than collapsing to one dominant code,
  - `no_context` now hurts much more than in the earlier mixed-task smoke.
- however, `wrong_shape` remains modest relative to `no_context`:
  - the branch is becoming more clearly **context-using**,
  - but it is still not strongly **shape-conditioned** yet.
- `wrong_type` and `shuffled_query` still dominate the controls, so the branch
  remains much more sensitive to the prompt/query side than to shape identity.

Decision:

- keep `udf_distance` as the main CQA task,
- next gate should be the hard-query pass:
  - `query_src_filter=near_surface`
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_near.yaml`
- do not rerun `pc_bank` yet; first test whether harder `udf_distance` queries
  can convert the current context use into stronger shape-conditioned behavior.

## 67. CQA `udf_distance` near-surface hard-query pass collapsed; the naive hard-query recipe is a negative result (2026-03-15)

Canonical sources:

- ABCI log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_near_20260315/cqa_udfdist_near_g2_s2000/eval_controls_128.json`

Scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_near.yaml`
- task:
  - `udf_distance` only
- context:
  - `context_source=surf`
- query subset:
  - `query_src_filter=near_surface`
- job:
  - `113125`

Teacher-forced token eval (`128` eval shapes):

| control | `ce` | token acc |
|---|---:|---:|
| `correct` | `3.6379` | `0.0417` |
| `no_context` | `3.6387` | `0.0409` |
| `wrong_shape` | `3.6377` | `0.0424` |
| `shuffled_context` | `3.6377` | `0.0409` |
| `wrong_type` | `~1e9` | `0.0000` |
| `shuffled_query` | `3.6380` | `0.0437` |

Per-task `correct` read:

| task | `ce` | token acc | majority baseline | prediction spread |
|---|---:|---:|---:|---|
| `udf_distance` | `3.6379` | `0.0417` | `0.0439` | `pred_top1_share=0.3959`, `pred_unique_codes=6` |

Control deltas vs `correct`:

- `no_context`: `delta_ce=+0.0008`, `delta_acc=-0.09pt`
- `wrong_shape`: `delta_ce=-0.0002`, `delta_acc=+0.06pt`
- `shuffled_context`: `delta_ce=-0.0002`, `delta_acc=-0.09pt`
- `shuffled_query`: `delta_ce=+0.0001`, `delta_acc=+0.20pt`

Interpretation:

- the naive `near_surface` hard-query pass is a clear negative result:
  - `udf_distance` no longer beats the same-slice majority baseline,
  - prediction spread collapses from the full-range run's `31` codes to `6`,
  - `wrong_shape`, `no_context`, and `shuffled_query` all become effectively
    inert.
- this does **not** mean `udf_distance` is fundamentally dead:
  - the full-range single-task run remains healthy,
  - but the current `query_src_filter=near_surface` recipe is not a useful way
    to increase context dependence.

Decision:

- keep the full-range single-task `udf_distance` result as the CQA branch's
  best evidence,
- record the naive `near_surface` filter as a negative task-spec result,
- only run the delayed `pc_bank` retry to confirm whether heterogeneous
  context can rescue this setting; do not promote it as a mainline candidate.

## 68. CQA `pc_bank` retry under near-surface queries is also negative; heterogeneous context still does not rescue the branch (2026-03-15)

Canonical sources:

- ABCI log:
  - `logs/cqa_pretrain/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000.pbs.log`
- save root:
  - `runs/cqa/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000`
- control eval:
  - `runs/cqa/patchnepa_cqa_udfdist_near_pcdiag_20260315/cqa_udfdist_near_pcdiag_g2_s2000/eval_controls_128.json`

Scope:

- config:
  - `nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_near_pcdiag.yaml`
- task:
  - `udf_distance` only
- context:
  - `context_source=pc_bank`
- query subset:
  - `query_src_filter=near_surface`
- job:
  - `113127`

Teacher-forced token eval (`128` eval shapes):

| control | `ce` | token acc |
|---|---:|---:|
| `correct` | `3.6426` | `0.0433` |
| `no_context` | `3.6432` | `0.0442` |
| `wrong_shape` | `3.6427` | `0.0435` |
| `shuffled_context` | `3.6427` | `0.0442` |
| `wrong_type` | `~1e9` | `0.0000` |
| `shuffled_query` | `3.6426` | `0.0431` |

Per-task `correct` read:

| task | `ce` | token acc | majority baseline | prediction spread |
|---|---:|---:|---:|---|
| `udf_distance` | `3.6426` | `0.0433` | `0.0441` | `pred_top1_share=0.4690`, `pred_unique_codes=7` |

Control deltas vs `correct`:

- `no_context`: `delta_ce=+0.0005`, `delta_acc=+0.09pt`
- `wrong_shape`: `delta_ce=+0.0001`, `delta_acc=+0.01pt`
- `shuffled_context`: `delta_ce=+0.0001`, `delta_acc=+0.09pt`
- `shuffled_query`: `delta_ce=-0.0000`, `delta_acc=-0.02pt`

Interpretation:

- `pc_bank` does not rescue the failed `near_surface` setting:
  - `udf_distance` remains below majority,
  - prediction spread is even more concentrated than the `surf` hard-query run,
  - all context/query controls except `wrong_type` are effectively inert.
- therefore the current negative read is not specific to the `surf` carrier:
  - the problem is the present hard-query/task recipe itself,
  - not merely the lack of heterogeneous context.

Decision:

- stop repeating `pc_bank` diagnostics on the current CQA task recipe,
- keep full-range single-task `udf_distance` as the only viable CQA task so
  far,
- any next CQA iteration should change task/target design before revisiting
  heterogeneous context claims.

## 69. `world_v3` freeze sprint completed on the existing `worldvis` cache; no full rebuild is justified yet (2026-03-16)

Canonical sources:

- freeze log:
  - `logs/preprocess/world_v3_freeze/world_v3_freeze_20260316_qc/world_v3_freeze_20260316_qc.out`
- augmentation summary:
  - `results/data_freeze/world_v3_freeze_20260316_qc/augment_world_v3_summary.json`
- audit summary:
  - `results/data_freeze/world_v3_freeze_20260316_qc/world_v3_audit_summary.json`
- clean-subset manifest:
  - `results/data_freeze/world_v3_freeze_20260316_qc/subset_watertight_manifest.json`
- clean-subset summary:
  - `results/data_freeze/world_v3_freeze_20260316_qc/subset_watertight_summary.json`
- schema spec:
  - `nepa3d/docs/patch_nepa/spec_world_v3_schema.md`

Scope:

- existing cache:
  - `data/shapenet_cache_v2_20260311_worldvis`
- worker:
  - `scripts/preprocess/freeze_shapenet_world_v3_qc.sh`
- submit wrapper:
  - `scripts/preprocess/submit_world_v3_freeze_qc.sh`
- ABCI CPU job:
  - `113209`

Summary:

- `world_v3` augmentation completed in place with no recorded errors:
  - `updated=39761`
  - `skipped=12550`
  - `errors=[]`
- shape-quality audit on the frozen cache:
  - `total shapes = 52311`
  - `watertight_rate = 0.002409`
  - `winding_consistent_rate = 0.586263`
  - `visibility_fallback_used_rate = 0.0`
- task-validity means:
  - `udf_surf_hit_out_rate_mean = 0.744845`
  - `udf_clear_front_valid_rate_mean = 0.744845`
  - `udf_clear_back_valid_rate_mean = 0.773156`
  - `udf_probe_valid_rate_mean = 1.0`
- strict clean subset (`watertight + winding_consistent`) is tiny:
  - `kept = 119 / 52311`
  - `train = 108`
  - `test = 11`

Interpretation:

- the existing `worldvis` cache is good enough to freeze as `world_v3`;
  contract-level defects did **not** appear in the freeze sprint.
- therefore a full corpus rebuild is not the right next move.
- the strict watertight subset is far too small to replace the main corpus:
  it should be treated as a future pivot / clean-subset manifest, not as the
  main training cache.
- from this point onward, CQA and recon results should be described against the
  fixed `world_v3` contract rather than an implicitly moving cache schema.

Decision:

- mark Phase 1 dataset freeze as complete,
- keep `data/shapenet_cache_v2_20260311_worldvis` as the base source cache,
- use the strict clean subset only as a side manifest for future watertight /
  neural-field pivots,
- do not escalate to a full rebuild unless a future task requires fields that
  the `world_v3` contract cannot support.

## 70. Frozen-`world_v3` CQA revalidation is positive; `udf_distance` is now the canonical surviving branch task (2026-03-16)

Canonical sources:

- rerun save root:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_reval_20260316/cqa_udfdist_worldv3_g2_s2000`
- rerun eval:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_reval_20260316/cqa_udfdist_worldv3_g2_s2000/eval_controls_128.json`
- curve save root:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000`
- curve final eval:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/eval_controls_128.json`
- curve checkpoints:
  - `runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/eval_curve_128/curve_summary.json`

Key read:

- `C008` exact rerun reproduced the branch on frozen `world_v3`:
  - `acc=0.1064` vs majority `0.0376`
  - `pred_top1_share=0.2118`
  - `pred_unique_codes=33`
  - `delta_ce(no_context)=+1.0599`
  - `delta_ce(wrong_shape_same_synset)=+0.3193`
  - `delta_ce(wrong_shape_other_synset)=+1.1527`
- `C009` 10k run strengthened the same task substantially:
  - final `acc=0.3173` vs majority `0.0376`
  - `pred_top1_share=0.1339`
  - `pred_unique_codes=40`
  - `delta_ce(no_context)=+6.5064`
  - `delta_ce(wrong_shape_same_synset)=+4.1433`
  - `delta_ce(wrong_shape_other_synset)=+10.2148`
- query-source breakdown:
  - `uniform` is already easy (`acc=0.5597`)
  - `near_surface` remains much weaker (`acc=0.0818`)

Interpretation:

- the earlier positive `udf_distance` signal was **not** a provisional
  pre-freeze artifact; it reproduces on the fixed `world_v3` contract.
- more importantly, the 10k curve crosses the earlier barrier:
  the branch is no longer merely `query-conditioned` or weakly
  `context-conditioned`; the `wrong_shape_same/other_synset` controls now move
  strongly, so shape-conditioned use is genuinely emerging.
- this does **not** promote CQA above `recong2/composite`, but it upgrades
  `udf_distance` from a fragile side signal to a real surviving branch.

Decision:

- keep `recong2/composite` as the paper mainline,
- keep CQA alive with `udf_distance` as the only canonical task for now,
- do **not** reopen `mesh_visibility`, `udf_thickness`, `clearance(front)`, or
  naive `near_surface` as headline tasks yet,
- any next CQA step should build on the frozen `world_v3` contract and the
  now-positive `udf_distance` branch rather than restarting from broader task
  sweeps.

## 71. Zero-shot `pc_bank -> udf_distance` off-diagonal transfer is positive on frozen `world_v3` (2026-03-16)

Canonical sources:

- off-diagonal eval root:
  - `results/cqa_eval/patchnepa_cqa_udfdist_offdiag_20260316_171008`
- eval json:
  - `results/cqa_eval/patchnepa_cqa_udfdist_offdiag_20260316_171008/cqa_udfdist_offdiag_eval.json`
- log:
  - `logs/cqa_eval/patchnepa_cqa_udfdist_offdiag_20260316_171008/cqa_udfdist_offdiag_eval.pbs.log`

Key read:

- eval-only zero-shot off-diagonal (`train=surf`, `eval=pc_bank`) stays well above
  the held-out majority baseline:
  - `acc=0.1766` vs majority `0.0397`
  - `pred_top1_share=0.0757`
  - `pred_unique_codes=39`
- controls remain meaningfully positive:
  - `delta_ce(no_context)=+4.4730`
  - `delta_ce(wrong_shape_same_synset)=+1.8570`
  - `delta_ce(wrong_shape_other_synset)=+4.6121`
  - `delta_ce(shuffled_query)=+5.8515`
- query-source breakdown:
  - `uniform acc=0.3169`
  - `near_surface acc=0.0338`

Interpretation:

- this is the first clean off-diagonal result for the CQA branch:
  the `surf -> udf_distance` checkpoint transfers zero-shot to `pc_bank`
  without paired cross-primitive training.
- the transfer is not broad across all query regimes, because `near_surface`
  remains weak, but the off-diagonal signal itself is now real rather than
  speculative.
- therefore the branch is no longer just "same/oracle context only":
  shared-context transfer exists, albeit on the easier broad query regime.

Decision:

- promote zero-shot `pc_bank -> udf_distance` as the next canonical CQA result,
- keep paired `pc_bank -> udf_distance` training as a later upper-bound
  diagnostic rather than a mainline move,
- continue to treat `near_surface` as a failure mode rather than reopening it
  immediately.

## 72. Dense-grid `udf_distance` completion pilot is positive; CQA now has method-native completion evidence (2026-03-16)

Canonical sources:

- completion root:
  - `results/cqa_completion/patchnepa_cqa_udfdist_completion_20260316_171008`
- completion json:
  - `results/cqa_completion/patchnepa_cqa_udfdist_completion_20260316_171008/cqa_udfdist_completion_grid12.json`
- log:
  - `logs/cqa_completion/patchnepa_cqa_udfdist_completion_20260316_171008/cqa_udfdist_completion_grid12.pbs.log`

Key read:

- dense-grid pilot on `16` held-out shapes (`grid_res=12`) is stable:
  - `MAE = 0.0479 +- 0.0103`
  - `RMSE = 0.0638 +- 0.0132`
  - `IoU@0.01 = 0.0814`
  - `IoU@0.02 = 0.1850`
  - `IoU@0.05 = 0.5712`
- shape-level spread is non-trivial but sane:
  - `MAE min/max = 0.0308 / 0.0674`
  - `RMSE min/max = 0.0476 / 0.0899`

Interpretation:

- `udf_distance` CQA is no longer only a token-accuracy story.
  The same checkpoint can be decoded back into a dense field with meaningful
  completion metrics on held-out shapes.
- this is still a modest pilot (`grid_res=12`, no mesh-level MC metrics yet),
  but it is enough to establish method-native answering/completion rather than
  a purely probe-style readout.

Decision:

- keep `udf_distance` as the only canonical CQA task for now,
- treat dense-grid completion as established pilot evidence,
- postpone larger-grid or mesh-level sweeps until the paper-story placement is
  finalized.

## 73. `udf_distance` mainline and zero-shot off-diagonal reads reproduce across seeds; CQA is no longer a single-seed branch (2026-03-16)

Canonical sources:

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

Key read:

- mainline `surf -> udf_distance` reproduces strongly on both new seeds:
  - seed `1`: `acc=0.3330` vs majority `0.0385`, `pred_top1_share=0.1210`,
    `pred_unique_codes=41`
  - seed `2`: `acc=0.3446` vs majority `0.0383`, `pred_top1_share=0.1404`,
    `pred_unique_codes=40`
- same-shape controls remain strongly positive:
  - seed `1`: `delta_ce(no_context)=+8.3455`,
    `delta_ce(wrong_shape_same)=+4.5927`,
    `delta_ce(wrong_shape_other)=+10.9698`
  - seed `2`: `delta_ce(no_context)=+6.2504`,
    `delta_ce(wrong_shape_same)=+4.2873`,
    `delta_ce(wrong_shape_other)=+10.7701`
- zero-shot `pc_bank -> udf_distance` off-diagonal also reproduces:
  - seed `1`: `acc=0.1584` vs majority `0.0402`, `pred_top1_share=0.0654`,
    `pred_unique_codes=41`
  - seed `2`: `acc=0.1620` vs majority `0.0379`, `pred_top1_share=0.0834`,
    `pred_unique_codes=40`
- off-diagonal controls remain clearly positive:
  - seed `1`: `delta_ce(no_context)=+6.2622`,
    `delta_ce(wrong_shape_same)=+1.7640`,
    `delta_ce(wrong_shape_other)=+4.6542`
  - seed `2`: `delta_ce(no_context)=+3.9062`,
    `delta_ce(wrong_shape_same)=+1.6813`,
    `delta_ce(wrong_shape_other)=+4.0375`

Interpretation:

- the `udf_distance` branch is no longer supported by a single favorable run.
  Seed `0` (`C009/C010`) and seeds `1/2` now all agree on the same read:
  strong same-context learning, positive shape-conditioned controls, and
  reproducible zero-shot off-diagonal transfer to `pc_bank`.
- this removes the main reason to immediately launch paired
  `pc_bank -> udf_distance` training. A paired upper bound remains useful as a
  reviewer-facing diagnostic, but it is no longer the next default step.

Decision:

- promote `udf_distance` CQA from a promising single-seed branch to a stable
  paper-method branch,
- keep zero-shot off-diagonal transfer as canonical evidence rather than
  replacing it with paired cross-primitive training,
- defer the paired `pc_bank` upper bound until a later reviewer/diagnostic need
  rather than treating it as the next mandatory experiment.

## 74. Paired `pc_bank -> udf_distance` upper bound is positive, but it stays a diagnostic rather than the main claim (2026-03-16)

Canonical sources:

- paired upper-bound root:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000`
- final eval:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000/eval_controls_128.json`
- curve summary:
  - `runs/cqa/patchnepa_cqa_udfdist_pcbank_upper_20260316_182341/cqa_udfdist_pcbank_upper_g2_s10000/eval_curve_128/curve_summary.json`

Key read:

- paired `pc_bank -> udf_distance` is strongly above majority:
  - `acc=0.2533` vs majority `0.0375`
  - `pred_top1_share=0.1112`
  - `pred_unique_codes=41`
- controls remain meaningfully positive:
  - `delta_ce(no_context)=+5.5981`
  - `delta_ce(wrong_shape_same)=+1.8220`
  - `delta_ce(wrong_shape_other)=+4.5715`
  - `delta_ce(shuffled_query)=+7.8943`
- query-source breakdown:
  - `uniform acc=0.4417`
  - `near_surface acc=0.0676`

Interpretation:

- the paired upper bound behaves as expected: it improves over the zero-shot
  off-diagonal checkpoint (`0.2533` vs `0.1766`) and shows that the
  `pc_bank -> udf_distance` task itself is learnable under the same model and
  recipe.
- however, this does **not** replace the cleaner zero-shot story. The paired
  result is best used as a diagnostic ceiling and reviewer-facing reference,
  not as the central evidence for shared-context transfer.

Decision:

- keep zero-shot `surf-trained -> pc_bank eval-only` as the main off-diagonal
  claim,
- retain paired `pc_bank -> udf_distance` as an upper-bound / diagnostic row in
  tables or appendix,
- do not let the paired result pre-empt the cleaner transfer narrative.
