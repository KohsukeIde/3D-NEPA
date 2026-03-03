# Patch-NEPA Restart Plan (Data-v2 + CPAC Alignment)

Last updated: 2026-03-03

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
