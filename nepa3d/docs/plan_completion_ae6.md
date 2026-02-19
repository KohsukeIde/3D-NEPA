# Completion A-E / 6 Execution Plan (Feb 16, 2026)

This plan converts the feedback into an execution order that matches current repo state.

## Scope decision

- Primary track: **CPAC completion** (`pointcloud_noray -> udfgrid/udf`) for main claims.
- Retrieval: keep as appendix/diagnostic.
- Goal: improve `grid` and mixed-query completion while keeping current strong `pool` performance.

## Current status

- `6 (scaling hooks)` is merged: `max_len`, pos-emb resize, `n_point/n_ray` schedules.
- `6-3 (Encoder-Decoder split for scaling)` is **not implemented yet** in this branch (current backbone is still single-stream causal transformer).
- A-minimal eval controls are merged in `completion_cpac_udf.py`:
  - `query_source=hybrid`
  - `grid_sample_mode={uniform,near_surface,stratified,coarse_to_fine}`
  - `grid_res_schedule`, `grid_c2f_expand`, `grid_c2f_stage_weights`
  - `target_transform={none,trunc,log1p}`
  - near-only metrics (`near@tau_report`)
- A-2 status clarification:
  - eval/probe-side query mix (`query_source=hybrid`) is implemented and evaluated.
  - pretrain-side query-mix in `nepa3d/train/pretrain.py` is not implemented in this branch yet.
- First smoke confirms:
  - `grid near_surface` > `grid uniform`
  - trunc helps near-surface metrics but changes raw-MAE interpretation.
- Full A-minimal eval matrix (`max_shapes=800`, `htrain4k`, `eval_seed=0,1,2`) is completed and logged in `results_ucpr_cpac_active.md`.
- A-1 (`octree/coarse-to-fine` query staging) is now implemented in eval-time grid sampling and quick-validated at full CPAC protocol (`seed0`, `max_shapes=800`, `htrain4k`, `n_context=512`, `n_query=256`):
  - `uniform`: `MAE=0.03079`, `RMSE=0.03889`, `IoU@0.03=0.28000`
  - `near_surface`: `MAE=0.02042`, `RMSE=0.02657`, `IoU@0.03=0.48901`
  - `coarse_to_fine(16->32->64)`: `MAE=0.02318`, `RMSE=0.02950`, `IoU@0.03=0.48444`
  - readout: A-1 gives a large gain over uniform and reaches near-surface-level IoU, but does not yet beat tuned `near_surface` on this checkpoint.
- B-1 probe-time Lipschitz pilot is implemented in `completion_cpac_udf.py` and evaluated.
  - Status: no gain on current `grid_near08` setting; keep default OFF (`ridge_lipschitz_lambda=0`) for main runs.
- B-2/B-3 pretrain hooks are integrated in `nepa3d/train/pretrain.py`:
  - B-2: ray hit/depth supervision + depth-rank hinge (`--aux_b2_*`)
  - B-3: near-surface point-distance target (`--aux_b3_*`)
- Smoke validation (`mix_num_samples=256`, `epochs=1`) and tiny CPAC check (`max_shapes=120`) completed.
  - Result: pipeline is stable; proceed to fixed-size full pilot at `n_point=n_ray=256`.
- B-3 full-protocol fill (`max_shapes=800`, `htrain4k`) completed:
  - B-3-only remains below ref/B-2 on both pool and grid.
  - B-2+B-3 improves over ref but stays below B-2-only.
- B-2 longer confirm (seed0, `mix_num_samples=20000`, continuation to `ep054`) completed:
  - CPAC pool improved vs `ep049`.
  - UCPR hard-pair regressed; keep as side-effect guardrail during C.
- C-0 hook (minimal C prototype) added:
  - teacher-student refresh distillation: `--teacher_ckpt`, `--teacher_distill_weight`
  - smoke run passed.
- C-0 seed0 quick compare completed:
  - underperforms B-2 on both CPAC `pool` and `grid_near08`.
  - keep C-0 as plumbing baseline, not as primary C candidate.
- C-1/C-2 hooks are integrated in `nepa3d/train/pretrain.py`:
  - C-1: teacher refresh with answer-drop (`--teacher_answer_drop_prob`)
  - C-2: cycle consistency across answer-drop views (`--cycle_weight`, `--cycle_answer_drop_prob`)
- C-1/C-2 seed0 quick compare completed (`ep049 -> ep050`, `mix_num_samples=8000`):
  - C-2 is current best on CPAC `pool` and beats B-2 there.
  - On `grid_near08`, C-2 improves over ref/C-0 but is still below B-2.
  - Keep multi-seed postponed until this fixed-size direction is finalized.
- B-2 + C-2 seed0 joint confirm completed (`ep049 -> ep050`, `mix_num_samples=8000`):
  - best so far on CPAC `pool` (better than B-2-only and C-2-only).
  - partial recovery on `grid_near08` vs C-2, still below B-2-only.
  - UCPR hard-pair remains mixed but within acceptable guardrail for completion-first track.
- D/E prototypes are now integrated and seed0 quick compared (`ep049 -> ep050`, single-factor):
  - D-only (`--d_hard_*`) and E-only (`--aux_e_weight`) both improved vs ref `ep049`.
  - both remained below `B-2+C-2 ep050` on pool and grid.
- 6 quick scale pilot completed on top of `B-2+C-2`:
  - short schedule `n_point: 256 -> 512` with auto `max_len` expansion.
  - under scaled eval context (`n_context=512, n_query=256`), both pool and grid improved vs pre-scale.
- 6 longer scale attempt (`... -> 1024`) was executed and logged:
  - this attempt regressed strongly vs quick-scale (`ep051`) on both pool and grid.
  - keep quick-scale checkpoint as best current seed0 scale candidate.
- 6 stability-adjusted scale retry (`ep053`) was executed from `scalequick(ep051)`:
  - error metrics recovered vs the unstable long attempt on most settings.
  - `512/pool` IoU stayed below quick-scale, so overall best seed0 remains `ep051`.
- 6 additional scale retry sweep (Feb 20, 2026, `v3`) completed in parallel:
  - `w64_lr2e-4_v3` and `w96_lr1e-4_v3` from `scalequick(ep051)` to `ep054`.
  - both runs completed full CPAC (`pc512/pc1024`, pool + grid_near08) and UCPR guardrail.
  - `w96_lr1e-4_v3` > `w64_lr2e-4_v3`, but both are below prior `scalequick(ep051)` and `scale_stab(ep053)`.
  - keep as diagnostic references; do not promote to main candidate.
- Coverage status:
  - A/B/C/D/E/6 all have at least one seed0 full-protocol result set recorded.

## Priority order (execute in this order)

1. **B-2 (ray monotonicity) if ray supervision is available**  
   Add ranking/monotonic loss along sampled ray point order.
   Run at fixed `n_point=n_ray=256` first to isolate algorithmic gains.

2. **B-3 (near-surface geometry target) prototype**  
   Add auxiliary near-surface normal/geometry target (or equivalent local-geometry signal).
   Evaluate with the same CPAC protocol (`htrain4k`, non-trans) before moving to pseudo-parallel training.

3. **C (back-translation/cycle) prototype**  
   Start with 1-step pseudo-parallel refresh (teacher-student), then consider cycle losses.
   Keep fixed-size (`256/256`) for first comparison.

4. **D (hard-query / adversarial sampling) prototype**  
   Add hard point mining from current model errors (or uncertainty), then re-train/eval.
   Keep protocol fixed (`htrain4k`, non-trans) to compare directly with A/B/C results.

5. **E (decoder-side completion head) branch**  
   Add an end-to-end implicit decoder branch and compare against current ridge probe.
   Keep this as separate branch so current probe protocol remains reproducible.

6. **6 (scale) full train/eval (final stage)**  
   Apply scaling only to the best B/C/D/E configuration:
   - baseline: fixed `n_point=n_ray=256`
   - scaled: `n_point_schedule="0:256,10:512,20:1024"` (then 2048 if needed) with compatible `max_len`
   This reduces compute waste and avoids confounding early ablations.

## Recommended experiment matrix (minimum publishable)

- Seeds: `0/1/2` for evaluation (and at least seed0+seed1 pretrain for scaled runs).
- Core rows (main table):
  - NEPA QA+dualmask, `query_source=pool`, `grid`, `hybrid`
  - MAE objective baseline, same settings
  - NN-copy baseline in each setting
- Core controls:
  - `context_mode_test={normal,none,mismatch}`
  - near-only metrics at `report_near_tau=0.05`

## Reporting rules

- For `target_transform=trunc/log1p`, report:
  - raw-space metrics (`mae/rmse/iou@tau`)
  - transformed-space metrics (`metrics_transformed`)
  - near-only metrics (`near@tau_report`)
- Do **not** mix transformed and non-transformed numbers in one main ranking without a clear label.

## Immediate next run (updated)

Next immediate run:

- Freeze `B-2+C-2 + scalequick(ep051)` as the current best seed0 direction.
- Keep `scale_long(ep054)` and `scale_stab(ep053)` as reference runs (not promotion targets).
- Keep `scale_retry_v3` (`w64_lr2e-4`, `w96_lr1e-4`) as additional reference-only runs.
- Multi-seed remains intentionally deferred in the current policy.
- Next run should therefore focus on a new algorithmic delta (A/B/C/D/E) on top of `ep051`, not another scale-only continuation.

## Feedback-driven fix priority (added Feb 19, 2026)

Before expanding ablations, prioritize these fixes to reduce interpretation risk:

1. Scale-transition stability at `512 -> 1024`
- keep optimizer state for non-resized parameters when `pos_emb` is resized.
- avoid full optimizer-state reset unless explicitly intended.
- add transition diagnostics in logs: token length, gradient norms, and per-stage schedule state.

2. EncDec training-path sanity
- enforce decoder causal behavior for answer decoding path.
- verify no unintended answer-token leakage path in training/eval extraction.
- keep `qa_layout=split` explicit in metadata and result rows.

3. A-1 interpretation guardrail
- report A-1 as a query-design gain vs `grid uniform`, not as an unconditional win over tuned `near_surface`.
- keep `grid_res_schedule`, `grid_c2f_expand`, and `query budget` in protocol signatures for all A-1 rows.

4. Reporting hygiene
- keep retrieval column naming as `MRR (= single-positive mAP)` in all new tables.
- include protocol signatures in captions/row notes (`eval_seed_gallery`, `head_train_split`, `query_source`, `grid_sample_mode`, etc.).

### Fix status (Feb 19, 2026 update)

- Implemented:
  - resume-time optimizer partial restore for `pos_emb` resize
    - new arg: `--resume_optimizer_partial` (default `1`)
    - behavior: drop only shape-mismatched optimizer slots; keep others.
  - plusgut chain launch is now independent by default
    - `WAIT_FOR_PRIOR=0` default in plusgut chain runners.
    - launchers support `RUN_ID` suffix to keep rerun logs/pids separate.
- Remaining:
  - add explicit scale-transition diagnostics (token length / grad norms) to training logs.
  - run dedicated fresh pretrain for `encdec_plusgut_bbox` (current bbox is diagnostic ckpt-path variant).
