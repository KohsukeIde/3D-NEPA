# Completion A-E / 6 Execution Plan (Feb 16, 2026)

This plan converts the feedback into an execution order that matches current repo state.

## Scope decision

- Primary track: **CPAC completion** (`pointcloud_noray -> udfgrid/udf`) for main claims.
- Retrieval: keep as appendix/diagnostic.
- Goal: improve `grid` and mixed-query completion while keeping current strong `pool` performance.

## Current status

- `6 (scaling hooks)` is merged: `max_len`, pos-emb resize, `n_point/n_ray` schedules.
- A-minimal eval controls are merged in `completion_cpac_udf.py`:
  - `query_source=hybrid`
  - `grid_sample_mode={uniform,near_surface,stratified}`
  - `target_transform={none,trunc,log1p}`
  - near-only metrics (`near@tau_report`)
- First smoke confirms:
  - `grid near_surface` > `grid uniform`
  - trunc helps near-surface metrics but changes raw-MAE interpretation.
- Full A-minimal eval matrix (`max_shapes=800`, `htrain4k`, `eval_seed=0,1,2`) is completed and logged in `results_ucpr_cpac_active.md`.
- B-1 probe-time Lipschitz pilot is implemented in `completion_cpac_udf.py` and evaluated.
  - Status: no gain on current `grid_near08` setting; keep default OFF (`ridge_lipschitz_lambda=0`) for main runs.

## Priority order (execute in this order)

1. **6 (scale) full train/eval**  
   Run long pretrain with size curriculum (same objective, seed0 first):
   - baseline: fixed `n_point=n_ray=256`
   - scaled: `n_point_schedule="0:256,10:512,20:1024"` with compatible `max_len`
   Compare CPAC pool/grid + NN-copy.

2. **B-2 (ray monotonicity) if ray supervision is available**  
   Add ranking/monotonic loss along sampled ray point order.
   Keep as optional loss term and test after one full scale run is complete.

3. **C (back-translation/cycle) prototype**  
   Start with 1-step pseudo-parallel refresh (teacher-student), then consider cycle losses.
   This is high-complexity and should start only after A/B/6 are stable.

4. **E (decoder-side completion head) branch**  
   Add an end-to-end implicit decoder branch and compare against current ridge probe.
   Keep this as separate branch so current probe protocol remains reproducible.

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

Use `scripts/pretrain/nepa3d_pretrain.sh` for scale run + existing CPAC eval wrappers:

- `MAX_LEN=8192`
- `N_POINT=256`
- `N_RAY=256`
- `N_POINT_SCHEDULE=\"0:256,10:512,20:1024\"`
- `N_RAY_SCHEDULE=\"0:256,20:256\"`
- `RESUME_OPTIMIZER=0` (safe for pos-emb size changes)
- then evaluate with CPAC:
  - `QUERY_SOURCE=pool` (main)
  - `QUERY_SOURCE=grid`, `GRID_SAMPLE_MODE=near_surface` (bottleneck track)
