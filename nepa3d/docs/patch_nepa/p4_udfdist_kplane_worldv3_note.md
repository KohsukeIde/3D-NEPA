# P4 udf_distance world_v3 plane baselines

This note freezes the minimum external/control baseline policy for the current
CQA `udf_distance` line.

Why P4 is still useful even though the task is original:
- the goal is **not** to claim a community-standard leaderboard,
- the goal is to avoid a self-serving task narrative by showing that a distinct
  geometric baseline family can be evaluated under the same world_v3 carriers
  and the same same/off-diagonal/completion protocol.

Recommended baseline family:
- `KPlaneRegressor` with `fusion=product` (k-plane)
- `KPlaneRegressor` with `fusion=sum` (tri-plane)

Current implementation path added here:
- `nepa3d/tracks/kplane/data/udfdist_worldv3_dataset.py`
- `nepa3d/tracks/kplane/train/pretrain_udfdist_worldv3.py`
- `nepa3d/tracks/kplane/analysis/eval_udfdist_worldv3_controls.py`
- `nepa3d/tracks/kplane/analysis/completion_udfdist_worldv3.py`
- wrappers:
  - `scripts/pretrain/nepa3d_kplane_udfdist_worldv3_qg.sh`
  - `scripts/analysis/nepa3d_kplane_udfdist_controls_qg.sh`
  - `scripts/analysis/nepa3d_kplane_udfdist_translation_qg.sh`

Interpretation boundary:
- these are **control baselines**, not a claim that the task has an accepted
  public benchmark.
- they should be compared primarily on:
  - same-context performance
  - zero-shot off-diagonal transfer
  - completion / meshization quality
  - control sensitivity (`no_context`, `wrong_shape_*`, `shuffled_query`)

Suggested paper positioning:
- keep `recong2/composite` as internal bridge baseline,
- keep plane baselines as external/control baselines,
- do **not** oversell them as canonical public baselines for the task.
