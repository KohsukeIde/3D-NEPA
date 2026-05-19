# ViewAction-NEPA

Phase-0/1 implementation scaffold for object-level 3D action-conditioned latent prediction.

This directory is designed to be added to `KohsukeIde/3D-NEPA` as a new track. It deliberately does **not** modify existing `PointGPT/`, `pointnepa/`, or `posetnepa_mask_order_preflight/` code.

ABCI Phase-0/1 smoke results are tracked in
`docs/phase01_abci_results_active.md`.

## Core question

Can an object-level 3D encoder learn a latent state that predicts the next partial observation under a sensor action?

We test this with multi-view partial point clouds:

```text
partial view X_t + camera action a_t  ->  latent prediction of next partial view X_{t+1}
```

## Why this track exists

Previous PointNEPA / PosetNEPA diagnostics showed that ordered latent AR is entangled with local continuity and position/center side channels. Rather than keep patching immediate-next-token AR, this track changes the prediction problem: learn controlled transitions between partial observations.

## Phase-0/1 goal

Do not start with ScanObjectNN. First verify the internal world-model-like behavior:

1. Build ShapeNet multi-view partial point clouds.
2. Train action-conditioned latent transition model.
3. Compare action-conditioned vs no-action vs shuffled-action.
4. Evaluate next-view retrieval and goal-view planning.
5. Only then move to ScanObjectNN / linear probe / full fine-tune.

## Quick start

From the 3D-NEPA repo root:

```bash
# 0. Create a tiny HPR/z-buffer-style multi-view cache
bash viewaction_nepa/scripts/00_build_views_hpr.sh

# 1. Train action-conditioned smoke model
bash viewaction_nepa/scripts/01_train_smoke_forward.sh

# 2. Train no-action baseline
bash viewaction_nepa/scripts/02_train_noaction_baseline.sh

# 3. Evaluate next-view retrieval
bash viewaction_nepa/scripts/03_eval_next_view_retrieval.sh

# 4. Evaluate goal-view planning
bash viewaction_nepa/scripts/04_eval_goal_planning.sh
```

On ABCI, the single-node smoke wrapper is:

```bash
bash viewaction_nepa/scripts/abci/submit_viewaction_phase01_smoke_qf.sh
```

## Important defaults

- View graph: 12 icosahedron camera positions.
- Action graph: directed k-nearest-neighbor edges on the view sphere.
- Default data: `/groups/gag51402/datasets/ShapeNet55-34/shapenet_pc`
  with official ShapeNet55 train/test files.
- Action encoding: shared local relative action id + relative camera motion
  vector. Absolute source/target camera directions are intentionally excluded
  to avoid target-view leakage.
- Partial views: z-buffer-style visibility fallback; Open3D HPR optional.
- Target encoder: EMA, momentum 0.996.
- Predictor: simple concat MLP to start.
- Inverse model: included with default weight 0.1.

## Smoke success criterion

The first go/no-go is not ScanObjectNN. It is:

```text
action-conditioned next-view retrieval > no-action by a clear margin,
shuffled-action drops,
goal-view planning reduces graph distance to target view,
2-step rollout does not immediately collapse.
```

See `docs/pf_kill_tests_active.md` for thresholds.
