# Operational criteria for using "world model"

The term "world model" should be used only if the implementation demonstrates the following properties.

## C1. Partial observation

The input must be a partial observation, not the full object. Each `X_t` is a visible partial point cloud from a camera viewpoint.

## C2. Action-conditioned transition

The model predicts the next latent state using an explicit action:

```text
z_t, a_t -> z_{t+1}
```

No-action and shuffled-action baselines must underperform.

## C3. Multi-step rollout

The model must support at least 2-step rollout:

```text
z_t -> z_{t+1} -> z_{t+2}
```

and must not immediately collapse.

## C4. Goal-view planning

Given current view `X_t`, target view `X_g`, and candidate actions, the model should choose the action that moves the latent state toward the target view.

## C5. Downstream transfer is not the only evidence

ScanObjectNN / ShapeNetPart are secondary. The primary smoke metrics are next-view retrieval, goal-view planning, and rollout consistency.

## Minimum world-model evidence for Phase 1

- next-view retrieval: action-conditioned > no-action by at least 10–15 top-1 points;
- shuffled-action lower than action-conditioned;
- goal-view planning reduces graph distance more than no-action;
- 2-step rollout retrieves the correct view significantly above chance.
