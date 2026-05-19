# Research framing: ViewAction-NEPA

## One-line thesis

Object-level 3D SSL should not only reconstruct masked geometry or predict ordered patches. It should learn how partial observations change under sensor actions.

## Central claim

We train a point-cloud encoder to form latent states that support action-conditioned prediction between partial object views:

\[
\hat z_{t+1} = F(z_t, a_t), \quad z_t = E(X_t)
\]

where `X_t` and `X_{t+1}` are partial point clouds from different viewpoints, and `a_t` is a known camera/viewpoint action.

## What this is not

- Not a pure rotation-augmentation method.
- Not a ScanObjectNN-first SSL benchmark paper.
- Not a PointGPT patch-ordering variant.
- Not a claim that SO(3) rotation alone is a world model.
- Not a replacement for scene-level world models such as video JEPA.

## What this is

A controlled object-level setting where we can operationally test world-model-like behavior:

1. partial observation;
2. action-conditioned transition;
3. no-action / shuffled-action controls;
4. multi-step rollout;
5. goal-view planning.

## Why this avoids the earlier PointNEPA / PosetNEPA weakness

Previous diagnostics suggested that immediate next-patch latent AR can exploit local continuity and center/position side channels. ViewAction-NEPA changes the prediction target from local patch continuity to sensor-driven partial-view dynamics. The model must predict what a new sensor action will reveal.

## Difference from close work

### PointGPT

PointGPT shows that point-cloud autoregressive pretraining is strong. It orders point patches and predicts/generates next patches. ViewAction-NEPA instead predicts the next partial observation under an explicit viewpoint action.

### PseudoNeg-MAE / transformation-sensitive MAE

Transformation-conditioned representation learning can look similar if we only use global rotations. ViewAction-NEPA must include partial-view transitions, next-view retrieval, goal-view planning, and rollout consistency to avoid becoming just a transformation-sensitive MAE variant.

### RI-MAE / rotation-invariant SSL

RI-MAE learns rotation-invariant latent spaces. ViewAction-NEPA learns action-conditioned latent transitions: the latent state should change predictably when the sensor moves.

### T-MAE

T-MAE uses temporal adjacent LiDAR frames with masked reconstruction. ViewAction-NEPA uses object-level multi-view partial point clouds and latent action-conditioned prediction.

### V-JEPA 2 / action-conditioned world models

V-JEPA 2 is a large-scale video/action world-model direction. ViewAction-NEPA is an object-level controlled reduction where we can cheaply test partial-view latent dynamics.
