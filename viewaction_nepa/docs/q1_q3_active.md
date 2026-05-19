# Q1-Q3 for ViewAction-NEPA

## Q1. Research theme

**ViewAction-NEPA: Action-conditioned latent prediction from partial 3D object views.**

Given a partial point cloud observed from one viewpoint and an explicit camera action, the model predicts the latent embedding of the next partial view. The goal is to train an object-level 3D representation that supports next-view retrieval, multi-step rollout, and goal-view planning.

## Q2. What is new?

1. **Action-conditioned latent dynamics for object-level point clouds.**
   Existing object-centric 3D SSL mainly reconstructs masked geometry or predicts ordered patches. We predict the next partial observation under a known sensor action.

2. **World-model-like evaluation.**
   We evaluate next-view retrieval, shuffled-action controls, goal-view planning, and multi-step rollout. These are not captured by ScanObjectNN alone.

3. **Partial-observation setting.**
   Global rotations alone are too close to augmentation. We use partial views generated from camera trajectories, so action changes what is visible.

## Q3. Closest prior work

1. **PointGPT** — point-cloud autoregressive pretraining over ordered patches. Our transition is action-driven between partial views, not next-patch generation.

2. **T-MAE / temporal LiDAR MAE** — uses adjacent LiDAR frames, but with masked reconstruction and driving scenes. We use object-level partial views and latent action-conditioned prediction.

3. **V-JEPA 2 / action-conditioned world models** — large-scale video/action world-model direction. Our contribution is a controlled object-level 3D version.

Additional related work: PseudoNeg-MAE, RI-MAE, Point-MAE, PCP-MAE, transformation prediction, active predictive coding.
