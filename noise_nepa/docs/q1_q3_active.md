# Q1-Q3

## Q1. Research theme

**Denoising-Time NEPA for 3D Point Clouds.** We treat diffusion time as a natural predictive axis for point clouds and train a latent predictor to map noisy representations to less-noisy representations, without reconstructing points.

## Q2. What is new?

1. **Latent denoising dynamics instead of point reconstruction.** Existing diffusion pretraining reconstructs points or masked points; we predict target latent states at lower noise levels.
2. **Semigroup consistency.** We explicitly test whether multi-step latent denoising composes: `t->s->r` should agree with `t->r`.
3. **Pre-mortem before training.** We first test whether the frozen latent path over noise time is too trivial, too broken, or learnable.

## Q3. Closest prior work

1. **PointDif**: diffusion-based point cloud pretraining with point reconstruction / conditional generator.
2. **Point-MaDi / DiffPMAE**: masked autoencoding plus diffusion reconstruction.
3. **PointGPT / PointNEPA**: point-cloud autoregressive latent/generative pretraining; our predictive axis is diffusion time, not spatial token order.
