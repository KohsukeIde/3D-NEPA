# Denoising-Time NEPA research framing

## One-sentence thesis

Denoising-Time NEPA replaces arbitrary spatial token order with diffusion time: instead of reconstructing noisy point clouds, it learns latent denoising dynamics that predict less-noisy representations and satisfy multi-step semigroup consistency.

## Why this is not ordinary diffusion pretraining

PointDif and Point-MaDi use diffusion to reconstruct point clouds or masked point clouds. Denoising-Time NEPA does not train a point generator, Chamfer decoder, or masked reconstruction head. The target is not clean geometry but the latent state at a lower noise level.

## Core equation

For clean point cloud `x0`, sample two noise levels `t > s`:

```text
x_t = sqrt(alpha_t) x0 + sqrt(1-alpha_t) eps_t
x_s = sqrt(alpha_s) x0 + sqrt(1-alpha_s) eps_s
z_t = E_online(x_t)
z_s = E_target(x_s)
z_hat_s = F(z_t, embed(t,s))
L = 1 - cos(z_hat_s, stopgrad(z_s))
```

## Semigroup consistency

Diffusion has a natural temporal composition:

```text
t -> s -> r should agree with t -> r
```

The latent predictor should therefore satisfy:

```text
F(F(z_t, t, s), s, r) ≈ F(z_t, t, r)
```

This is the key novelty beyond a decoder-free diffusion-JEPA variant.

## Current positioning

This is a smoke track. It should not yet claim SOTA or replacement of PointDif / Point-MaDi. It first tests whether diffusion-time latent dynamics is a viable predictive signal.
