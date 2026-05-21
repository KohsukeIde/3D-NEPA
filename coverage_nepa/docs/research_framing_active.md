# Coverage-NEPA research framing

## Thesis

NEPA-style objectives are convincing when their predictive axis defines ordered intermediate states that are:

1. externally verifiable;
2. monotonic in downstream-relevant information;
3. not solvable by trivial raw statistics;
4. compatible with multi-step latent consistency.

Coverage states satisfy this more naturally than patch order, diffusion time, or camera action in the previous failed tracks:

- moving forward in coverage adds visible surface information;
- coverage levels are externally verifiable;
- nested states admit a semigroup-like transition structure;
- no point reconstruction is required.

## What this is not

This is not partial-to-full point reconstruction, not two-view cross-reconstruction, and not a multi-view contrastive method. The novelty claim is only valid if:

- all coverage states are fixed-point-count to remove point-count shortcuts;
- coverage-conditioned latent prediction beats no-level / level-only / z-shuffled controls;
- early downstream probes show that coverage latents contain object-relevant information.

## Relation to crowded areas

- Point-PQAE / multi-view reconstruction: reconstruct or cross-reconstruct views. Coverage-NEPA predicts latent states over nested coverage without a point decoder.
- Completion methods: reconstruct full geometry. Coverage-NEPA tests latent transition and semigroup consistency.
- Point-MAE / PointGPT: masked reconstruction or ordered AR. Coverage-NEPA uses nested partial-observation filtration.

## Current status

This is a candidate track only. Full training should not be attempted before Week-1 pre-mortem passes.
