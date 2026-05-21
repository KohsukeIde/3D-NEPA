# Coverage-NEPA Q1-Q3

## Q1. Research theme

**Coverage-NEPA: decoder-free latent prediction over nested partial-observation states for 3D point clouds.**

We build fixed-size nested coverage states from multiple partial views of the same object and train a latent predictor to map a lower-coverage state to a higher-coverage state.

## Q2. What is new?

1. **Coverage as a predictive axis.** Unlike arbitrary patch order or noise time, coverage is externally verifiable and tends to add downstream-relevant surface information.
2. **Decoder-free latent transition.** We do not reconstruct points; we predict the latent state of a more complete observation.
3. **Semigroup consistency.** Because coverage states are nested, Ck -> Cl -> Cm should be consistent with Ck -> Cm.

## Q3. Closest prior work

1. **Point-PQAE / multi-view cross-reconstruction.** Closest crowded area. Difference: latent coverage transition, no point reconstruction.
2. **Partial-to-complete completion.** Difference: no decoder and no full-shape reconstruction objective.
3. **I-JEPA / point2vec / latent predictive SSL.** Difference: object-level nested coverage states as the predictive axis.
