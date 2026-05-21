# Coverage-NEPA PF kill tests

## KT0: point-count shortcut

All coverage states must be FPS/random sampled to exactly the same number of input points.

Kill if:

- the input point count differs by level;
- raw metadata accidentally enters the model input;
- raw-stat descriptors classify coverage level above 60% top-1 after fixed-N sampling.

## KT1: internal verifier

Run same-shape coverage-level retrieval.

Go if:

- coverage-conditioned top1 > no-level by >= 10pt;
- coverage-conditioned top1 > level-only by >= 5pt;
- coverage-conditioned top1 > z-shuffled by >= 5pt;
- target-current margin is positive.

No-go if:

- level-only ~= coverage-conditioned;
- z-shuffled ~= coverage-conditioned;
- identity/current dominates.

## KT2: monotonic downstream relevance

Run coverage-level category probes.

Go if:

- category linear probe improves with increasing coverage level;
- greedy-new-coverage order is better than random order;
- early ScanObjectNN linear probe is at least non-trivial.

Suggested thresholds:

- <45% ScanObjectNN linear: kill or retreat;
- 45-50%: marginal;
- 50-55%: continue cautiously;
- >=55%: strong signal.

## KT3: semigroup consistency

For k < l < m:

```text
Ck -> Cm direct
Ck -> Cl -> Cm rollout
```

Go if rollout is not random and direct/rollout predictions agree.

Kill if rollout is near random and direct-only retrieval is weak.

## KT4: novelty risk

Before full training, write a comparison table against:

- Point-PQAE;
- completion / partial-to-complete methods;
- Point-MAE / PointGPT;
- multi-view contrastive methods.

Kill if the contribution reduces to "partial-to-full completion in latent space" without a new verifier or semigroup result.
