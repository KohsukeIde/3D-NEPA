# Coverage-NEPA week-1 plan

## Day 1: cache and shortcut tests

- Build fixed-N nested coverage cache from existing `viewaction_nepa` cache.
- Run point-count and raw-stat shortcut tests.
- Kill immediately if raw statistics solve coverage level too well.

## Day 2: frozen axis analysis

- Evaluate coverage-level retrieval and category probes with raw features / SimpleEncoder / optional checkpoint.
- Check whether category signal increases with coverage.

## Day 3: Coverage-NEPA smoke training

- Train a small SimpleEncoder + EMA target + level-conditioned predictor.
- Train no-level and level-only variants if time permits.

## Day 4: internal verifier + early downstream

- Same-shape coverage retrieval.
- z-shuffled and level-only controls.
- Early ScanObjectNN linear probe.

## Day 5-6: semigroup

- Direct vs rollout consistency for Ck -> Cl -> Cm.

## Day 7: decision memo

Outcomes:

1. clear Coverage-NEPA go;
2. marginal but promising;
3. kill and switch to second candidate;
4. all weak, stop AAAI rush.
