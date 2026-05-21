# Coverage-NEPA: one-week pre-mortem track

This track is a **stage-gated smoke implementation**, not a finished paper implementation.

It tests whether a NEPA-style objective becomes convincing when the predictive axis is an externally verifiable, downstream-relevant filtration: nested coverage states of the same 3D object.

Core idea:

```text
C1 ⊂ C2 ⊂ ... ⊂ CK
z(Ck), (k,m) -> z(Cm), k < m
```

where `Ck` is a fixed-point-count partial observation created by unioning multiple partial views. The key distinction from completion or multi-view reconstruction is that this track first tests whether **coverage is a valid latent predictive axis** before doing any large-scale training.

## Why this track exists

Previous attempts exposed a common failure mode: internal predictive games can be solved while providing no useful object representation. This track therefore requires both:

1. an internal verifier: coverage-level latent prediction must beat no-level / level-only / shuffled controls;
2. an early downstream probe: ScanObjectNN linear probe must be non-trivial very early.

## Week-1 decision logic

Run:

1. Build fixed-N nested coverage cache from `viewaction_nepa` multi-view cache.
2. Point-count / raw-stat shortcut tests.
3. Frozen feature coverage-axis analysis.
4. Small Coverage-NEPA smoke training.
5. Coverage retrieval and semigroup evaluation.
6. Early ScanObjectNN linear probe.

Do **not** run full pretraining until these pass.

## Minimal commands

From repo root:

```bash
# Build nested coverage cache from existing viewaction multi-view cache.
VIEW_CACHE=data/viewaction_shapenet55_hpr_v12_phase15_randomized_200_seed15_current \
OUT_CACHE=data/coverage_shapenet55_v12_greedy_fixed1024_smoke \
bash coverage_nepa/scripts/00_build_coverage_cache.sh

# Shortcut tests.
CACHE_ROOT=data/coverage_shapenet55_v12_greedy_fixed1024_smoke \
bash coverage_nepa/scripts/01_eval_shortcuts.sh

# Train a small coverage NEPA smoke model.
CACHE_ROOT=data/coverage_shapenet55_v12_greedy_fixed1024_smoke \
OUT=outputs/coverage_nepa/smoke_conditioned \
bash coverage_nepa/scripts/03_train_coverage_smoke.sh

# Internal verifier.
CACHE_ROOT=data/coverage_shapenet55_v12_greedy_fixed1024_smoke \
CKPT=outputs/coverage_nepa/smoke_conditioned/ckpt_last.pth \
bash coverage_nepa/scripts/04_eval_coverage_retrieval.sh

# Semigroup check.
CACHE_ROOT=data/coverage_shapenet55_v12_greedy_fixed1024_smoke \
CKPT=outputs/coverage_nepa/smoke_conditioned/ckpt_last.pth \
bash coverage_nepa/scripts/05_eval_semigroup.sh
```

## Critical kill tests

Kill immediately if:

- fixed-N coverage still leaks level through point count or raw stats;
- raw-stat descriptors solve coverage level too well;
- coverage-conditioned retrieval does not beat no-level / level-only / z-shuffled controls;
- coverage semigroup is random;
- early ScanObjectNN linear probe is below 45%.
