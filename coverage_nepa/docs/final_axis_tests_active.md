# Coverage-NEPA Final Axis Tests

Status: completed.

## Purpose

Coverage-NEPA Week 1 already failed the internal verifier. These final tests check only whether the fixed-N coverage states still contain a supervised, downstream-relevant coverage axis.

No additional NEPA training or ScanObjectNN probe was run.

## Inputs

```text
greedy cache: data/coverage_shapenet55_v12_greedy_fixed1024_smoke
random cache: data/coverage_shapenet55_v12_random_fixed1024_smoke
levels: 6
points per level: 1024
source-held-out test shapes: 43
category-monotonicity seen-category test shapes: 38
excluded unseen-category test shapes: 5
```

## Supervised Coverage-Level Ceiling

Command outputs:

```text
outputs/coverage_nepa/supervised_ceiling_greedy_200/supervised_coverage_ceiling.json
outputs/coverage_nepa/supervised_ceiling_random_200/supervised_coverage_ceiling.json
```

Five seeds were used for each model.

| cache | model | mean top1 | worst top1 | best input shortcut | margin |
|---|---|---:|---:|---:|---:|
| greedy | input_stats_mlp | 0.2349 | 0.2209 | 0.2349 | 0.0000 |
| greedy | pointnet_small | 0.1791 | 0.1705 | 0.2349 | -0.0558 |
| greedy | simple_encoder | 0.1667 | 0.1512 | 0.2349 | -0.0682 |
| random | input_stats_mlp | 0.2783 | 0.2558 | 0.2783 | 0.0000 |
| random | pointnet_small | 0.1798 | 0.1589 | 0.2783 | -0.0984 |
| random | simple_encoder | 0.1705 | 0.1628 | 0.2783 | -0.1078 |

Interpretation:

- Chance is 0.1667.
- A trained PointNet-small does not recover coverage level from the fixed-N point input.
- PointNet-small is worse than the input-stat shortcut on both greedy and random caches.
- SimpleEncoder is at chance.
- This fails the axis-existence threshold: supervised level top1 is below 35%, far below 45%, and has negative margin over input-side shortcuts.

## Category Monotonicity

Command outputs:

```text
outputs/coverage_nepa/category_monotonicity_greedy_200/category_monotonicity.json
outputs/coverage_nepa/category_monotonicity_random_200/category_monotonicity.json
```

This uses one shared category probe across all levels. The headline paired metric is the true-class log-probability change from C0 to C5 on the same held-out shape.

| cache | model | acc C5-C0 | true-logp C5-C0 | positive sign rate | spearman | max adjacent drop |
|---|---|---:|---:|---:|---:|---:|
| greedy | input_stats_mlp | -0.1140 | 0.0362 | 0.5000 | -0.6964 | -0.0965 |
| greedy | pointnet_small | 0.0263 | 0.0671 | 0.5702 | 0.5340 | -0.0088 |
| greedy | simple_encoder | 0.0351 | 0.0169 | 0.5263 | 0.5578 | -0.0088 |
| random | input_stats_mlp | 0.0702 | 0.1641 | 0.6228 | 0.5021 | -0.0351 |
| random | pointnet_small | 0.0263 | 0.0515 | 0.6053 | 0.2380 | -0.0175 |
| random | simple_encoder | -0.0088 | 0.0076 | 0.6228 | -0.1746 | -0.0175 |

Interpretation:

- Greedy PointNet-small improves only +2.6pt from C0 to C5, below the +5pt no-go threshold and below the +7pt pass threshold.
- Greedy paired sign rate is 0.5702, below the required signal level.
- Greedy does not beat random-order control. Random has equal PointNet accuracy delta and better paired sign rate.
- SimpleEncoder is not stable across seeds and does not show a meaningful monotonic signal.
- Input-stat behavior is inconsistent and stronger for random than greedy, which argues against a Coverage-specific downstream axis.

## Final Decision

Coverage-NEPA is killed for the current AAAI main route.

The specific conclusion is:

```text
fixed-N coverage states do not expose a reliable supervised coverage axis,
and increased greedy coverage does not add category information in a monotonic,
greedy-specific way under these probes.
```

This closes the remaining loophole after the internal verifier failure. Do not proceed to full pretraining, ScanObjectNN, PointGPT integration, or nestedness-preserving redesign for Coverage-NEPA unless a substantially different coverage representation is proposed with a stronger supervised ceiling before implementation.
