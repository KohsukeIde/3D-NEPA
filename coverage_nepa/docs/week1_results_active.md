# Coverage-NEPA Week 1 Results

Status: completed through internal verifier.

## Current Job

```text
job_id: 1787473.pbs1
job_name: coverage_nepa_week1_greedy_200_20260521_200329
view_cache: data/viewaction_shapenet55_hpr_v12_phase15_randomized_200_seed15_current
coverage_cache: data/coverage_shapenet55_v12_greedy_fixed1024_smoke
num_shapes: 200
num_levels: 6
points_per_level: 1024
order_mode: greedy_new_coverage
```

## Implementation Hardening

The initial package needed several fixes before interpreting results:

- Preserve upstream manifest order unless explicitly shuffling before `max_shapes`.
- Respect `source_split` from the ViewAction cache when selecting train/test files.
- Avoid splitting by coverage state, because that leaks the same shape across train/test levels.
- Evaluate shortcut and frozen-axis probes with held-out shapes.
- Add input-side shortcut diagnostics, not only metadata diagnostics.
- Fix `z_shuffled` retrieval so it uses a wrong latent from another shape, never the target latent.
- Report current-select rate in coverage retrieval.
- Evaluate semigroup over all `k < l < m`, not only adjacent `k -> k+1 -> m`.
- Train separate Week 1 checkpoints for `conditioned`, `no_level`, `level_only`, and `z_shuffled`.

## Day 1 Shortcut Tests

Greedy coverage cache:

```text
cache: data/coverage_shapenet55_v12_greedy_fixed1024_smoke
output: outputs/coverage_nepa/shortcut_greedy_200_v2/shortcut.json
```

| metric | value |
|---|---:|
| chance | 0.1667 |
| fixed_point_count_unique | [1024] |
| duplicate_rate_mean | 0.00018 |
| unique_voxel_count_mean | 929.29 |
| acc_raw_union_count | 0.6667 |
| acc_fixed_input_count | 0.1667 |
| acc_input_unique_voxel_count | 0.1938 |
| acc_input_duplicate_rate | 0.1667 |
| acc_input_stats | 0.1977 |
| acc_raw_union_stats | 0.2016 |

Interpretation:

- The raw union count leaks level, but this is metadata and is not fed to the encoder.
- Fixed-N input count is exactly constant.
- Input-side density/stat shortcuts are near chance to modestly above chance, well below the 60% kill threshold.
- Day 1 shortcut check passes for the greedy cache.

Random-order control:

```text
cache: data/coverage_shapenet55_v12_random_fixed1024_smoke
output: outputs/coverage_nepa/shortcut_random_200_v2/shortcut.json
```

| metric | value |
|---|---:|
| chance | 0.1667 |
| fixed_point_count_unique | [1024] |
| duplicate_rate_mean | 0.00638 |
| unique_voxel_count_mean | 916.41 |
| acc_raw_union_count | 0.6667 |
| acc_fixed_input_count | 0.1667 |
| acc_input_unique_voxel_count | 0.2093 |
| acc_input_duplicate_rate | 0.1705 |
| acc_input_stats | 0.2209 |
| acc_raw_union_stats | 0.2016 |

Interpretation:

- Random-order fixed input also passes the point-count shortcut check.

## Day 2 Frozen Axis

Greedy coverage cache:

```text
output: outputs/coverage_nepa/frozen_axis_greedy_200_v2
```

| encoder | level_acc | category_all | category_by_level |
|---|---:|---:|---|
| raw_stats | 0.2016 | 0.5465 | 0.488, 0.535, 0.488, 0.512, 0.465, 0.488 |
| simple_random | 0.1899 | 0.5659 | 0.465, 0.535, 0.581, 0.581, 0.535, 0.581 |

Random-order cache:

```text
output: outputs/coverage_nepa/frozen_axis_random_200_v2
```

| encoder | level_acc | category_all | category_by_level |
|---|---:|---:|---|
| raw_stats | 0.2016 | 0.5426 | 0.442, 0.488, 0.535, 0.535, 0.535, 0.535 |
| simple_random | 0.1938 | 0.5504 | 0.535, 0.605, 0.535, 0.535, 0.558, 0.535 |

Interpretation:

- Coverage level itself is not trivially recoverable from raw/simple frozen features.
- Category accuracy is non-trivial even with raw/simple random features, so category probe alone is not a strong signal.
- Greedy order is not clearly better than random order in frozen-axis analysis. This weakens the Coverage-specific story unless training smoke shows a clear conditioned margin.

## Early Local Smoke

A tiny CPU debug run completed training and small eval plumbing:

```text
out: outputs/coverage_nepa/local_debug/conditioned
max_steps: 4
eval_max_shapes: 3
```

Retrieval on 3 test shapes:

| variant | top1 | current_select |
|---|---:|---:|
| conditioned | 0.2222 | 0.2000 |
| no_level | 0.1778 | 0.1556 |
| level_only | 0.1556 | 0.1778 |
| z_shuffled | 0.2222 | 0.2000 |
| identity | 0.1778 | 0.1556 |

Interpretation:

- This is only a plumbing check.
- The important warning is that `conditioned` equals `z_shuffled` in this tiny run, which is exactly the loophole the Week 1 ABCI run must address.

## ABCI Internal Smoke

```text
out_root: outputs/coverage_nepa/coverage_nepa_week1_greedy_200_20260521_200329
stderr: non-empty only because run_week1_smoke.sh was edited while the job was already executing
```

The job completed all four training/eval stages before the final shell message failed.

Final training logs:

| checkpoint | final loss | final cos | current_cos | z_var |
|---|---:|---:|---:|---:|
| conditioned | 0.0013 | 0.9987 | 0.9363 | 8.41e-7 |
| no_level | 0.0015 | 0.9985 | 0.9357 | 3.81e-6 |
| level_only | 0.0099 | 0.9901 | 0.9999 | 7.70e-5 |
| z_shuffled | 0.0013 | 0.9987 | 0.9328 | 7.06e-7 |

### Conditioned Checkpoint

Retrieval:

| eval variant | top1 | margin | current_select |
|---|---:|---:|---:|
| conditioned | 0.1178 | -0.00002 | 0.2155 |
| no_level | 0.1225 | -0.00004 | 0.2109 |
| level_only | 0.0884 | -0.00019 | 0.2341 |
| z_shuffled | 0.1178 | -0.00002 | 0.2109 |
| identity | 0.1504 | -0.00006 | 0.1953 |

Semigroup:

| eval variant | direct_top1 | rollout_top1 | direct_rollout_cos | direct_target_cos |
|---|---:|---:|---:|---:|
| conditioned | 0.1233 | 0.1186 | 1.0000 | 0.9989 |
| no_level | 0.1256 | 0.1221 | 0.9999 | 0.9908 |
| level_only | 0.0593 | 0.0686 | 0.9662 | 0.0627 |
| z_shuffled | 0.1233 | 0.1186 | 1.0000 | 0.9989 |
| identity | 0.1523 | 0.1523 | 1.0000 | 0.9376 |

### Separate Baseline Checkpoints

Retrieval, using each separately trained checkpoint's intended variant:

| checkpoint | intended eval | top1 | margin | current_select |
|---|---|---:|---:|---:|
| conditioned | conditioned | 0.1178 | -0.00002 | 0.2155 |
| no_level | no_level | 0.1318 | -0.00003 | 0.1953 |
| level_only | level_only | 0.1535 | -0.00016 | 0.1845 |
| z_shuffled | z_shuffled | 0.1271 | -0.00002 | 0.2062 |

Semigroup, using each separately trained checkpoint's intended variant:

| checkpoint | intended eval | direct_top1 | rollout_top1 | direct_rollout_cos |
|---|---|---:|---:|---:|
| conditioned | conditioned | 0.1233 | 0.1186 | 1.0000 |
| no_level | no_level | 0.1419 | 0.1360 | 0.9999 |
| level_only | level_only | 0.1535 | 0.1547 | 1.0000 |
| z_shuffled | z_shuffled | 0.1326 | 0.1302 | 1.0000 |

## Decision

Coverage-NEPA does not pass the Week 1 internal verifier.

Reasons:

- `conditioned` does not beat `no_level` by +10pt.
- `conditioned` does not beat `level_only` by +5pt.
- `conditioned` is exactly tied with `z_shuffled` on its own checkpoint.
- Separately trained shortcut checkpoints are equal or better than `conditioned`.
- Margins remain negative.
- Current/identity selection remains substantial.
- Semigroup cosine is near-perfect but retrieval is weak, matching the Noise-NEPA false-positive pattern.

Therefore, do not proceed to ScanObjectNN linear probe or full pretraining for this SimpleEncoder Coverage-NEPA variant.

The useful result is diagnostic:

- Fixed-N nested coverage states clear the simple point-count shortcut.
- Raw/input stats do not trivially reveal coverage level.
- But the latent transition objective still collapses into shortcut-equivalent behavior and does not demonstrate state-conditioned coverage prediction.

## Final Axis Tests

The remaining loophole was that fixed-N coverage states might still contain a supervised coverage axis even though the NEPA objective failed.

This was tested in:

```text
coverage_nepa/docs/final_axis_tests_active.md
```

Summary:

| test | key result | decision |
|---|---|---|
| supervised coverage-level ceiling | greedy PointNet mean top1 0.1791, SimpleEncoder 0.1667, chance 0.1667 | fail |
| shortcut margin | greedy PointNet is -0.0558 below best input shortcut | fail |
| category monotonicity | greedy PointNet C5-C0 accuracy delta +0.0263, sign rate 0.5702 | fail |
| greedy vs random | random control matches or beats greedy on monotonicity | fail |

Final decision:

```text
Coverage-NEPA is killed for the current AAAI main route.
```

Do not run full pretraining, ScanObjectNN, PointGPT integration, or nestedness-preserving redesign for this track without a new representation proposal and a stronger supervised ceiling first.

Current recommendation:

1. Archive Coverage-NEPA as a no-go for the current SimpleEncoder / same-shape latent retrieval setup.
2. Do not spend multi-day compute on this exact route.
3. If revisiting Coverage, require a stronger candidate pool before training: cross-shape hard negatives, current-level negative, and an objective aligned to retrieval rather than cosine-only latent matching.
4. Given repeated failures of internal predictive games to transfer, prioritize a theme whose pretext has direct downstream supervision pressure or validated feature reuse from a strong pretrained backbone.
