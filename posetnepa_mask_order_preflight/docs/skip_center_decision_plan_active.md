# Skip-k / Center Leakage Decision Plan And Results

Status: active decision record for the post-PosetNEPA-Lite diagnostic chain.

Date: 2026-05-19 JST.

## Current Decision

PosetNEPA-Lite is no longer the main route. Diffusion-shell collapsed into a
total order changed shortcut diagnostics but did not improve downstream
classification or frozen readouts.

This does not kill PosetNEPA-Core. It only says the cheap total-order proxy is
not sufficient.

The next chain was:

1. skip-k NEPA;
2. center / position side-channel controls;
3. full fine-tune + linear probe + copy diagnostics;
4. choose Plan A/B/C before implementing frontier-level Core.

The chain completed as `skipcenter_20260517_213653` on 2026-05-18 17:59 JST.
Generated outputs live under:

```text
posetnepa_mask_order_preflight/generated/skipcenter_20260517_213653/
```

That generated directory is intentionally gitignored; this document promotes
the selected results into tracked documentation.

## Why This Chain

The strongest current observation is the mismatch between readouts:

- full fine-tune: `simplified_morton_m0p0` reaches about `82%` PB-T50-RS;
- frozen-head / true linear probe: only about `50-52%`.

So full fine-tune is not clean evidence that the frozen representation is
linearly separable. It may measure optimization utility, initialization quality,
or the ability to adapt under full supervision.

The next experiments separate these claims:

| Claim | Metric |
|---|---|
| benchmark competitiveness | full fine-tune |
| representation separability | linear / frozen probe |
| optimization prior | scratch / early fine-tune curves |
| local shortcut | skip-k, copy/gap |
| position leakage | zero/shuffled position controls |

## Implemented Variants

Default `20_run_skip_center_chain.sh` variants:

| variant | skip_k | pretrain position | center aux | mask | purpose |
|---|---:|---|---:|---:|---|
| `skip2` | 2 | normal | 0.0 | 0.0 | weakly block immediate next-token locality |
| `skip4` | 4 | normal | 0.0 | 0.0 | medium local-continuity block |
| `skip8` | 8 | normal | 0.0 | 0.0 | stronger long-range AR diagnostic |
| `poszero` | 1 | zero | 0.0 | 0.0 | remove pretrain absolute/relative position side-channel |
| `posshuffle` | 1 | shuffle | 0.0 | 0.0 | corrupt position-token alignment during pretrain |
| `centeraux` | 1 | normal | 0.1 | 0.0 | PCP-MAE-style center-awareness auxiliary |
| `skip4_poszero` | 4 | zero | 0.0 | 0.0 | combine local-continuity block and position hiding |

All variants use `order=simplified_morton` and `group_mode=fps_knn` by default.

## Completed Results

All rows use ScanObjectNN PB-T50-RS / hardest, `mask=0.0`, `group_mode=fps_knn`,
and 30-epoch ShapeNet pretrain followed by 50-epoch fine-tune unless noted.

| variant | skip_k | pretrain position | center aux | full FT PB-T50-RS | linear probe best | copy_win | gap |
|---|---:|---|---:|---:|---:|---:|---:|
| `skip2` | 2 | normal | 0.0 | 81.0548 | 52.9146 | 0.4829 | 0.0074 |
| `skip4` | 4 | normal | 0.0 | 80.1527 | 51.5961 | 0.4819 | 0.0089 |
| `skip8` | 8 | normal | 0.0 | 78.9035 | 49.2019 | 0.4970 | 0.0048 |
| `poszero` | 1 | zero | 0.0 | 76.5094 | 45.9056 | 0.6224 | -0.0090 |
| `posshuffle` | 1 | shuffle | 0.0 | 80.8813 | 51.4920 | 0.6374 | -0.0213 |
| `centeraux` | 1 | normal | 0.1 | 82.0264 | 53.5045 | 0.5395 | -0.0020 |
| `skip4_poszero` | 4 | zero | 0.0 | 77.3074 | 48.6468 | 0.4899 | 0.0034 |

Reference Stage 1 rows for comparison:

| reference | full FT PB-T50-RS | linear probe best | copy_win | gap |
|---|---:|---:|---:|---:|
| `simplified_morton_m0p0` | 82.3040 | 51.8390 | 0.5440 | -0.0033 |
| `diffusion_shell_m0p0` | 79.4587 | 45.4892 | 0.4466 | 0.0096 |
| `fixed_random_m0p0` | 79.2852 | 47.7099 | 0.5098 | -0.0026 |

## Learning Curve Notes

The pretrain curves did converge in the narrow sense that all rows show a large
loss drop and the final loss is close to the last-5-epoch mean:

| variant | first loss | last loss | min loss | last5 mean |
|---|---:|---:|---:|---:|
| `skip2` | 740.7787 | 140.6509 | 140.6509 | 141.5789 |
| `skip4` | 740.5824 | 190.4615 | 190.4615 | 191.4138 |
| `skip8` | 740.6043 | 194.2003 | 191.7368 | 194.8667 |
| `poszero` | 669.3646 | 56.7727 | 56.7727 | 57.0447 |
| `posshuffle` | 739.6170 | 109.6031 | 109.6031 | 110.3646 |
| `centeraux` | 768.6165 | 101.4085 | 101.4085 | 102.1596 |
| `skip4_poszero` | 660.5967 | 92.0307 | 92.0307 | 92.4333 |

Do not use raw pretrain loss as the main cross-variant ranking. The targets and
side information differ across skip-k, position-zero, position-shuffle, and
center-aux rows. Use the curves mainly to rule out failed optimization, then
interpret copy/gap, linear probe, and full fine-tune together.

Fine-tune best epochs are mostly near the end, but several rows still require
care when read as final accuracy:

| variant | best PB-T50-RS | best epoch | last PB-T50-RS | curve status |
|---|---:|---:|---:|---|
| `skip2` | 81.0548 | 46 | 80.5343 | plateau |
| `skip4` | 80.1527 | 45 | 79.4240 | plateau |
| `skip8` | 78.9035 | 50 | 78.9035 | late/still improving |
| `poszero` | 76.5094 | 46 | 75.8848 | plateau |
| `posshuffle` | 80.8813 | 50 | 80.8813 | late/still improving |
| `centeraux` | 82.0264 | 49 | 81.0201 | peaked/unstable |
| `skip4_poszero` | 77.3074 | 46 | 76.5441 | plateau |

The `centeraux` row is the numeric best, but the best-last gap is `1.0063pt`;
read it as a useful control rather than a stable headline until seed repeats or
a longer curve confirm it.

## Result Reading

The immediate previous-token-only explanation is too weak. `skip2` still reaches
`81.0548` PB-T50-RS and slightly improves the frozen linear probe over the
Stage 1 `simplified_morton_m0p0` row (`52.9146` vs `51.8390`). However, larger
skip distances degrade both full fine-tune and linear probe:

```text
skip2 -> skip4 -> skip8
PB-T50-RS:      81.0548 -> 80.1527 -> 78.9035
linear probe:   52.9146 -> 51.5961 -> 49.2019
```

So the model is not merely copying `z_{t-1}`, but it does depend strongly on a
local-continuity neighborhood.

Position and center side-channels are important. Removing pretrain positional
signal hurts sharply:

```text
poszero:       76.5094 PB, 45.9056 linear probe
skip4_poszero: 77.3074 PB, 48.6468 linear probe
```

Shuffling position is less destructive than zeroing it in full fine-tune
(`80.8813` PB), but the copy diagnostic becomes worse (`copy_win=0.6374`,
`gap=-0.0213`). That suggests full fine-tune can recover from corrupted
pretrain position while the pretext objective itself remains shortcut-prone.

The `centeraux` row is the strongest numeric row in this chain:

```text
centeraux: 82.0264 PB, 53.5045 linear probe
```

But it also keeps the shortcut warning alive: `copy_win=0.5395`, `gap=-0.0020`,
and the previous-token proxy remains competitive. It should be treated as a
useful control, not as a clean paper headline.

## Plan A: Local Continuity Shortcut

Trigger:

- skip-k full fine-tune drops at least `1.5pt` from `simplified_morton_m0p0`;
- linear probe does not improve;
- copy/gap improves while downstream worsens.

Interpretation:

The baseline depends strongly on immediate local continuity rather than broader
3D structure.

Paper route:

`Local Continuity Shortcuts in 3D Autoregressive Self-Supervised Learning`.

This is more analysis-heavy and would need broader baselines before becoming an
AAAI main route.

Status after `skipcenter_20260517_213653`: partially supported. Skip-k hurts
as distance increases, but `skip2` remains strong enough that the story is not
"immediate copy only."

## Plan B: Long-Range Latent AR

Trigger:

- some `k in {2,4,8}` improves linear probe by at least `+3pt`;
- full fine-tune is within `-0.5pt` of baseline or improves;
- copy/gap improves.

Interpretation:

Immediate next-patch prediction is too local; longer-range latent AR yields a
more transferable representation.

Paper route:

`Long-Range Latent Autoregression for 3D Point Clouds`.

This is the cleanest method route if the numbers appear.

Status after `skipcenter_20260517_213653`: not supported. No skip-k row gives a
large linear-probe gain, and larger skips degrade both full fine-tune and frozen
readout.

## Plan C: PosetNEPA-Core Revival

Trigger:

- skip-k exposes local-continuity dependence;
- center/position controls show side-channel dependence;
- no simple skip-k variant improves frozen/linear representation.

Interpretation:

Token-level next prediction is the limitation. The method needs frontier-set
prediction rather than another total-order target.

Core smoke go condition:

- linear probe improves over `simplified_morton_m0p0` by at least `+2pt`, or
- full fine-tune is within `-0.5pt` and structured robustness improves by
  `+5pt`, and
- geometry frontier beats random frontier, with no aggregation collapse.

Status after `skipcenter_20260517_213653`: this is now the preferred method
route if we continue this paper. The evidence points to a limitation of
token-level total-order next prediction rather than a simple skip-k fix:

- diffusion-shell total-order Lite reduced copy pressure but did not improve
  downstream or frozen readout;
- skip-k shows local-neighborhood dependence but does not produce a better
  representation on its own;
- position/center controls show strong side-channel dependence;
- full fine-tune can obscure representational weakness.

Therefore, the next method experiment should not be another total-order variant.
It should test frontier-set prediction with explicit position/center controls,
and compare geometry-induced frontiers against random frontiers.

## Active Decision

Do not frame the current results as a clean PointNEPA accuracy paper. The
current scaffold is best used as diagnostic evidence:

> PointGPT-style latent AR works in part because local geometric continuity and
> position/center side-channels make the next-patch objective tractable.
> Total-order alternatives can change shortcut diagnostics, but they do not
> yet produce a stronger frozen representation. A meaningful 3D AR method must
> define the context/future structure as a geometry-induced frontier/set,
> not merely as another 1D order.

The recommended next implementation is **PosetNEPA-Core smoke**, not another
PosetNEPA-Lite sweep.

## Decision Date

Paper direction should be fixed by 2026-06-30. After that, the project should
either execute Plan B/C or pivot away from this paper route.
