# PF3 — kill tests and pivot thresholds

These are not “stop research” criteria. They are framing criteria.

## Kill Test 0: mask-off naive AR copy shortcut

Condition: `mask_ratio=0.0`, `order=simplified_morton`, `group=fps_knn`.

| Observation | Interpretation | Action |
|---|---|---|
| `copy_win >= 0.60` or `gap <= 0.02` | strong copy-like shortcut | copy/shortcut motivation is viable |
| `copy_win 0.35–0.60` | partial shortcut | use copy only as supporting evidence |
| `copy_win < 0.35` and `gap > 0.05` | no strong copy shortcut | do not use copy as the main motivation |

## Kill Test 1: order effect under fixed grouping

Compare `simplified_morton`, `fixed_random`, `axis_x`, `radial`, `bfs_shell`, `geodesic_shell`, `diffusion_shell` with `group=fps_knn`.

| Observation | Interpretation | Action |
|---|---|---|
| geometry order improves PB-T50-RS by `>= +1.0%` | meaningful order effect | commit to PosetNEPA-lite/core |
| accuracy within `±0.5%`, but `copy_win` visibly decreases | partial support | use shortcut/coherence framing; accuracy headline weak |
| geometry order worse by `>= -1.0%` | strong counterevidence | do not use that order as default |
| all orders indistinguishable | weak order story | consider full frontier loss or pointNEPA paper |

## Kill Test 2: masking/order interaction

Compare the best geometry order against baseline order for `mask_ratio = 0.0, 0.3, 0.7`.

| Observation | Interpretation | Action |
|---|---|---|
| geometry order helps at `mask=0.0` only | order mainly addresses naive shortcut | present as alternate/diagnostic, not enough alone |
| geometry order helps at `mask=0.7` too | order is complementary to masking | strong PosetNEPA argument |
| mask-on baseline already wins everything | masking is sufficient in current scaffold | pivot to pointNEPA/masking analysis |

## Single-run noise floor

Use single runs. Treat ScanObjectNN gaps below ~0.5% as inconclusive. Treat gaps >= 1.0% as meaningful enough to guide framing.

## Extra safeguards

- Use `fixed_random` for the stable arbitrary-order control. A bad stochastic
  `random` row alone is not evidence that geometry ordering is better. The core
  comparison should include `simplified_morton`, `morton`, `fixed_random`,
  `axis_x`, and at least one geometry-induced shell order.
- Stage 1 can kill copy-shortcut framing, but it should not kill the entire
  PosetNEPA idea. A negative Stage 1 moves the motivation to
  filtration/coherence and requires Stage 2 or frontier-core evidence.
- Stage 1/2 accuracy is a screening signal only. Paper-facing claims require
  Stage 3 confirmation and then seed repeats/full splits.
