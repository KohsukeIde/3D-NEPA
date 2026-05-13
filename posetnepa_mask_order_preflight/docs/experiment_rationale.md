# Experiment rationale: what this pre-flight is actually checking

## The earlier design problem

The old order-only pre-flight tried to measure `copy_win` under the default PointGPT mask-on setup. That is not a clean test of copy shortcut because PointGPT masks most tokens during pretraining. If masking already prevents local copying, then changing order may not affect `copy_win`, and a negative result would not invalidate geometry-induced order.

## Correct decomposition

We separate three hypotheses:

### H0: Naive mask-off 1D latent AR has a copy-like shortcut.

Test: `mask_ratio=0.0`, fixed grouping, baseline order.

Metrics:
- `copy_win`
- `gap = cos_tgt - cos_prev`
- copy baseline if implemented later
- PB-T50-RS downstream

If `copy_win` is low and `gap` is healthy, copy shortcut is not the main motivation.

### H1: Geometry-induced ordering matters independently of grouping.

Test: fixed `group_mode=fps_knn`, vary only `order_mode`.

This avoids the trivial grouping critique. We do not compare `fps_knn` against `random_group` as a primary argument.

### H2: Geometry-induced order is complementary to masking.

Test: mask sweep `{0.0, 0.3, 0.7}` across the same order modes.

Possible outcomes:
- Improves only at `mask=0.0`: order mainly addresses copy shortcut.
- Improves at both `mask=0.0` and `mask=0.7`: order addresses a deeper filtration/coherence issue.
- No improvement: PosetNEPA-lite is not supported; consider full frontier loss or retreat to pointNEPA.

## Why grouping is not the main claim

Changing `group_mode` from local kNN patches to random groups can be viewed as breaking the input tokenization, analogous to breaking ViT patch embeddings. It is useful as a sanity check but weak as a main contribution. This pack fixes grouping by default and changes order only.

## Remaining loopholes and fixes

- A stochastic `random` order can be interpreted as augmentation noise rather
  than a stable arbitrary total order. Fix: include `fixed_random`, `morton`,
  `axis_x`, `radial`, and `farthest_greedy` controls before making the order
  claim.
- Short pretrain runs can mis-rank final downstream performance. Fix: use Stage
  1 for direction only, Stage 2 for broader screening, and Stage 3 for
  confirmation before implementing PosetNEPA-Core.
- Copy diagnostics can be absent from logs if a run fails early or uses a
  non-NEPA loss. Fix: `04_extract_pretext_diag.py` emits missing rows and
  `05_summarize_pf3.py` marks the diagnostic incomplete rather than silently
  deciding.
- PB_T50_RS alone can be noisy. Fix: Stage 3 adds `objbg` and `objonly` before
  promoting the result to paper-facing evidence.
