# PF kill tests for Denoising-Time NEPA

## KT0: initial cosine viability

Run `premortem_initial_cosine.py` with a frozen encoder.

| Observation | Interpretation | Action |
|---|---|---|
| mean cosine > 0.95 for most t/s | Identity shortcut risk | Increase noise range or use stronger corruption; do not train yet |
| mean cosine 0.70-0.90 | Workable signal | Proceed to train smoke |
| mean cosine < 0.50 | Noise destroys representation | Reduce noise range |

## KT1: noise-level identifiability

Run `premortem_noise_id.py`.

| Observation | Interpretation | Action |
|---|---|---|
| noise-bin acc > 80% | Feature encodes noise level trivially | Use stronger object negatives and no-time baseline; watch for shortcuts |
| noise-bin acc 40-80% | Useful but nontrivial | Proceed |
| noise-bin acc near chance | Encoder is noise-invariant | Time conditioning may be essential; proceed carefully |

## KT2: time-conditioned retrieval

Evaluate `eval_denoising_time_retrieval.py`.

Go:
- time-conditioned top1 > no-time by >= 10 points
- shuffled-time drops clearly
- prediction is closer to target `z_s` than current `z_t`

No-go:
- time-conditioned ≈ no-time
- shuffled-time ≈ time-conditioned
- identity/current wins

## KT3: semigroup consistency

Evaluate `eval_semigroup_consistency.py`.

Go:
- rollout `t->s->r` is closer to target `z_r` than no-time / identity
- direct `t->r` and rollout are consistent

No-go:
- rollout collapses
- random / no-time equals time-conditioned

## KT4: downstream only after KT2/KT3

Do not run expensive ScanObjectNN until internal dynamics pass.

Minimum downstream go if proceeding:
- linear probe improves over PointNEPA baseline by >= 3 pts
- full fine-tune does not collapse
- noise/corruption OOD improves by >= 5 pts
