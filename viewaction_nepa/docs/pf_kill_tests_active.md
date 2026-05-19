# ViewAction-NEPA kill tests

These tests decide whether this track becomes the AAAI main route.

## KT0: Data sanity

Before training:

- visible point count histogram has no severe collapse;
- random object/view visualizations show meaningful partial views;
- action graph is connected;
- each view has at least 3 outgoing actions;
- train/val/test category distribution is reasonable.

No-go if many views have <128 visible points before resampling.

## KT1: Does action matter?

Train:

- action-conditioned;
- no-action;
- shuffled-action.

Evaluate next-view retrieval among same-object candidate views.

Go:

- action-conditioned top-1 > no-action by >= 10 points;
- action-conditioned top-1 > shuffled-action by >= 10 points;
- MRR improves clearly.

No-go:

- action-conditioned and no-action differ by <5 points at 50 epochs.

## KT2: Is the model not collapsed?

Log:

- forward cosine mean/std;
- z variance;
- z norm online/target;
- per-action loss.

No-go if z variance collapses to near-zero or per-action losses are indistinguishable and retrieval does not improve.

## KT3: Goal-view planning

Given current view and target view, choose from outgoing actions.

Go:

- selected action reduces graph distance to target more often than no-action/random;
- top-1 action accuracy above chance by a clear margin.

## KT4: 2-step rollout

Go:

- 2-step action-conditioned retrieval > no-action by >= 10 points;
- rollout cosine does not collapse.

No-go:

- 2-step prediction degenerates to a constant or current-view embedding.

## KT5: Transfer smoke

Only after KT1-KT4 are positive:

- linear probe target: >=55% on ScanObjectNN PB-T50-RS-like probe;
- full fine-tune target: >=81%;
- partial/OOD robustness improvement: >=5 points over PointNEPA baseline.

If KT1-KT4 pass but KT5 is weak, the paper may still be framed around object-level world-model capability rather than generic SSL SOTA.
