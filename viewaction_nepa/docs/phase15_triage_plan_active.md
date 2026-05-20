# ViewAction-NEPA Phase 1.5 Triage Plan

Status: active diagnostic plan, 2026-05-20 JST.

## Decision From Phase 1

Phase 1 is negative for the current implementation:

- action-conditioned retrieval only beats no-action by about `0.5` to `1.7`
  top1 points.
- action-only is equal to or better than action-conditioned.
- goal-view planning is indistinguishable from random.
- plain forward training shows collapse/current-view shortcut pressure.

This kills the current `SimplePointEncoder + cosine latent transition +
same-object retrieval` implementation. It does not yet kill the whole
action-conditioned route.

## Purpose

Phase 1.5 is not a scale-up. It is a short triage loop to determine whether
the failure is caused by:

- weak or non-discriminative partial-view data,
- an evaluation that does not force state-action composition,
- a weak encoder,
- an objective mismatch,
- or an action/camera graph prior that action-only can exploit.

The first pass should run on the existing 200-shape cache and avoid new large
training jobs.

## Diagnostics

### D1. View Discriminability

Question: can a partial view `X_t` identify its camera/view id at all?

Minimum probes:

- raw geometry descriptor: centroid, covariance, bbox, scale statistics,
- frozen random `SimplePointEncoder`,
- trained Phase-1 checkpoint encoder when available,
- small supervised PointNet classifier.

Interpretation:

| result | interpretation |
|---|---|
| raw and PointNet both low | cache/view-generation problem |
| PointNet high but SimpleEncoder low | encoder problem |
| checkpoint encoder high but transition low | objective/eval problem |
| all low | ViewAction and simple multi-view consistency are both risky |

### D2. Hard Next-View Retrieval

Question: does the learned transition use both `z_t` and `a_t` under candidate
pools where action-only should fail?

Candidate modes:

- `same_object_all_views`: original all-view same-object retrieval.
- `same_source_outgoing`: candidates are outgoing target views from the current
  source view only.
- `include_current_negative`: outgoing candidates plus current view as a hard
  negative.
- `cross_object_same_action`: positive is the same-object action target,
  negatives are other-object target views for the same source/action.

Interpretation:

| result | interpretation |
|---|---|
| action > no-action/action-only by `>=10pt` | action route can be revived |
| action-only remains equal or better | current action route should be killed |
| current view selected often | transition is not moving away from `X_t` |
| cross-object collapses | state-action composition is not learned |

### D3. Raw Geometry Oracle

Question: are the candidate pools geometrically distinguishable before blaming
the model?

Baselines:

- raw descriptor target-query oracle,
- optional Chamfer target-query oracle,
- current-view query shortcut,
- random/action-prior baseline.

Interpretation:

| result | interpretation |
|---|---|
| target oracle high, model low | model/objective problem |
| target oracle low | candidate/task/view-generation problem |
| current-query high | current-view shortcut is structurally easy |
| action-prior high | graph prior is too strong |

## Go / No-Go

Go conditions for a single PointGPT/Point-MAE retry:

- raw target-query oracle on `same_source_outgoing` is comfortably above
  chance, ideally `>50%`,
- small PointNet view-id classifier is high, ideally `>70%`,
- hard retrieval shows action-conditioned `>=10pt` top1 above no-action and
  action-only in at least one meaningful hard pool,
- include-current-negative does not mostly select current view.

No-Go conditions:

- raw geometry oracle fails on hard candidate pools,
- PointNet view-id classifier fails,
- action-only remains equal or better after hard candidate pools,
- goal/planning remains random after any revived hard-retrieval signal.

## Strategy Confidence

We should not claim 100% confidence in a research route before these diagnostics
run. The factual confidence target for Phase 1.5 is narrower:

- confidence that the implemented diagnostics match the failure modes above,
- confidence that the resulting decision tree is applied consistently,
- confidence that we stop scaling the current Phase-1 implementation unless a
  real action margin appears.
