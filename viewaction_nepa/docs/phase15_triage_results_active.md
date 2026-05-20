# ViewAction-NEPA Phase 1.5 Triage Results

Status: active diagnostic result, 2026-05-20 JST.

## Inputs

- Cache: `data/viewaction_shapenet55_hpr_v12_phase01_20260519_200828`
- Split: `test`
- Test shapes/views/transitions: `43` / `516` / `2580`
- Checkpoints: `outputs/viewaction_smoke/viewaction_p01_hard_20260519_202419/{action,no_action,shuffled_action,action_only}/ckpt_last.pth`

Result files:

- `outputs/viewaction_smoke/phase15_triage/view_discriminability_test.json`
- `outputs/viewaction_smoke/phase15_triage/raw_geometry_oracle_test.json`
- `outputs/viewaction_smoke/phase15_triage/hard_next_view_retrieval_test_all_ckpts.json`

## D1. View Discriminability

Probe setup: train split as probe training data, test split as held-out eval.
The default run uses `256` points per partial view for the supervised PointNet
probe to keep CPU runtime low.

| feature/probe | view-id top1 | category top1 | note |
|---|---:|---:|---|
| raw geometric descriptor | `0.5446` | `0.5167` | chance view-id is `0.0833` |
| random SimplePointEncoder | `0.2771` | `0.6146` | category stronger than view |
| Phase-1 checkpoint encoder | `0.2713` | `0.5979` | no view improvement over random encoder |
| small PointNet, view-only, 20 epochs | `0.1919` | n/a | still weak and likely underfit |

Readout:

- The partial views do contain view signal: raw geometry is well above chance.
- The Phase-1 checkpoint encoder is not more view-sensitive than a random
  SimplePointEncoder.
- The current encoder/training route appears to encode category/object cues
  more readily than camera/view cues.

## D2. Raw Geometry Oracle

Descriptor distance, same 43 test shapes:

| mode | target-query oracle | current-query shortcut | random | action-prior |
|---|---:|---:|---:|---:|
| same_source_outgoing | `1.0000` | `0.2000` | `0.2000` | `1.0000` |
| include_current_negative | `1.0000` | `0.0000` | `0.1667` | `1.0000` |
| same_object_all_views | `1.0000` | `0.0000` | `0.0833` | `1.0000` |
| cross_object_same_action | `1.0000` | `0.6558` | `0.0233` | n/a |

Readout:

- Candidate target views are geometrically distinguishable.
- Same-source outgoing and same-object all-view modes expose a perfect
  graph/action prior because source view plus local action identifies the
  target on the fixed camera graph.
- Cross-object same-action is dominated by object/shape identity: using the
  current-view descriptor retrieves the same-object candidate `65.6%` of the
  time even though random is `2.3%`.

## D3. Hard Next-View Retrieval

Action checkpoint, inference variants:

| mode | action | no-action | shuffled | action-only | readout |
|---|---:|---:|---:|---:|---|
| same_object_all_views | `0.1078` | `0.0926` | `0.0934` | `0.1097` | action-only >= action |
| same_source_outgoing | `0.2295` | `0.2000` | `0.2023` | `0.2298` | only +2.95pt over no-action |
| include_current_negative | `0.1942` | `0.1686` | `0.1717` | `0.1957` | current selected `15.6%`; action-only >= action |
| cross_object_same_action | `0.5926` | `0.6225` | `0.5845` | `0.5562` | no-action > action |

Action-only trained checkpoint:

| mode | action | no-action | shuffled | action-only | readout |
|---|---:|---:|---:|---:|---|
| same_source_outgoing | `0.2453` | `0.2000` | `0.2109` | `0.2450` | stronger than action checkpoint |
| include_current_negative | `0.2074` | `0.1686` | `0.1783` | `0.2078` | stronger than action checkpoint |
| cross_object_same_action | `0.3295` | `0.1899` | `0.3279` | `0.3372` | action-only remains competitive |

Readout:

- The hard candidate pools do not revive the current ViewAction model.
- Action-conditioned prediction does not beat no-action/action-only by the
  required `>=10pt`.
- Action-only remains equal or better in same-source and include-current modes.
- Cross-object retrieval does not cleanly prove state-action composition; the
  best behavior depends on checkpoint/variant and is still compatible with
  identity and graph priors.

## Current Decision

Phase 1.5 diagnostics support the following narrower conclusion:

> The 200-shape cache contains view-discriminative geometry, but the current
> SimplePointEncoder Phase-1 checkpoint does not learn a view-sensitive,
> action-controllable latent transition.

This strengthens the Phase-1 no-go. It does not prove the full ViewAction idea
is impossible, but it makes further SimplePointEncoder scale-up unjustified.

## Next Loop

Only two follow-ups are still rational for the ViewAction route:

1. Build a per-shape randomized camera-frame cache and rerun the same D1-D3
   diagnostics to reduce global graph/action prior.
2. Try one view-sensitive backbone or objective smoke only if randomized-cache
   diagnostics show that action-only prior drops and view discrimination
   remains high.

If action-only remains equal or better after randomized camera frames,
ViewAction should be killed as an AAAI main route.

## Remaining Loopholes

- The raw target-query oracle is an upper-bound sanity check because the true
  target geometry is used as the query. Margins and current-query behavior are
  more informative than the `1.0` target accuracy.
- The small PointNet probe is not a fully tuned supervised model; it is a cheap
  diagnostic. A stronger view classifier could still improve.
- Current `cross_object_same_action` uses fixed graph semantics. It is useful
  for exposing identity shortcuts, but randomized per-shape graph orientation
  is still required before making a final action-route kill decision.
- Phase 1.5 does not include PointGPT/Point-MAE features yet. That is a
  deliberate gate: the current SimplePointEncoder route must not be scaled, but
  one stronger-backbone retry can still be justified after randomized-cache
  controls.
