# ViewAction-NEPA Phase 1.5 Randomized-Cache Results

Status: active diagnostic result, 2026-05-20 JST.

## Inputs

- Cache: `data/viewaction_shapenet55_hpr_v12_phase15_randomized_200_seed15_current`
- Cache build: per-shape random SO(3) camera frame, `random_seed=15`
- Cache size: `200` shapes, `157` train, `43` test
- Visibility: min `238`, mean `2283.47`, max `4576`
- Randomized training run: `outputs/viewaction_smoke/viewaction_p15_randomized_hard_20260520_153916`
- Training setting: same hard-negative smoke as Phase 1, `max_steps=500`,
  `variance_weight=0.1`, `hard_contrast_weight=1.0`

Result files:

- `outputs/viewaction_smoke/phase15_triage_randomized_200_seed15_current/view_discriminability_test.json`
- `outputs/viewaction_smoke/phase15_triage_randomized_200_seed15_current/raw_geometry_oracle_test.json`
- `outputs/viewaction_smoke/phase15_triage_randomized_200_seed15_current/hard_next_view_retrieval_test_all_ckpts_trained_randomized.json`

## Cache Sanity

The randomized cache stores per-shape camera metadata:

- `camera_rotation`: `[3, 3]`, determinant `1.0`
- `camera_pos`: `[12, 3]`
- `camera_frame`: `[12, 3, 3]`
- `camera_frame_seed`

The global `view_graph.npz` is still used for shape-local topology and local
action classes. Absolute camera pose is intentionally per-shape.

## D1. View Discriminability

Global view-id labels are no longer shared semantic labels under per-shape
camera randomization. Therefore low global view-id probe accuracy is expected
and should not be read as a cache failure.

| feature/probe | global view-id top1 | category top1 | readout |
|---|---:|---:|---|
| raw geometric descriptor | `0.0911` | `0.5417` | view-id near chance `0.0833` |
| random SimplePointEncoder | `0.0891` | `0.6000` | view-id near chance |
| fixed-cache Phase-1 ckpt encoder | `0.0969` | `0.5729` | OOD probe, view-id near chance |
| randomized-trained ckpt encoder | `0.0891` | `0.5729` | no global view-id signal |

Readout:

- Randomization successfully removes global view-index semantics.
- Category/object signal remains.
- D1 should now be used mainly as a sanity check that global view-id prior is
  gone, not as a go threshold.

## D2. Raw Geometry Oracle

Descriptor distance on randomized test split:

| mode | target-query oracle | current-query shortcut | random | action-prior |
|---|---:|---:|---:|---:|
| same_source_outgoing | `1.0000` | `0.2000` | `0.2000` | `1.0000` |
| include_current_negative | `1.0000` | `0.0000` | `0.1667` | `1.0000` |
| same_object_all_views | `1.0000` | `0.0000` | `0.0833` | `1.0000` |
| cross_object_same_action | `1.0000` | `0.7291` | `0.0233` | n/a |

Readout:

- Candidate target views remain geometrically distinguishable.
- Shape-local source/action still deterministically identifies the local target
  in same-source modes; this is expected and cannot be removed without changing
  the action-task definition.
- Cross-object remains heavily identity-driven: current-view raw geometry finds
  same-object target `72.9%` of the time.

## D3. Hard Retrieval After Randomized-Cache Training

Action checkpoint:

| mode | action | no-action | shuffled | action-only | readout |
|---|---:|---:|---:|---:|---|
| same_object_all_views | `0.0915` | `0.0915` | `0.0922` | `0.0919` | no action margin |
| same_source_outgoing | `0.1981` | `0.2000` | `0.2004` | `0.1981` | action below no-action/chance |
| include_current_negative | `0.1686` | `0.1674` | `0.1694` | `0.1686` | no action margin |
| cross_object_same_action | `0.5950` | `0.6260` | `0.5950` | `0.5926` | no-action wins |

Action-only trained checkpoint:

| mode | action | no-action | shuffled | action-only | readout |
|---|---:|---:|---:|---:|---|
| same_source_outgoing | `0.2023` | `0.2000` | `0.1977` | `0.2023` | action-only >= action checkpoint |
| include_current_negative | `0.1725` | `0.1709` | `0.1698` | `0.1721` | action-only competitive |
| cross_object_same_action | `0.3209` | `0.1938` | `0.3198` | `0.3287` | action-only >= action |

Standard action-checkpoint metrics:

| metric | action | no-action | shuffled | action-only | random |
|---|---:|---:|---:|---:|---:|
| same-object next-view top1 | `0.0915` | `0.0915` | `0.0922` | `0.0919` | n/a |
| goal best-step acc | `0.3645` | `0.3619` | `0.3649` | `0.3642` | `0.3675` |
| rollout2 top1 | `0.0713` | `0.0736` | `0.0713` | `0.0678` | n/a |

Readout:

- Per-shape randomized camera frame removes the earlier global view/action
  prior, but it does not create a positive action-conditioned result.
- Action-conditioned is not better than no-action/action-only by the required
  `>=10pt`; it is often equal or worse.
- Goal planning remains below random.
- Two-step rollout remains worse than no-action.

## Decision

This closes the main Phase 1.5 loophole:

> After per-shape randomized camera frames and retraining on the randomized
> cache, the current ViewAction-NEPA route still shows no action-controllable
> latent transition.

Therefore ViewAction-NEPA should be removed from the AAAI main route in its
current form. A stronger backbone retry is not justified unless the task is
redefined beyond the current same-object/action-graph setup.

## Remaining Loopholes

- The current cross-object candidate pool is an identity stress test, not a
  clean action-composition benchmark.
- Same-source modes still have deterministic shape-local topology, so a future
  action task would need stochastic/held-out graph topology or non-local goals.
- A PointGPT/Point-MAE backbone might improve view-sensitive representation,
  but Phase 1.5 shows the current objective/evaluation does not produce the
  required action margin even after the major camera-prior fix.
