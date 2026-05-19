# ViewAction-NEPA Phase 0/1 ABCI Results

Status: active smoke record, 2026-05-19 JST.

## Data And Runtime

- Data: `/groups/gag51402/datasets/ShapeNet55-34/shapenet_pc`
- Split files: `/groups/gag51402/datasets/ShapeNet55-34/ShapeNet-55/{train,test}.txt`
- Cache: `data/viewaction_shapenet55_hpr_v12_phase01_20260519_200828`
- Cache size: 200 shapes, 157 train shapes, 43 test shapes
- Visibility summary: min `207`, mean `2228.04`, max `4542`
- ABCI queue: `rt_HG`, 1 GPU

## Leakage Fixes Before Interpreting

- Changed `action_id` from globally unique edge id to shared local move id
  `0..4`.
- Removed absolute source/target camera directions from `action_vec`; it now
  uses relative camera motion only.
- Added manifest-based ShapeNet55 split handling with source split metadata.
- Added no-action, shuffled-action, action-only, random, and oracle controls.
- Added current-view similarity and prediction variance diagnostics.

## Runs

### 1. Plain EMA Forward Smoke

Run: `outputs/viewaction_smoke/phase01_20260519_200828`

Next-view retrieval on action checkpoint:

| variant | top1 | top5 | MRR |
|---|---:|---:|---:|
| action | 0.1128 | 0.5163 | 0.3060 |
| no_action | 0.0957 | 0.4558 | 0.2793 |
| shuffled_action | 0.0826 | 0.4302 | 0.2631 |
| action_only | 0.1140 | 0.5147 | 0.3056 |

Readout: fails the go condition. Action improves over no-action by only
`+1.7pt` top1, and action-only on the action checkpoint is essentially equal
to action. Training diagnostics show collapse/current-view shortcut pressure:
`forward_cos ~= 0.9995`, `current_cos ~= 0.9997`, and low prediction variance.

### 2. Batch Contrast + Variance Anti-Collapse

Run: `outputs/viewaction_smoke/viewaction_p01_contrast_20260519_201817`

Next-view retrieval on action checkpoint:

| variant | top1 | top5 | MRR |
|---|---:|---:|---:|
| action | 0.0988 | 0.4659 | 0.2827 |
| no_action | 0.0942 | 0.4570 | 0.2770 |
| shuffled_action | 0.0942 | 0.4547 | 0.2768 |
| action_only | 0.1023 | 0.4678 | 0.2858 |

Readout: collapse diagnostics improve, but action use does not. The action
margin is only `+0.5pt` top1 over no-action.

### 3. Same-Source Hard-Negative Contrast

Run: `outputs/viewaction_smoke/viewaction_p01_hard_20260519_202419`

Next-view retrieval on action checkpoint:

| variant | top1 | top5 | MRR |
|---|---:|---:|---:|
| action | 0.1078 | 0.4973 | 0.2988 |
| no_action | 0.0926 | 0.4578 | 0.2762 |
| shuffled_action | 0.0934 | 0.4593 | 0.2767 |
| action_only | 0.1097 | 0.5023 | 0.3007 |

Separately trained baselines:

| trained baseline | top1 | top5 | MRR |
|---|---:|---:|---:|
| no_action | 0.0922 | 0.4570 | 0.2757 |
| shuffled_action | 0.0969 | 0.4578 | 0.2799 |
| action_only | 0.1151 | 0.5283 | 0.3069 |

Goal planning on action checkpoint:

| variant | best_step_acc | distance_reduction_rate |
|---|---:|---:|
| action | 0.3691 | 0.3691 |
| no_action | 0.3645 | 0.3645 |
| shuffled_action | 0.3670 | 0.3670 |
| action_only | 0.3689 | 0.3689 |
| random | 0.3675 | 0.3675 |
| oracle | 1.0000 | 1.0000 |

Readout: still fails. Hard-negative training creates a small top1 margin over
no-action (`+1.5pt`), but action-only is equal or better, and goal planning is
indistinguishable from random.

## Current Decision

The current SimplePointEncoder Phase-0/1 implementation is not sufficient
evidence for ViewAction-NEPA as a world-model claim. It should not be scaled to
large multi-node runs in this form.

Supported:

- The ABCI pipeline works.
- ShapeNet55 cache construction works with useful visibility counts.
- Leakage controls and official split handling are now in place.
- Anti-collapse and same-source hard-negative variants run correctly.

Not supported:

- `action-conditioned >> no-action` retrieval.
- shuffled-action collapse relative to action by the required margin.
- goal-view planning above random.
- 2-step rollout advantage.

## Next Strategy

Do not spend more ABCI time on larger SimplePointEncoder sweeps.

The next meaningful loop is one of:

1. Replace `SimplePointEncoder` with a PointGPT/Point-MAE style view-sensitive
   backbone and keep the same controls.
2. Redesign the action task so action-only cannot solve view identity:
   randomized per-shape camera graph orientation, held-out camera graph, and
   cross-object candidate pools.
3. Add a view-sensitive auxiliary target that cannot be satisfied by object
   identity alone, then re-run the same no-action/shuffled/action-only controls.

Until one of these passes, ViewAction-NEPA should remain a diagnostic track,
not the AAAI main route.
