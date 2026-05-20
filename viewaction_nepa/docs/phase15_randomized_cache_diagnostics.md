# ViewAction-NEPA Phase 1.5 Randomized-Cache Diagnostics

Status: active run note, 2026-05-20 JST.

## Purpose

The randomized-cache pass tests whether the Phase 1.5 failure is still present
after removing the fixed global camera-frame shortcut. The builder should rotate
the camera positions by a deterministic random SO(3) transform per shape while
keeping the 12-view graph topology and local action classes intact.

This pass is a control, not a scale-up. Use the same 200-shape budget and rerun
D1-D3 against the randomized cache.

## Build

```bash
RUN_TAG=phase15_randomized_$(date +%Y%m%d_%H%M%S) \
MAX_SHAPES=200 \
SHUFFLE_FILES=1 \
bash viewaction_nepa/scripts/13_phase15_build_randomized_cache.sh
```

The script writes to:

- cache: `data/viewaction_shapenet55_hpr_v12_phase15_randomized_${RUN_TAG}`
- diagnostics: `outputs/viewaction_smoke/phase15_triage_randomized_${RUN_TAG}`

Useful overrides:

- `CAMERA_FRAME_SEED`: per-shape camera-frame randomization seed, default `1505`.
- `SEED`: file shuffle / sampling seed, default `0`.
- `CACHE_ROOT`: explicit cache output path.
- `ON_LOW_VISIBLE=fail`: fail if any built cache falls below `MIN_VISIBLE`.

Verify `metadata.json` contains
`"randomize_camera_frame_per_shape": true`, and spot-check `manifest.json` rows
for `"randomized_camera_frame": true`, `camera_frame_seed`, and
`camera_rotation`.

## Diagnostics

Build and run D1-D3 in one job:

```bash
RUN_DIAGNOSTICS=1 \
CKPT_ROOT=outputs/viewaction_smoke/viewaction_p01_hard_20260519_202419 \
bash viewaction_nepa/scripts/13_phase15_build_randomized_cache.sh
```

Or run diagnostics after a cache already exists:

```bash
CACHE_ROOT=data/viewaction_shapenet55_hpr_v12_phase15_randomized_<tag> \
OUT_DIR=outputs/viewaction_smoke/phase15_triage_randomized_<tag> \
RUN_BUILD=0 \
RUN_DIAGNOSTICS=1 \
MAX_SHAPES=200 \
bash viewaction_nepa/scripts/13_phase15_build_randomized_cache.sh
```

The script evaluates:

- D1 view discriminability with raw descriptors, random encoder, optional Phase
  1 checkpoint encoder, and PointNet.
- D2 raw geometry oracle.
- D3 hard next-view retrieval, using `CKPT_ROOT` to evaluate all Phase 1
  checkpoint variants when available.

## Readout

- Global view-id discrimination is expected to drop toward chance because view
  index no longer maps to a shared absolute camera direction across shapes.
- If target-query raw oracle remains high and action-only drops below
  action-conditioned retrieval after retraining on the randomized cache, one
  stronger view-sensitive backbone smoke is still defensible.
- If action-only remains equal or better, the ViewAction route should be killed
  as an AAAI main route.
- If target-query oracle collapses or visibility counts degrade, inspect
  randomized-cache camera-frame generation before blaming the transition model.
