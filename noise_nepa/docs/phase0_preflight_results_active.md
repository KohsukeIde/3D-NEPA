# Noise-NEPA Phase 0 Preflight Results

Status: active.

## Context

ViewAction-NEPA is removed from the AAAI main route after per-shape randomized camera-frame diagnostics failed to produce an action-conditioned margin. Noise-NEPA / Vector C is the next smoke track:

> use denoising time as the prediction axis, and learn decoder-free latent denoising dynamics with optional semigroup consistency.

## Local Pre-mortem Mini

Command used 50 shapes, 512 points:

```bash
DATA_ROOT=/groups/gag51402/datasets/ShapeNet55-34/shapenet_pc \
OUT=outputs/noise_nepa/premortem_mini \
MAX_SHAPES=50 NPOINTS=512 BATCH_SIZE=32 \
bash noise_nepa/scripts/00_premortem_all.sh
```

Results:

| Diagnostic | Metric | Value | Interpretation |
|---|---|---:|---|
| initial cosine | `cos_mean_t750_s350` | 0.991967 | high identity-shortcut risk |
| initial cosine | `cos_mean_t650_s250` | 0.981822 | high identity-shortcut risk |
| initial cosine | `cos_mean_t500_s100` | 0.948310 | still highly aligned |
| noise id | chance | 0.200000 | 5 bins |
| noise id | test acc | 0.340000 | noise level is weakly identifiable, not trivial |
| latent path | `step_cos_mean` | 0.992430 | path is smooth but near-identity |
| latent path | `endpoint_cos_mean` | 0.935417 | high similarity even across large noise gap |

Important caveat: these pre-mortems use the smoke `SimplePointEncoder` unless an actual pretrained/frozen encoder is wired in. They are implementation diagnostics, not evidence for the final backbone.

## Local Runtime Debug

Command used 80 shapes, 256 points, 5 train steps:

```bash
DATA_ROOT=/groups/gag51402/datasets/ShapeNet55-34/shapenet_pc \
OUT=outputs/noise_nepa/local_debug_time_seeded \
MAX_SHAPES=80 NPOINTS=256 BATCH_SIZE=16 EPOCHS=1 MAX_STEPS=5 \
NUM_WORKERS=0 DEVICE=cpu SEED=7 \
bash noise_nepa/scripts/01_train_smoke_timecond.sh
```

Training completed and wrote `ckpt_last.pth`.

## Fixes Applied Before ABCI Smoke

- Added deterministic `SEED` plumbing for train/eval.
- Made fixed eval deterministic but varied by shape, so shuffled-time is meaningful.
- Ensured eval variants compare the same `x_t/x_s/x_r` tensors across separate processes.
- Added semigroup eval variants: `time`, `no_time`, `shuffled_time`, `identity`.
- Added direct `t -> r` target term to semigroup training loss.
- Added same-shape noise-level retrieval to measure denoising-time prediction rather than only shape identity.
- Updated ABCI submit script to PBS/qsub style used by this repo.

## Local Debug Metrics After Fixes

Cross-shape denoising retrieval on the 5-step debug checkpoint is not evidence of quality, but confirms scripts run:

| Variant | top1 | top5 | MRR | cos target | cos current |
|---|---:|---:|---:|---:|---:|
| time | 0.0125 | 0.0125 | 0.0632 | 0.9865 | 0.9273 |
| no_time | 0.0000 | 0.0250 | 0.0444 | 0.9853 | 0.9234 |
| shuffled_time | 0.0125 | 0.0375 | 0.0579 | 0.9847 | 0.9273 |
| identity | 0.0000 | 0.0500 | 0.0481 | 0.9321 | 1.0000 |

Same-shape noise-level retrieval, 7 candidates including current view, chance top1 0.1429:

| Variant | top1 | top3 | MRR | current select |
|---|---:|---:|---:|---:|
| time | 0.2000 | 0.4125 | 0.4029 | 0.2750 |
| no_time | 0.1500 | 0.4375 | 0.3665 | 0.0375 |
| shuffled_time | 0.1375 | 0.3875 | 0.3567 | 0.2125 |
| identity | 0.1750 | 0.4875 | 0.4080 | 0.0000 |

This debug run is too small for a go/no-go decision. It mainly shows why same-shape noise-level retrieval is required: cross-shape retrieval alone can be dominated by object identity.

## ABCI Smoke

Submitted:

```text
job_id: 1779643.pbs1
out: outputs/noise_nepa/noise_nepa_p0_seed0_20260520_215351
max_shapes: 500
npoints: 512
epochs: 20
max_steps: 200
batch_size: 64
semigroup_weight: 0.1
seed: 0
```

The run trains:

- time-conditioned
- no-time baseline
- shuffled-time baseline

It evaluates:

- cross-shape denoising-time retrieval
- same-shape noise-level retrieval
- semigroup consistency

## ABCI Smoke Results

Run completed successfully.

### Premortem

| Diagnostic | Metric | Value |
|---|---|---:|
| initial cosine | `cos_mean_t750_s350` | 0.991314 |
| initial cosine | `cos_mean_t650_s250` | 0.984168 |
| initial cosine | `cos_mean_t500_s100` | 0.957812 |
| noise id | test acc | 0.480000 |
| latent path | `step_cos_mean` | 0.992240 |
| latent path | `endpoint_cos_mean` | 0.927796 |

The high initial cosine remains an identity-shortcut risk. Noise-bin identification is above chance but not saturated.

### Main Time-Conditioned Checkpoint

Same-shape noise-level retrieval, 7 candidates including current, chance top1 0.1429:

| Eval variant | top1 | top3 | MRR | current select |
|---|---:|---:|---:|---:|
| time | 0.6180 | 0.9160 | 0.7668 | 0.2220 |
| no_time | 0.1680 | 0.5040 | 0.4058 | 0.0000 |
| shuffled_time | 0.1300 | 0.4240 | 0.3622 | 0.2140 |
| identity | 0.0860 | 0.3700 | 0.3231 | 0.5080 |

Cross-shape denoising-time retrieval:

| Eval variant | top1 | top5 | MRR | cos target | cos current |
|---|---:|---:|---:|---:|---:|
| time | 0.1080 | 0.3320 | 0.2270 | 0.9997 | 0.9961 |
| no_time | 0.0020 | 0.0100 | 0.0134 | 0.9855 | 0.9785 |
| shuffled_time | 0.0040 | 0.0120 | 0.0174 | 0.9955 | 0.9960 |
| identity | 0.0040 | 0.0140 | 0.0183 | 0.9958 | 1.0000 |

Semigroup eval:

| Eval variant | direct top1 | rollout top1 | direct target cos | rollout target cos | direct/rollout cos |
|---|---:|---:|---:|---:|---:|
| time | 0.0220 | 0.0280 | 0.9997 | 0.9997 | 0.9999 |
| no_time | 0.0000 | 0.0020 | 0.9953 | 0.9905 | 0.9985 |
| shuffled_time | 0.0020 | 0.0000 | 0.9952 | 0.9951 | 0.9999 |
| identity | 0.0020 | 0.0020 | 0.9863 | 0.9863 | 1.0000 |

### Baseline Checkpoints

The no-time and shuffled-time trained checkpoints stay near chance on same-shape noise-level retrieval:

| Checkpoint | time eval top1 | no-time eval top1 | shuffled eval top1 | identity top1 |
|---|---:|---:|---:|---:|
| no-time trained | 0.1700 | 0.1700 | 0.1700 | 0.0160 |
| shuffled-time trained | 0.1660 | 0.1560 | 0.1700 | 0.0300 |

## Phase 0 Interpretation

This is the first positive internal-dynamics result after the ViewAction failure:

- time-conditioned latent transition clearly beats no-time, shuffled-time, and identity on the hard same-shape noise-level task;
- shuffled-time does not reproduce the time-conditioned result;
- identity/current is not enough for noise-level retrieval.

The remaining weakness is semigroup evidence. Direct and rollout cosines are very high, and top1 remains low. This means the current smoke supports one-step denoising-time conditioning, but not yet a strong semigroup/world-model-style claim.

## Current Decision Rule

Continue Noise-NEPA only if the ABCI smoke shows:

- same-shape noise-level retrieval: `time` clearly beats `no_time`, `shuffled_time`, and `identity`;
- cross-shape retrieval: `time` is not only preserving shape identity;
- semigroup: direct and rollout are both above no-time/shuffled and do not collapse to current/identity;
- current-select rate is not the main source of top1.

If the same-shape noise-level eval does not show a margin, Vector C should be treated as another weak smoke rather than escalated to PointGPT/full pretraining.

Current decision after this run: continue to a narrow Phase 0.5 focused on semigroup hardening and shortcut controls. Do not jump to full PointGPT pretraining yet.
