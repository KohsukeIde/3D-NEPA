# Noise-NEPA / Denoising-Time NEPA

This track is a drop-in experimental addition for `KohsukeIde/3D-NEPA`.
It is designed after checking the repo organization of PointGPT, PointDif, and Point-MaDi:

- PointGPT / PointDif / Point-MaDi all keep a `cfgs/`, `models/`, `datasets` or `data/`, `tools` / `scripts`, and main training entrypoint structure.
- This track mirrors that style but is intentionally independent, so it does not mutate `PointGPT/` or existing PointNEPA / ViewAction-NEPA code.

## Core idea

Existing 3D diffusion pretraining methods usually reconstruct clean or masked point clouds. Noise-NEPA instead treats diffusion time as a controlled latent dynamics axis:

```text
x_t  --E_online--> z_t
(t -> s) --TimeEncoder--> e_ts
F(z_t, e_ts) -> z_hat_s
x_s  --E_target/EMA--> z_s
loss = 1 - cosine(z_hat_s, stopgrad(z_s))
```

No point generator, no Chamfer reconstruction, and no masked point decoder are required in the smoke version.

## Why this track exists

Previous routes exposed failure modes:

- PointNEPA / skip-k: spatial next-token prediction overuses local continuity / center signals.
- PosetNEPA-Lite: changing spatial order did not transfer downstream.
- ViewAction-NEPA: camera-view action conditioning failed even after per-shape camera randomization.

Noise-time gives a natural predictive axis that point clouds lack spatially:

> Point clouds have no natural patch order, but denoising trajectories have a natural time order.

## What is included

### Pre-mortem scripts

Run these before training. They test whether the idea is viable.

1. `premortem_initial_cosine.py`  
   Checks whether frozen features at two noise levels are already too similar. If cosine > 0.95, identity prediction may be enough.

2. `premortem_noise_id.py`  
   Checks whether noise level is trivially identifiable from features. If noise bin is too easy, time conditioning may be a shortcut.

3. `premortem_latent_path.py`  
   Checks whether latent trajectories across noise levels are smooth enough for semigroup prediction.

4. `premortem_noise_visualize.py`  
   Saves quick 2D projections of noisy point clouds.

### Training

- `train/train_noise_nepa.py` trains the one-step latent denoising transition.
- EMA target encoder is supported.
- Optional semigroup consistency can be enabled.
- `simple` encoder works out of the box; `PointGPT` adapter is included as a best-effort hook.

### Evaluation

- `eval/eval_denoising_time_retrieval.py`
- `eval/eval_semigroup_consistency.py`

## Minimal run

From repo root:

```bash
unzip noise_nepa_addition.zip -d .
```

Pre-mortem:

```bash
DATA_ROOT=/path/to/shapenet_pc \
MAX_SHAPES=500 \
bash noise_nepa/scripts/00_premortem_all.sh
```

Smoke train:

```bash
DATA_ROOT=/path/to/shapenet_pc \
OUT=outputs/noise_nepa/smoke_timecond \
EPOCHS=50 \
BATCH_SIZE=64 \
bash noise_nepa/scripts/01_train_smoke_timecond.sh
```

Baselines:

```bash
DATA_ROOT=/path/to/shapenet_pc \
OUT=outputs/noise_nepa/smoke_notime \
bash noise_nepa/scripts/02_train_no_time_baseline.sh

DATA_ROOT=/path/to/shapenet_pc \
OUT=outputs/noise_nepa/smoke_shuffled_time \
bash noise_nepa/scripts/03_train_shuffled_time_baseline.sh
```

Evaluate:

```bash
CKPT=outputs/noise_nepa/smoke_timecond/ckpt_last.pth \
DATA_ROOT=/path/to/shapenet_pc \
bash noise_nepa/scripts/04_eval_retrieval.sh

CKPT=outputs/noise_nepa/smoke_timecond/ckpt_last.pth \
DATA_ROOT=/path/to/shapenet_pc \
bash noise_nepa/scripts/05_eval_semigroup.sh
```

## Go / no-go

See `docs/pf_kill_tests_active.md`.

The short version:

- Pre-mortem pass: initial cosine between moderate noise levels should not be > 0.95 everywhere, and should not be < 0.5 everywhere.
- Training pass: time-conditioned retrieval should beat no-time and shuffled-time by >= 10 points.
- Semigroup pass: rollout prediction should beat no-time and identity controls.
- If none of these pass, do not proceed to ScanObjectNN or PointGPT integration.
