# Classification Migration Plan: PatchCls + Ray Patch (Option A)

Last updated: 2026-02-28

## 1. Goal

Unify all future classification experiments under a single backbone family:

- `patchcls` backbone (`nepa3d/train/finetune_patch_cls.py`)
- Option-A ray binding (`ray -> nearest point patch center`)
- no new benchmark claims from legacy/tokenizer-only classifier routes

This migration is required to make stage-2 (NEPA pretrain on patch tokens) coherent.

## 2. Canonical references

- Benchmark canonical: `nepa3d/docs/active/benchmark_scanobjectnn_variant.md`
- Run ledger: `nepa3d/docs/active/runlog_202602.md`

A/B naming policy:

- `A/B` always means pretrain family (`Run A`/`Run B`) in benchmark tables.
- local factor runs use non-overlapping labels (`F1..`, `R1..`) to avoid ambiguity.

## 3. Classification inventory (what existed)

From docs + logs, classification work has three major branches:

1. `finetune_cls.py` tokenizer-sequence branch (legacy and review chain)
2. `patchcls` scratch/parity branch (current patch baseline)
3. Point-MAE strict sanity branch (external baseline parity)

Only branch (2) is retained as the main classification backbone for forward runs.

## 4. Deprecation policy (from now)

For new classification benchmark evidence:

- do not launch new tokenizer-sequence classifier runs as headline evidence
- do not add new rows from `scanobjectnn_main_split_v2`
- use variant caches only:
  - `data/scanobjectnn_obj_bg_v3_nonorm`
  - `data/scanobjectnn_obj_only_v3_nonorm`
  - `data/scanobjectnn_pb_t50_rs_v3_nonorm`

## 5. New primary classifier spec

## 5.1 Backbone

- `MODEL_SOURCE=patchcls`
- point patch: FPS+kNN (`num_groups=64`, `group_size=32`, `n_point=1024`)

## 5.2 Ray patch (Option A)

Implemented in `patchcls`:

- sample ray pool from cache (`ray_o_pool`, `ray_d_pool`, `ray_t_pool`, `ray_hit_pool`)
- compute anchor per ray:
  - hit: `x = o + t d`
  - miss: `x = o + miss_t d`
- assign ray to nearest point patch center
- per-ray encode: `[x-center, d, hit, t] -> MLP`
- per-patch pool (`max`/`mean`)
- fuse with point patch token (`concat+proj` or `add`)

Current control flags (`finetune_patch_cls.py`):

- `--use_ray_patch 0/1`
- `--n_ray`
- `--ray_sample_mode_train`, `--ray_sample_mode_eval`
- `--ray_pool_mode {max,mean}`
- `--ray_fuse_mode {concat,add}`
- `--ray_hidden_dim`
- `--ray_miss_t`, `--ray_hit_threshold`

## 5.3 Validation policy

Benchmark-comparison default:

- use Point-MAE-style split policy (`val_split_mode=pointmae`) unless an ablation explicitly targets split effects.

## 6. Stage plan

## Stage S0 (sanity, required)

Verify `patchcls + ray patch` trains/evaluates stably on all 3 variants.

Runs:

1. ray OFF baseline (`use_ray_patch=0`)
2. ray ON (`use_ray_patch=1`, `ray_fuse_mode=concat`, `ray_pool_mode=max`)

for each variant (`obj_bg`, `obj_only`, `pb_t50_rs`) under same recipe.

## Stage S1 (classification ablation)

On `obj_only` then full 3 variants:

- `ray_fuse_mode`: `concat` vs `add`
- `ray_pool_mode`: `max` vs `mean`
- `n_ray`: `64/128/256`

## Stage S2 (A/B pretrain integration)

After patch-token NEPA pretrain path is connected:

- evaluate `A` and `B` checkpoints through the same `patchcls + ray patch` classifier
- report SOTA-fair and NEPA-full separately under variant-split protocol

## 7. Immediate execution order

1. finish S0 sanity submission and collect `test_acc`
2. update `runlog_202602.md` with S0 results
3. start S1 minimal grid on `obj_only`
4. then launch S2 A/B benchmark reruns
