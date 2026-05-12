# Config Inventory

Last updated: 2026-03-30

## Purpose

This file is the canonical ownership inventory for `nepa3d/configs/`.

Use it to answer:

- which track owns each top-level YAML family,
- whether the top-level file is a canonical config or a compatibility copy,
- where the track-local canonical home already exists,
- which families can be migrated later without breaking maintained wrappers.

## Snapshot

As of 2026-03-30:

- top-level YAML count in `nepa3d/configs/`: `72`
- exact duplicates of track-local configs: `29`
- top-level-only configs: `43`
- all `29` duplicates are byte-identical to their track-local counterpart

Interpretation:

- `nepa3d/configs/` is no longer a single ownership surface
- it currently mixes:
  - compatibility copies of track-owned configs
  - launcher-facing top-level configs that have not yet been migrated

## Canonical Rule

When a track-local duplicate already exists:

- the track-local file is the canonical ownership location
- the top-level file is retained for launcher / legacy-path compatibility

When no track-local duplicate exists yet:

- ownership is still assigned to a track family
- but the current maintained launcher-facing path may remain top-level until
  wrappers and active docs migrate together

## Family Inventory

| family | files | owner | canonical home now | top-level role now | current refs | next action |
|---|---:|---|---|---|---|---|
| QueryNEPA legacy pretrain / finetune configs | `10` | `tracks/query_nepa` | `nepa3d/tracks/query_nepa/configs/` | exact compatibility copies | `docs=18`, `scripts=23` | keep top-level duplicates for now; do not add new QueryNEPA YAML here |
| PatchNEPA mainline unpaired mix | `1` | `tracks/patch_nepa/mainline` | `nepa3d/tracks/patch_nepa/mainline/configs/` | exact compatibility copy | `docs=81`, `scripts=6` | keep top-level duplicate because historical docs reference it heavily |
| PatchNEPA token-path configs | `17` | `tracks/patch_nepa/tokens` | `nepa3d/tracks/patch_nepa/tokens/configs/` | exact compatibility copies | `docs=20`, `scripts=8` | migrate wrappers later; canonical ownership already moved |
| PatchNEPA CQA base smoke config | `1` | `tracks/patch_nepa/cqa` | `nepa3d/tracks/patch_nepa/cqa/configs/` | exact compatibility copy | `docs=2`, `scripts=0` | maintained wrappers already point track-local; keep top-level copy for historical compatibility only |
| PatchNEPA CQA derived config pool | `43` | `tracks/patch_nepa/cqa` | owner is PatchNEPA CQA, but files still live only in `nepa3d/configs/` | maintained launcher-facing top-level pool | `docs=36`, `scripts=126` | keep top-level for now; move only family-by-family with wrapper/doc updates |

## Exact-Duplicate Families

These top-level files already have a byte-identical canonical copy under
`nepa3d/tracks/*/configs/`.

### QueryNEPA legacy family

Canonical home:

- `nepa3d/tracks/query_nepa/configs/`

Files:

- `finetune_cls_modelnet40.yaml`
- `pretrain_mixed_eccv.yaml`
- `pretrain_mixed_shapenet_mesh_udf.yaml`
- `pretrain_mixed_shapenet_mesh_udf_onepass.yaml`
- `pretrain_mixed_shapenet_mesh_udf_scan.yaml`
- `pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml`
- `pretrain_mixed_shapenet_pointcloud_only.yaml`
- `pretrain_mixed_shapenet_pointcloud_only_onepass.yaml`
- `pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml`
- `pretrain_querynepa3d_modelnet40.yaml`

Policy:

- read these as QueryNEPA-owned
- keep top-level copies because old scripts and historical docs still use the
  `nepa3d/configs/...` path

### PatchNEPA mainline family

Canonical home:

- `nepa3d/tracks/patch_nepa/mainline/configs/`

Files:

- `shapenet_unpaired_mix.yaml`

Policy:

- ownership is already PatchNEPA mainline
- do not remove the top-level copy while completion / historical docs still
  cite it broadly

### PatchNEPA token-path family

Canonical home:

- `nepa3d/tracks/patch_nepa/tokens/configs/`

Files:

- `shapenet_unpaired_mix_v2_queryless_drop1_pc100.yaml`
- `shapenet_unpaired_mix_v2_tokens.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh100.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50_cm15655.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50_visocc.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_mesh50_udf50_visocc_base.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc100.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc100_cm15655.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_cm15655.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_pc33_mesh33_udf33_visocc_base.yaml`
- `shapenet_unpaired_mix_v2_tokens_drop1_udf100.yaml`
- `shapenet_unpaired_mix_v2_tokens_mesh50_udf50.yaml`
- `shapenet_unpaired_mix_v2_tokens_pc100.yaml`
- `shapenet_unpaired_mix_v2_tokens_pc33_mesh33_udf33.yaml`

Policy:

- new token-path YAML should be added under
  `nepa3d/tracks/patch_nepa/tokens/configs/`
- top-level duplicates can be retired later by repointing maintained wrappers

### PatchNEPA CQA base family

Canonical home:

- `nepa3d/tracks/patch_nepa/cqa/configs/`

Files:

- `shapenet_unpaired_mix_v2_cqa.yaml`

Policy:

- canonical ownership is already track-local
- maintained wrappers now default to the track-local path
- top-level duplicate remains only for historical / manual compatibility

## Top-Level-Only CQA Derived Pool

These configs are owned by `nepa3d/tracks/patch_nepa/cqa/`, but today they
still exist only under `nepa3d/configs/`.

Why they stay top-level for now:

- maintained ABCI wrappers default to these paths
- active PatchNEPA docs and runlogs cite these paths directly
- some families are used as same/offdiag paired eval configs in wrappers, so a
  move requires synchronized script edits

### Current CQA derived families

Distance / normal families:

- `shapenet_unpaired_mix_v2_cqa_dist_norm*.yaml`
- `shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned*.yaml`
- `shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_continuous*.yaml`
- `shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_ao_continuous*.yaml`
- `shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_viscount*.yaml`

Thickness / AO / mesh-continuous families:

- `shapenet_unpaired_mix_v2_cqa_dist_thick_valid_qbin*.yaml`
- `shapenet_unpaired_mix_v2_cqa_mesh_ao_continuous*.yaml`

Single-task / early branch families:

- `shapenet_unpaired_mix_v2_cqa_udfdist.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfdist_continuous*.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfdist_near.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfdist_near_pcdiag.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfsurf.yaml`
- `shapenet_unpaired_mix_v2_cqa_udfpcdiag.yaml`
- `shapenet_unpaired_mix_v2_cqa_legacy_visthick_smoke.yaml`

Maintained v2/v3 multitype families:

- `shapenet_unpaired_mix_v2_cqa_v2_dist_norm_*.yaml`
- `shapenet_unpaired_mix_v2_cqa_v3_*.yaml`

Policy:

- ownership is PatchNEPA CQA
- current launcher-facing path remains top-level
- new configs in these families should be added track-local first, and mirrored
  top-level only when a maintained wrapper still requires the old path

## Migration Risk

### Low-risk / early candidates

- `shapenet_unpaired_mix_v2_cqa.yaml`
  - few references
  - track-local canonical copy already exists
- token-path duplicates referenced only by a small set of maintained wrappers
  plus active docs

### Medium-risk

- QueryNEPA duplicate family
  - active work is low, but historical docs and scripts still reference the
    top-level paths

### High-risk

- `shapenet_unpaired_mix.yaml`
  - appears widely in historical completion docs
- CQA derived pool
  - large script surface with many ABCI wrappers using the top-level path as
    the maintained default

## Current Rules For New Configs

1. New QueryNEPA configs go under `nepa3d/tracks/query_nepa/configs/`.
2. New PatchNEPA token configs go under
   `nepa3d/tracks/patch_nepa/tokens/configs/`.
3. New PatchNEPA mainline configs go under
   `nepa3d/tracks/patch_nepa/mainline/configs/`.
4. New PatchNEPA CQA configs should be authored under
   `nepa3d/tracks/patch_nepa/cqa/configs/` first.
5. Add a top-level mirror only when a maintained wrapper or active doc still
   requires `nepa3d/configs/...`.
6. Do not move or delete a top-level config in isolation; update wrappers and
   active guide docs in the same pass.

## Next Practical Step

The safest next migration order is:

1. repoint maintained token-path wrappers family-by-family
2. migrate active guide docs that still present top-level token/CQA paths as
   primary
3. leave heavy historical doc families and the large CQA derived pool for last
