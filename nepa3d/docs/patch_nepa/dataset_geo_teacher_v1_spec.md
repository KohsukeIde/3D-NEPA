# Geo-Teacher Dataset Spec v1

Last updated: 2026-04-02

## 1. Purpose

This file defines the paper-facing dataset and protocol semantics for the
geometric-teacher line.

It is intentionally separate from `spec_world_v3_schema.md`.

- `spec_world_v3_schema.md` freezes the raw cache contract
- this file freezes the paper-facing split / task / evaluation semantics

## 2. Immediate Answer

Yes, the data processing changes, but **not** first at the raw-cache level.

The primary change is:

- from historical `unpaired` task-specific split roots
- to same-shape paired manifests with shared shape support

The immediate job is manifest / protocol rebuild, not raw NPZ regeneration.

## 3. Raw Layer vs Paper Layer

### Raw layer

Keep the frozen `world_v3` raw NPZ contract.

- atomic observables remain paired inside each NPZ
- carriers remain:
  - `surf_xyz`
  - `udf_qry_xyz`
  - `pc_ctx_bank_xyz`

### Paper layer

Build a new protocol on top:

- one shared shape split for `train / val / eval`
- per-task supervision switched by `task_name`
- packed multi-task training on common shape support

## 4. Recommended Root Layout

```text
data/
  ShapeNetCore.v2/
  shapenet_cache_v2_..._worldvis/
  shapenet_geo_teacher_v1/
    manifests/
      train.txt
      val.txt
      eval.txt
    configs/
      distance_only.yaml
      distance_normal.yaml
      distance_normal_thickness.yaml
      distance_normal_thickness_aohq.yaml
```

Important:

- the raw NPZ payload still comes from the frozen `world_v3`-style source cache
- the new layer mainly adds shape-level manifests and paper-facing config
  choices

## 5. Split Semantics

Historical CQA configs often used:

- `train_mesh`
- `train_udf`
- `eval`

That was acceptable for the old unpaired framing, but it is not the clean
default for the geo-teacher line.

The paper-facing split semantics should be:

- `train`
- `val`
- `eval`

Each split contains the same shape-level raw NPZ format. We do not express
paper-facing task identity by separating `train_mesh` from `train_udf`.

## 6. Canonical Tasks

Tier-1 default tasks:

- `udf_distance`
- `mesh_normal_unsigned`
- `udf_thickness_valid_qbin`

Tier-2 / supplemental:

- `mesh_ao_hq`

These task names already exist in the current CQA task registry. The paper
layer narrows which ones are canonical.

## 7. Context Policy

Default training context:

- `context_source: surf`

Evaluation variants:

- `same_context`
  - train `surf`, eval `surf`
- `degraded_context`
  - train `surf`, eval `pc_bank`
- `local_only`
  - reduced context budget
- `controls`
  - `no_context`
  - `wrong_shape_same_synset`
  - `wrong_shape_other_synset`
  - `shuffled_query`

## 8. Training Protocol

The current code already contains the right training primitive for this:

- `build_packed_pretrain_cqa(...)` requires common shape support across task
  specs
- packed CQA also requires one shared `context_source`

That makes the preferred paper-facing training rule:

- `sampling_protocol = packed`
- `head_mode = multihead`

Why:

- same-shape support is explicit
- task switching happens inside one shared shape set
- this matches the new paper direction better than historical
  task-partitioned roots

## 9. Implementation Note

Do not claim in docs that `dataset_version=geo_teacher_v1` already exists in
runtime if it does not.

The minimal implementation path today is:

- runtime dataset layer: `dataset_version = v2_cqa`
- runtime codec layer: `codec_version = cqa_v2`
- paper-facing semantic layer: `geo_teacher_v1`

So the first migration step is semantic and protocol-level, not a forced loader
rewrite.

## 10. Minimal Migration Plan

1. Keep `world_v3` raw cache frozen.
2. Create shape-level `train / val / eval` manifests on top of that raw cache.
3. Create packed same-shape configs for the Tier-1 tasks.
4. Treat `surf -> pc_bank` as degraded-context evaluation, not the headline
   training identity.
5. Add paper-facing docs without deleting historical `recong2` / unpaired
   provenance docs.

## 10.1 Current Runnable Local Package

As of `2026-04-02`, the prepared local cache that is actually ready to run is:

- `data/shapenet_cache_v2_20260401_worldvis`

Current available splits are:

- `train` (`45,047` shapes)
- `test` (`4,995` shapes)

Current limitations:

- `val` is not materialized yet in the local cache
- `mesh_surf_ao_hq` is not present in the current base cache

Therefore, the first runnable matched compare should use:

- `split=train` for pretraining
- `packed_budget_unit=shape`
- `replacement=false`
- one full train-shape pass per epoch

The first runnable configs are intentionally smaller than the full Tier-1 set:

- `udf_distance`
- `udf_distance + mesh_normal_unsigned`

Thickness remains next, after the first matched compare is in place.

## 11. When Raw Rebuild Becomes Necessary

Raw rebuild is optional for the first paper-facing move.

It becomes necessary only if we decide to formalize one of these:

- a new raw field that is currently used informally or only by augmentation
- a stricter quality-filter contract that must be attached to the raw freeze
- a schema-level distinction that cannot be represented by manifests alone

Until then, manifest rebuild should come first.
