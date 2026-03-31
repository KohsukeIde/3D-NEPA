# PatchNEPA Paper Direction: Geometric Teacher Pretraining

Last updated: 2026-04-01

## 1. Purpose

This file defines the paper-facing direction added on top of the historical
PatchNEPA docs.

The goal is not to rewrite history. The goal is to add a cleaner layer for the
current paper story while preserving the existing `recong2` and CQA ledgers as
provenance.

## 2. One-Paragraph Direction

Historical motivation in this repo was cross-primitive learning: learn shared
representations across mesh / UDF / point-cloud style inputs and study transfer
across primitive boundaries.

The current paper-facing direction is narrower:

- pretrain point-context encoders from weak point/surface observations
- supervise them with derived geometric teacher targets computed from clean
  shape sources
- study supervision design and context robustness rather than symmetric
  multi-primitive input fusion

In short:

- historical lens: `cross-primitive learning`
- paper-facing lens: `derived geometric teacher pretraining for point-context encoders`

## 3. What Changes

### 3.1 Headline framing

Do not describe the active paper direction as symmetric cross-primitive input
learning.

Describe it as:

- point/surface context encoder pretraining
- heterogeneous geometric teacher targets
- clean-to-degraded context transfer

### 3.2 What becomes historical

These remain valid as historical motivation or protocol provenance, but they
should not be the paper headline:

- `cross-primitive transfer` as the main claim
- `unpaired` as the paper-facing dataset identity
- treating `mesh / udf / pc` as fully symmetric input modalities

### 3.3 What remains scientifically useful

These stay relevant under the new framing:

- same-context `surf -> udf_distance` answering
- zero-shot `surf -> pc_bank` evaluation, but framed as degraded-context or
  context-backend shift
- dense completion from the `udf_distance` line
- packed same-shape multi-task training

## 4. Data Interpretation Boundary

The raw `world_v3` cache contract already says the important part explicitly:

- raw NPZs store atomic observables
- unpairedness is implemented by split / manifest
- the schema freezes raw data semantics, not the paper theme

Therefore, the immediate paper-direction change is **not** "rebuild raw NPZs
for a new theory". The immediate change is:

- keep `world_v3` as the raw contract
- redesign split / manifest / task protocol above that raw layer
- separate historical unpaired protocol docs from paper-facing protocol docs

## 5. Current Implementation Boundary

The current code does **not** yet define a dedicated runtime named
`dataset_version=geo_teacher_v1` or `codec_version=geo_teacher_v1`.

The minimal implementation path today is:

- keep the existing `v2_cqa` dataset codepath
- keep the existing discrete codec family (`cqa_v2` for the current canonical
  discrete tasks)
- change shape splits, manifests, task selection, and training protocol

This distinction matters:

- `geo_teacher_v1` is the new paper-facing semantic layer
- `v2_cqa` / `cqa_v2` remain the nearest current implementation layer

## 6. Canonical Task Direction

The paper-facing canonical task set is intentionally smaller than the full
historical CQA registry.

Tier-1:

- `udf_distance`
- `mesh_normal_unsigned`
- `udf_thickness_valid_qbin`

Tier-2 / supplemental:

- `mesh_ao_hq`

The reason is pragmatic:

- strongest shared gate in the current branch is `DISTANCE + NORMAL`
- rescued thickness is already wired and codec-stable
- AO-HQ is useful but should not be treated as the default headline task

## 7. Evaluation Language

Preferred paper-facing evaluation names:

- `same-context`
  - train and eval on `surf`
- `degraded-context`
  - train on `surf`, eval on `pc_bank`
- `local-only`
  - smaller context budget / harder locality regime
- `controls`
  - `no_context`, `wrong_shape_same_synset`, `wrong_shape_other_synset`,
    `shuffled_query`

Avoid calling `surf -> pc_bank` the main "cross-primitive result". In the new
layer it is a clean-to-degraded observation shift test.

## 8. Source Docs

- raw contract: `nepa3d/docs/patch_nepa/spec_world_v3_schema.md`
- historical mainline guide:
  `nepa3d/docs/patch_nepa/collaborator_reading_guide_active.md`
- historical storyline:
  `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`
- paper-facing dataset semantics:
  `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
- paper-facing vocab semantics:
  `nepa3d/docs/patch_nepa/spec_geo_teacher_vocab_v1.md`
- migration memo:
  `nepa3d/docs/patch_nepa/migration_cross_primitive_to_geo_teacher_202604.md`
