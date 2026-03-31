# Migration: Cross-Primitive To Geo-Teacher

Last updated: 2026-04-01

## 1. Purpose

This file records what changes, what stays, and what should be retired from the
paper-facing language as the project moves from a cross-primitive story to a
geo-teacher story.

## 2. Old Claim vs New Claim

| old claim | new claim |
|---|---|
| learn symmetric shared representations across primitive types | pretrain point/surface context encoders with derived geometric teacher targets |
| cross-primitive transfer is the headline | degraded-context robustness is a supporting evaluation |
| unpaired mesh/UDF split defines the dataset identity | same-shape packed manifests define the paper-facing protocol |
| mesh / udf / pc are symmetric inputs | clean shape sources produce teacher targets; weak point/surface carriers provide context |

## 3. Terms To Deprecate

Deprecated paper-facing terms:

- `cross-primitive transfer`
- `primitive-native answers`
- `unpaired CQA` as the default dataset label

Replacement terms:

- `derived geometric teacher targets`
- `context-backend shift`
- `degraded-context transfer`
- `same-shape packed multi-task training`

## 4. What We Keep

Keep as valid evidence under the new framing:

- same-context `surf -> udf_distance`
- packed same-shape task support in the loader
- `surf -> pc_bank` zero-shot evaluation, but renamed as degraded-context
  evaluation
- dense `udf_distance` completion evidence
- historical `recong2` docs as provenance for how the branch evolved

## 5. What Becomes Historical

Move these to historical motivation / archive language:

- symmetric primitive-input framing
- `train_mesh` / `train_udf` as the default paper-facing split identity
- using `cross-primitive` as the central collaborator-facing description

## 6. Immediate Operational Migration

1. keep `world_v3` raw cache frozen
2. add same-shape `train / val / eval` manifests
3. define Tier-1 packed configs
4. keep `surf` as the default training context
5. reframe `pc_bank` as degraded-context evaluation
6. keep historical docs reachable for provenance, not as the new paper guide

## 7. Implementation Reality Check

Do not promise a runtime migration that has not happened yet.

Today:

- dataset implementation layer is still `v2_cqa`
- discrete answer implementation layer is still `cqa_v2`
- paper-facing semantic layer can still move now

So the safe order is:

- docs migration first
- manifest / config migration second
- runtime alias cleanup later if needed
