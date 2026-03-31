# Geo-Teacher Vocabulary Spec v1

Last updated: 2026-04-01

## 1. Purpose

This file defines the paper-facing canonical task vocabulary for the
geo-teacher line.

It is **not** a claim that a new runtime codec named `geo_teacher_v1` already
exists in code. It is a semantic paper-layer spec that maps onto the current
task registry and codec implementation.

## 2. Current Runtime Boundary

Current code facts:

- historical default codec constant is still `cqa_v1`
- the current multi-answer discrete mainline is `cqa_v2`
- `cqa_v3` keeps the same task set as `cqa_v2` but increases rescued thickness
  resolution

For the immediate paper-facing line, the practical default is:

- paper semantics: `geo_teacher_v1`
- runtime discrete codec: `cqa_v2`

## 3. Canonical Paper Tasks

Tier-1 canonical tasks:

- `udf_distance`
- `mesh_normal_unsigned`
- `udf_thickness_valid_qbin`

Tier-2 / supplemental:

- `mesh_ao_hq`

## 4. Task Mapping To Current Code

| paper task | current registry name | current query name | current discrete path |
|---|---|---|---|
| distance | `udf_distance` | `udf_distance` | `cqa_v2` |
| unsigned normal | `mesh_normal_unsigned` | `mesh_normal_unsigned` | `cqa_v2` |
| thickness valid-qbin | `udf_thickness_valid_qbin` | `udf_thickness_valid_qbin` | `cqa_v2` |
| AO-HQ supplemental | `mesh_ao_hq` | `mesh_ao` | `cqa_v2` |

Important:

- `mesh_ao_hq` is already a separate task name in the task registry
- it currently reuses `query_name="mesh_ao"` and `encode_mode="mesh_ao"`
  while reading `mesh_surf_ao_hq`

## 5. Deprecated Historical Vocabulary

These remain historical or implementation-level terms, but should not be the
paper-facing vocabulary center:

- `primitive-native answers`
- `cross-primitive transfer`
- `unpaired CQA` as the default dataset identity

Preferred replacements:

- `derived geometric teacher targets`
- `degraded-context transfer` or `context-backend shift`
- `same-shape packed multi-task protocol`

## 6. Why AO-HQ Is Not Tier-1

`mesh_ao_hq` is useful, but it should start as supplemental:

- current strongest shared read is still `distance + normal`
- rescued thickness already has a stable coded path and is closer to the
  current headline-safe multi-task route
- AO-HQ can stay as a second-phase teacher extension

## 7. Future Codec Note

If the paper-facing line later needs a dedicated runtime codec, create it as a
separate implementation step.

Until that step exists, do not overclaim a new runtime name in configs or docs.
Use the current code honestly:

- semantic layer: `geo_teacher_v1`
- runtime codec today: `cqa_v2`
