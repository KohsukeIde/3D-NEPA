# Geo-Teacher Hypothesis Matrix v1

Last updated: 2026-04-02

## 1. Purpose

This file keeps the first paper-facing geo-teacher hypotheses stable while the
historical PatchNEPA docs remain intact.

Status labels:

- `supported`
- `open`
- `rejected`
- `engineering pending`

## 2. Matrix

| ID | hypothesis | current status | current evidence | next minimal test |
|---|---|---|---|---|
| G1 | the paper-facing line should be described as geometric-teacher supervision rather than symmetric cross-primitive input learning | supported | raw `world_v3` contract already freezes carriers and atomic observables rather than a paper theme; current CQA code is task/context driven | keep new paper-facing docs on the geo-teacher framing |
| G2 | the first dataset migration should happen at manifest / protocol level rather than raw-cache level | supported | `world_v3` explicitly keeps paired atomic observables and says unpairedness is implemented by split / manifest | build same-shape `train/val/eval` manifests before any raw rebuild |
| G3 | packed same-shape multi-task training is the correct default for the new paper-facing line | supported | `build_packed_pretrain_cqa` already enforces common shape support and one shared context source | create a packed + multihead canonical config for the Tier-1 tasks |
| G4 | `udf_distance + mesh_normal_unsigned + udf_thickness_valid_qbin` is the right Tier-1 task set for the first paper-facing package | open | current docs most strongly support `distance + normal`; thickness rescue is implemented and codec-stable; AO-HQ is useful but less headline-safe | run the 3-task packed route before promoting AO-HQ into the default line |
| G5 | `surf -> pc_bank` should remain in the paper, but as degraded-context evaluation rather than cross-primitive headline evidence | supported | current runtime already switches `context_source` between `surf` and `pc_bank`; the new story centers supervision design rather than modality symmetry | keep off-diagonal controls but rename them in the new paper-facing docs |
| G6 | a dedicated runtime `geo_teacher_v1` codec / dataset version is required before the story can change | rejected | the current implementation can already express the first migration using `v2_cqa` + `cqa_v2` with new manifests/configs | do not block the doc and protocol migration on a new runtime alias |
| G7 | AO-HQ should be in the first headline-safe canonical task set | open | task registry supports `mesh_ao_hq`, but the strongest shared evidence still favors distance/normal and the current schema docs do not center AO-HQ | keep AO-HQ supplemental until a cleaner shared-result package exists |
| G8 | the first fair paper decision should be a matched `100`-epoch compare rather than another CQA feature expansion | supported | reviewer-facing fairness against Point-MAE-style baselines depends more on matched pretraining budget than on adding more teacher types first | run `recon / distance / distance+normal` under one fixed `100`-epoch protocol |
| G9 | the first teacher-target compare should use shape-level packed budgets rather than historical effective task-sample budgets | supported | packed training is now intended as same-shape supervision; `packed_budget_unit=shape` keeps epoch semantics explicit | use `replacement=false` and full-train shape count per epoch in the first matched configs |

## 3. Current Priority Order

1. keep the paper-facing docs fixed around the Route A / Route B decision rule
2. run the matched `100`-epoch matrix: `recon / distance / distance+normal`
3. read the result through `ScanObjectNN + ShapeNetPart` and `same/degraded/control/completion`
4. only then add thickness as the next teacher target

## 4. Source Docs

- `nepa3d/docs/patch_nepa/spec_world_v3_schema.md`
- `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
- `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
- `nepa3d/tracks/patch_nepa/cqa/data/mixed_pretrain_cqa.py`
- `nepa3d/tracks/patch_nepa/cqa/data/dataset_cqa.py`
