# Claude Context Pack: PatchNEPA / Geo-Teacher

Last updated: 2026-05-12

## Purpose

This is the compact handoff for Claude or another LLM when the full `nepa3d`
documentation set is too large.

Read this before opening raw runlogs. QueryNEPA documents are intentionally not
part of the default reading pack; they are historical provenance only.

## Default Reading Pack

Use these files first, in this order:

1. `nepa3d/docs/patch_nepa/current_llm_brief_active.md`
2. `nepa3d/docs/patch_nepa/claude_context_active.md`
3. `nepa3d/docs/patch_nepa/paper_direction_geo_teacher_202604.md`
4. `nepa3d/docs/patch_nepa/dataset_geo_teacher_v1_spec.md`
5. `nepa3d/docs/patch_nepa/experiment_route_ab_matrix_202604.md`
6. `nepa3d/docs/_meta/insight_register_active.md`

Add only when needed:

- ScanObjectNN headline boundary:
  `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
- file-split demotion rule:
  `nepa3d/docs/patch_nepa/scanobjectnn_ft_policy_audit_active.md`
- Itachi-local result boundary:
  `nepa3d/docs/patch_nepa/itachi/results_geo_teacher_itachi_active.md`
- PointGPT / pointNEPA sidecar boundary:
  `pointnepa/docs/results_scanobjectnn_active.md`

## Exclude By Default

Do not put these in the first Claude context window:

- `nepa3d/docs/query_nepa/runlog_202602.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- full raw PatchNEPA runlogs, unless exact job provenance is requested

Carry only this QueryNEPA lesson forward:

- QueryNEPA taught protocol hygiene and failure modes.
- It is not current benchmark evidence.
- Mixed historical ScanObjectNN cache lines and old validation policies must not
  be used as current paper claims.

## Current Truth

- The paper-facing story is derived geometric teacher pretraining for
  point-context encoders, short name `geo-teacher`.
- Do not describe the active direction as symmetric cross-primitive input
  learning.
- Historical `recong2` remains useful provenance and an old transfer candidate,
  but its ScanObjectNN rows are file-split historical until rerun under official
  `test-as-val`.
- Route A is alive only if matched 100-epoch geo-teacher pretraining improves
  ScanObjectNN and ShapeNetPart transfer.
- Route B remains alive if the strongest evidence is direct geometry readouts,
  controls, and completion.
- PointGPT / pointNEPA rows are sidecar comparison context, not PatchNEPA
  headline results.
- Itachi-local rows are local evidence until copied into canonical benchmark
  docs.

## Dataset / Protocol Timeline

### 2026-02: ScanObjectNN cache normalization correction

The early ScanObjectNN sanity gap was not just a label or split problem.

- Older NEPA cache-derived H5 used normalized `pc_xyz`, so it did not match the
  official H5 point distribution.
- The `v3_nonorm` cache plus dynamic query bbox made cache-derived H5 match
  official H5 exactly under the same Point-MAE checkpoint mapping.
- Canonical ScanObjectNN variant caches became:
  - `data/scanobjectnn_obj_bg_v3_nonorm`
  - `data/scanobjectnn_obj_only_v3_nonorm`
  - `data/scanobjectnn_pb_t50_rs_v3_nonorm`

Implication:

- dataset preprocessing domain shift was real.
- new ScanObjectNN benchmark claims should use the canonical non-normalized
  variant caches and official `test-as-val` selection.

### 2026-03-14: ScanObjectNN validation policy correction

Benchmark-facing ScanObjectNN parity follows the public Point-MAE / PointGPT
practice:

- maintained policy: `val_split_mode=pointmae`, meaning official `test-as-val`
- old PatchNEPA file-split fine-tune rows are historical/internal only

Implication:

- old `recong2` values such as `0.8485 / 0.8589 / 0.8140` are useful ranking
  provenance, not headline-safe benchmark rows.

### 2026-03-16: ShapeNet `world_v3` raw cache freeze

The existing `worldvis` cache became the fixed raw contract.

Key findings:

- frozen-cache shape count: `52,311`
- strict clean watertight subset: `119 / 52,311`
- strict watertight subset is too small to replace the main corpus
- no evidence that a full raw rebuild is required for the next paper step

Implication:

- keep `world_v3` as the raw layer.
- do not rebuild raw NPZs just to change the paper story.
- future changes should first happen through manifests, splits, task selection,
  and packed protocol.

### 2026-03-26 to 2026-03-27: AO-HQ / HKS derived-target wave

Derived-cache augmentation is the right mechanism for target experiments.

Key findings:

- AO-HQ smoke succeeded, then full AO-HQ additive build landed cleanly.
- full AO-HQ build covered `train_mesh 16004/16004`, `eval 5241/5241`,
  `overall 21245/21245`, with `0` errors.
- HKS remained partial and fragile because the eigensolver path was not stable.
- AO-HQ improved target quality but did not overturn the mainline.

Implication:

- AO-HQ is a useful supplemental target-quality upgrade.
- HKS is future-extension material, not near-term mainline.
- `DISTANCE + NORMAL_UNSIGNED` remains the safest current discrete geometry
  anchor.

### 2026-03-30: common-split mixture vs packed isolation

The old result was not merely a split artifact.

Key findings:

- common-split mixture did not collapse relative to the historical anchor.
- packed all-type-per-shape was a small positive over mixture.
- packed helped the main distance row and off-diagonal normal row, but the gain
  was modest.

Implication:

- use packed same-shape protocol for the paper-facing geo-teacher line.
- keep mixture as a legacy scientific control, not the default.

### 2026-04-02: geo-teacher protocol migration

The paper-facing data change is above the raw layer.

Current runnable package:

- `data/shapenet_cache_v2_20260401_worldvis`
- available splits:
  - `train`: `45,047`
  - `test`: `4,995`
- limitations:
  - no materialized `val` split yet
  - no `mesh_surf_ao_hq` in the base cache

First runnable matched compare:

- `udf_distance`
- `udf_distance + mesh_normal_unsigned`
- `packed_budget_unit=shape`
- `replacement=false`
- one full train-shape pass per epoch

Thickness comes next. AO-HQ stays supplemental until promoted by a cleaner
shared-result package.

## Highest-Value Findings To Preserve

- `C034` `DISTANCE + NORMAL_UNSIGNED` is the safest core CQA / geo-teacher
  anchor.
- `C035`, `C042`, and `C043` show that richer target packs can improve utility
  slightly, but they trade off against the core distance/normal answering and
  completion anchor.
- AO-HQ fixes a target-quality issue but is not yet headline-safe as the main
  branch.
- HKS is not operational enough for the near-term mainline.
- packed same-shape protocol is a small positive over common-split mixture.
- external frozen Point-MAE CQA is a real sidecar control, not a replacement for
  the CQA mainline.
- Itachi 300 epochs improves same-context geometry more than Route-A utility;
  the interim matched route-decision budget remains 100 pretrain epochs.
- PointGPT / pointNEPA is comparison context only.

## Good Claude Questions

- What is the current paper-facing direction, in one paragraph?
- Which docs or results should not be used as current truth?
- Is the dataset change a raw-cache rebuild or a manifest/protocol migration?
- Which findings support Route A, and which support Route B?
- Which ScanObjectNN numbers are headline-safe?
- What should be rerun before making a benchmark claim?
