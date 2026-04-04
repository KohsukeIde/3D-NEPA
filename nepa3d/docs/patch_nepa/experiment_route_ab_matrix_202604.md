# Route A/B Matched Compare Matrix (2026-04)

Last updated: 2026-04-04

## 1. Purpose

This file fixes the first paper-facing decision rule for the geo-teacher line.

The immediate goal is **not** to make CQA more complex. The immediate goal is
to run a matched pretraining comparison that can decide whether the project
should be written as:

- Route A: a target-design paper in the Point-MAE / PCP-MAE family
- Route B: a geometric capability / typed-answering paper

## 2. Decision Rule

Route A becomes the main paper route if geometric-teacher pretraining is
clearly favorable on transfer readouts under a matched 100-epoch regime.

Required readouts:

- `ScanObjectNN`
- `ShapeNetPart`

Route B remains the main paper route if the strongest signal is instead in
direct geometry readouts and typed answering:

- `same_context`
- `degraded_context`
- `controls`
- `completion`

Do not pick the paper route after ad hoc interpretation. Pick it from this
decision rule.

## 3. Current Interim Budget Choice

After the first local `300`-epoch `distance + normal_unsigned` run, the current
read is:

- Route-A utility (`ScanObjectNN`, `ShapeNetPart`) does not materially improve
  over the `100`-epoch pilot,
- Route-B same-context geometry readouts do improve at `300` epochs.

Therefore, for the immediate compare-and-decide loop, the interim matched
budget is frozen back to **100 pretrain epochs**.

This is an execution decision, not a final paper claim.

## 4. First Matched Matrix

The first compare is deliberately small:

1. internal xyz-reconstruction baseline
2. `udf_distance` only
3. `udf_distance + mesh_normal_unsigned`

All three runs should use:

- `100` pretrain epochs
- the same backbone family
- the same downstream fine-tuning recipe
- the same evaluation policy

`udf_thickness_valid_qbin` is phase 2.

`mesh_ao_hq` is supplemental and should stay out of the first decision matrix.

## 5. Runtime Mapping

For the teacher-target side, the current runnable implementation is:

- runtime dataset layer: `v2_cqa`
- runtime codec layer: `cqa_v2`
- training protocol: `packed + multihead`
- loss reduction: `per_task`
- answer factorization: `independent`
- query interface: `no_q` or `self_q`

For the reconstruction side, the baseline is **not** the CQA trainer. Use the
existing PatchNEPA reconstruction line as the internal xyz-reconstruction
baseline.

## 6. Current Local Data Boundary

The first local package is tied to the current prepared cache:

- cache root: `data/shapenet_cache_v2_20260401_worldvis`
- available splits today: `train`, `test`
- current train shape count: `45,047`
- current test shape count: `4,995`

Important:

- there is no dedicated `val` split in the current local cache yet
- `mesh_surf_ao_hq` is not present in the current base cache
- `udf_surf_thickness` is present, but thickness is not in the first matched
  matrix

## 7. Current Canonical Configs

The first teacher-target configs are:

- `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distance_only_v1.yaml`
- `nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml`

These configs intentionally:

- use the current `worldvis` cache directly
- use `split=train`
- set `packed_budget_unit=shape`
- set `replacement=false`
- set `mix_num_shapes=45047`

This means one CQA pretrain epoch corresponds to one full pass over the current
train-shape support.

## 8. Launcher Boundary

The matched teacher-target launcher should default to:

- `sampling_protocol=packed`
- `head_mode=multihead`
- `loss_balance=per_task`
- `answer_factorization=independent`
- `query_interface_mode=no_q`
- `epochs=100`

Current thin entrypoint:

- `scripts/abci/submit_patchnepa_geo_teacher_compare_pretrain.sh`

## 9. What Not To Claim Yet

Do not claim any of the following until the compare is actually run:

- geometric teachers beat xyz reconstruction on downstream transfer
- Route A is already the chosen paper route
- Route B is discarded
- AO-HQ belongs in the first headline-safe package

This file only fixes the first fair compare.

## 10. Current Itachi Automation Boundary

The current local `itachi` automation covers the subset that is already
runnable and maintained on this machine:

- `ScanObjectNN` direct FT on the three paper-facing variants
- multitype `same_context / degraded_context / controls`
- `udf_distance` completion under same and degraded context

`ShapeNetPart` remains part of the Route-A decision rule, but it is not yet in
the maintained `itachi` post-pretrain chain.
