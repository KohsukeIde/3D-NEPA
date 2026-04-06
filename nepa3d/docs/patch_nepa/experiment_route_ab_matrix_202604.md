# Route A/B Matched Compare Matrix (2026-04)

Last updated: 2026-04-05

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

1. `PCP-MAE` baseline
2. `Geo-PCP` (`recon + center + normal`)
3. `Geo-PCP` (`recon + center + normal + thickness`)

All three runs should use:

- `100` pretrain epochs
- the same backbone family
- the same downstream fine-tuning recipe
- the same evaluation policy

Global arbitrary-query `distance` stays in the Route-B CQA harness for now.

`AO_HQ` stays supplemental and should stay out of the first decision matrix.

## 5. Runtime Mapping

For the Route-A side, the current runnable implementation is:

- pretrain engine: `PCP-MAE`
- dataset source: `world_v3` surface carrier cache
- loss family: continuous (`recon + center + optional geometric teachers`)
- compare budget: `100` pretrain epochs
- downstream flow: existing `PCP-MAE` ScanObjectNN / ShapeNetPart fine-tune

For the Route-B side, keep the existing `3D-NEPA` CQA harness:

- `same_context`
- `degraded_context`
- `controls`
- `completion`
- `curvature`

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

The first Route-A configs are:

- `PCP-MAE/cfgs/geopcp/pcp_worldvis_base_100ep.yaml`
- `PCP-MAE/cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml`
- `PCP-MAE/cfgs/geopcp/geopcp_worldvis_base_normal_thickness_100ep.yaml`

These configs intentionally:

- use the current `worldvis` cache directly
- use `split=train`
- set `npoints=1024`
- keep the current PCP-MAE masking / grouping recipe

Implementation status on `itachi`:

- PCP-MAE-native Route-A v1 branch is now present under `PCP-MAE/`
- the local `world_v3` adapter is restricted to `train` and confirms `45,047`
  train NPZs with no `test` contamination
- the three compare arms have passed local forward/backward smoke
- PCP-MAE ScanObjectNN / ShapeNetPart checkpoint loading smoke has also passed
- 3D-NEPA `external_pointmae` now auto-detects PCP-MAE-style positional
  embeddings and can read the smoke checkpoint for Route-B harness use

This means one pretrain epoch corresponds to one full pass over the current
local train support.

## 8. Launcher Boundary

The matched Route-A launcher should default to:

- `4xA100`
- `total_bs=128`
- `epochs=100`
- `PCP-MAE` pretrain entrypoint
- existing `PCP-MAE` downstream fine-tune entrypoints

Current local thin entrypoints:

- `scripts/local/patchnepa_geopcp/build_pcpmae_geopcp_env.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_pretrain_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_ablation_triple_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh`

Runtime requirement:

- the Route-A path is now compiled-first on `itachi`
- `pointnet2_ops` and `chamfer_dist` must resolve to compiled/native backends
- slow fallback is debug-only and is not the normal compare path

## 9. What Not To Claim Yet

Do not claim any of the following until the compare is actually run:

- geometric teachers beat xyz reconstruction on downstream transfer
- Route A is already the chosen paper route
- Route B is discarded
- AO-HQ belongs in the first headline-safe package

This file only fixes the first fair compare.

## 10. Current Itachi Automation Boundary

The current local `itachi` Route-B automation remains the maintained geometry
evaluation path inside `3D-NEPA`.

For Route-A, the maintained pretrain engine is now `PCP-MAE`, not the CQA
trainer.

The local Route-A execution note lives in:

- `nepa3d/docs/patch_nepa/itachi/local_geopcp_pcpmae_ops_202604.md`
