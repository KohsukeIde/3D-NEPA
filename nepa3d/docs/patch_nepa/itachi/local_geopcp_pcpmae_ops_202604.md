# Itachi Route-A Geo-PCP Ops (PCP-MAE Engine)

Last updated: 2026-04-05

This note records the local-only execution boundary for the PCP-MAE-native
Geo-PCP Route-A branch on `itachi`.

## Engine split

- Route-A / Hybrid pretrain engine: `PCP-MAE`
- Route-B geometry evaluation harness: `3D-NEPA`

Do not treat the old external `/home/minesawa/ssl/geopcp_patch` bundle as the
runtime source of truth. The maintained implementation now lives inside:

- `PCP-MAE/`

## Canonical local data roots

- Geo-PCP pretrain source:
  - `data/shapenet_cache_v2_20260401_worldvis/train`
- ScanObjectNN raw H5:
  - `data/ScanObjectNN/h5_files/main_split`
  - `data/ScanObjectNN/h5_files/main_split_nobg`
- ShapeNetPart:
  - `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

Artifact roots are symlinked off `/home`:

- `PCP-MAE/experiments -> /mnt/urashima/users/minesawa/3D-NEPA-data/repo_artifacts/pcpmae_experiments`
- `PCP-MAE/wandb -> /mnt/urashima/users/minesawa/3D-NEPA-data/repo_artifacts/pcpmae_wandb`
- `PCP-MAE/log -> /mnt/urashima/users/minesawa/3D-NEPA-data/repo_artifacts/pcpmae_log`
- `runs -> /mnt/urashima/users/minesawa/3D-NEPA-data/repo_artifacts/repo_runs`

## Current compare matrix

- `pcp_worldvis_base_100ep`
- `geopcp_worldvis_base_normal_100ep`
- `geopcp_worldvis_base_normal_thickness_100ep`

All three use:

- `100` pretrain epochs
- `4xA100`
- `total_bs=128`

## Current local runtime note

Recommended Python:

- `/home/minesawa/anaconda3/envs/geopcp-pcpmae-cu118/bin/python`

Compiled-first policy:

- `pointnet2_ops` compiled backend is required for normal Route-A runs
- `extensions/chamfer_dist` native backend is required for normal Route-A runs
- debug fallback is allowed only by explicit opt-in

Build entrypoint:

- `scripts/local/patchnepa_geopcp/build_pcpmae_geopcp_env.sh`

Expected build variables:

- `CUDA_HOME=$CONDA_PREFIX`
- `CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc`
- `CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++`
- `TORCH_CUDA_ARCH_LIST=8.0`

## Downstream mapping

- `obj_bg` -> PCP-MAE `ScanObjectNN_objectbg`
- `obj_only` -> PCP-MAE `ScanObjectNN_objectonly`
- `hardest` -> local ledger `pb_t50_rs` equivalent
- `ShapeNetPart` -> PCP-MAE segmentation flow

## Local wrappers

- `scripts/local/patchnepa_geopcp/build_pcpmae_geopcp_env.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_pretrain_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_ablation_triple_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_compare_queue_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_scanobjectnn_ft_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_shapenetpart_ft_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_routeb_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh`
- `scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_pretrain_local.sh`
- `scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_full_chain_local.sh`
- `scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_compare_queue_local.sh`

W&B defaults:

- pretrain: `patchnepa-geopcp-pretrain`
- ScanObjectNN: `patchnepa-geopcp-scanobjectnn`
- ShapeNetPart: `patchnepa-geopcp-shapenetpart`

Resume downstream-only from an existing Route-A pretrain checkpoint:

```bash
PRETRAIN_SKIP=1 \
PRETRAIN_CKPT_OVERRIDE=/home/minesawa/ssl/3D-NEPA/PCP-MAE/experiments/pcp_worldvis_base_100ep/geopcp/pcp_worldvis_base_100ep/ckpt-epoch-100.pth \
FT_VARIANTS=obj_only,hardest \
bash scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh
```
