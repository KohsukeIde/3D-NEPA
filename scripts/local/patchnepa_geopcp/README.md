# PCP-MAE Geo-PCP Local Launchers

Local wrappers in this directory treat:

- `PCP-MAE` as the Route-A / Hybrid pretrain engine
- `3D-NEPA` as the Route-B geometry-evaluation harness

Current `itachi` compare policy:

- pretrain budget: `100 epochs`
- recipe: `4xA100`, `total_bs=128`
- compare arms:
  - `pcp_worldvis_base_100ep`
  - `geopcp_worldvis_base_normal_100ep`
  - `geopcp_worldvis_base_normal_thickness_100ep`

Current downstream policy:

- `ScanObjectNN obj_bg`
- `ScanObjectNN obj_only`
- `ScanObjectNN hardest` (`pb_t50_rs` equivalent in the local ledger)
- `ShapeNetPart`

All wrappers assume this repo root:

- `/home/minesawa/ssl/3D-NEPA`

and the vendored PCP-MAE root:

- `/home/minesawa/ssl/3D-NEPA/PCP-MAE`

Compiled-first local Python on `itachi`:

- `/home/minesawa/anaconda3/envs/geopcp-pcpmae-cu118/bin/python`

Runtime policy:

- `pointnet2_ops` compiled backend is required
- `extensions/chamfer_dist` native backend is required
- slow fallback paths remain debug-only and must be explicitly opted in
- normal local wrappers fail fast if the compiled backend is not available

Local setup / launch order:

1. `scripts/local/patchnepa_geopcp/build_pcpmae_geopcp_env.sh`
2. `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_pretrain_local.sh`
3. `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_scanobjectnn_ft_local.sh`
4. `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_shapenetpart_ft_local.sh`
5. `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_routeb_local.sh`

Arm-complete orchestration:

- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh`
- `scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_full_chain_local.sh`
- `scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_compare_queue_local.sh`
- `scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_compare_queue_local.sh`

Resume downstream-only from an existing pretrain checkpoint:

```bash
PRETRAIN_SKIP=1 \
PRETRAIN_CKPT_OVERRIDE=/abs/path/to/ckpt-epoch-100.pth \
FT_VARIANTS=obj_only,hardest \
bash scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh
```

W&B defaults:

- pretrain: `patchnepa-geopcp-pretrain`
- ScanObjectNN: `patchnepa-geopcp-scanobjectnn`
- ShapeNetPart: `patchnepa-geopcp-shapenetpart`

Obsolete prototype:

- `/home/minesawa/ssl/geopcp_patch` is no longer the runtime source of truth
- Route-A implementation now lives inside `PCP-MAE/`
