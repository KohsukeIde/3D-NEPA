# PointGPT-S No-Mask Order-Randomized Diagnostics

Purpose: rerun the PointGPT no-mask + order-randomized object diagnostics with the current local code and unique-retained ShapeNetPart scoring.

## Checkpoints

- pretrain: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300_20260504/ckpt-last.pth`
- ShapeNetPart FT: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_ordrand_e300_20260504/checkpoints/best_model.pth`

## Result Files

- ScanObjectNN `obj_bg` readout: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_bg_nomask_ordrand_readout.md`
- ScanObjectNN `obj_bg` support: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_bg_nomask_ordrand_support.md`
- ScanObjectNN `obj_bg` eval-time grouping: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_bg_nomask_ordrand_grouping.md`
- ScanObjectNN `obj_only` readout: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_only_nomask_ordrand_readout.md`
- ScanObjectNN `obj_only` support: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_only_nomask_ordrand_support.md`
- ScanObjectNN `obj_only` eval-time grouping: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_obj_only_nomask_ordrand_grouping.md`
- ScanObjectNN `pb_t50_rs` readout: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_pb_t50_rs_nomask_ordrand_readout.md`
- ScanObjectNN `pb_t50_rs` support: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_pb_t50_rs_nomask_ordrand_support.md`
- ScanObjectNN `pb_t50_rs` eval-time grouping: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/scanobjectnn_pb_t50_rs_nomask_ordrand_grouping.md`
- ShapeNetPart support: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/shapenetpart_nomask_ordrand_support_unique.md`
- ShapeNetPart eval-time grouping: `3D-NEPA/results/pointgpt_nomask_orderrandom_full/shapenetpart_nomask_ordrand_grouping_unique.md`

## ScanObjectNN Readout

| split | top1 | top2 hit | top5 hit | hardest pair |
|---|---:|---:|---:|---|
| `obj_bg` | `0.9053` | `0.9639` | `0.9931` | `bag -> box` |
| `obj_only` | `0.8830` | `0.9398` | `0.9845` | `sink -> table` |
| `pb_t50_rs` | `0.8397` | `0.9310` | `0.9847` | `bed -> sofa` |

## ScanObjectNN Support

| split | clean | random80 | random50 | random20 | random10 | structured80 | structured50 | structured20 | structured10 | xyz_zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `obj_bg` | `0.9002` | `0.8950` | `0.8950` | `0.5697` | `0.1532` | `0.8709` | `0.7194` | `0.2496` | `0.1188` | `0.0929` |
| `obj_only` | `0.8881` | `0.8812` | `0.8675` | `0.5577` | `0.1807` | `0.8503` | `0.7470` | `0.3322` | `0.1532` | `0.0929` |
| `pb_t50_rs` | `0.8334` | `0.8393` | `0.8116` | `0.3765` | `0.1405` | `0.8102` | `0.6724` | `0.2401` | `0.1374` | `0.0708` |

## Notes

- ShapeNetPart support metrics use unique retained original point indices; fixed-size forward resampling is aggregated back by original point.
- ScanObjectNN rows use the PointGPT single-label classification support/readout protocol; random/structured keep conditions include 80/50/20/10.
- Grouping rows are eval-time patchization perturbations with checkpoint/readout fixed.
