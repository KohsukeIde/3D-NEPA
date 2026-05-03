# PointGPT Unique-Retained Diagnostics Refresh

Snapshot: `2026-05-04 JST`

## Scope

This refresh reruns the PointGPT-side diagnostics that are relevant to the retained-support metric issue.

- ScanObjectNN classification is rerun for official PointGPT-S `obj_bg`, `obj_only`, and `PB_T50_RS` checkpoints.
- ShapeNetPart official PointGPT-S support stress is rerun with metrics computed on unique retained original points.
- ShapeNetPart official PointGPT-S eval-time grouping is rerun with the same unique-retained metric.
- Scene-side `concerto-shortcut-mvp` support code was audited, not rerun here. Its masking battery already separates `retained` scoring from `full_nn` full-scene propagation, so it does not have the ShapeNetPart fixed-size resampled-sequence mIoU issue.

These diagnostics are support/readout probes. They are not new PointGPT training runs.

## Result Files

- ScanObjectNN official support:
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_objbg_official_support.md`
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_objonly_official_support.md`
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_hardest_official_support.md`
- ScanObjectNN official readout:
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_objbg_official_readout.md`
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_objonly_official_readout.md`
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_hardest_official_readout.md`
- ScanObjectNN official eval-time grouping:
  - `3D-NEPA/results/pointgpt_unique_retained/scanobjectnn_objbg_official_grouping.md`
- ShapeNetPart official support:
  - `3D-NEPA/results/pointgpt_unique_retained/shapenetpart_official_support_unique.md`
- ShapeNetPart official eval-time grouping:
  - `3D-NEPA/results/pointgpt_unique_retained/shapenetpart_official_grouping_unique.md`

JSON/CSV companions are in the same directory.

## ScanObjectNN Readout

| split | top1 | top2 hit | top5 hit | hardest pair |
|---|---:|---:|---:|---|
| `obj_bg` | `0.9036` | `0.9690` | `0.9966` | `bag -> box` |
| `obj_only` | `0.9053` | `0.9656` | `0.9931` | `pillow -> bag` |
| `PB_T50_RS` | `0.8668` | `0.9486` | `0.9868` | `bag -> box` |

## ScanObjectNN Support

| split | clean | random80 | random50 | random20 | random10 | structured80 | structured50 | structured20 | structured10 | xyz_zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `obj_bg` | `0.9139` | `0.9088` | `0.8675` | `0.4234` | `0.2685` | `0.8726` | `0.7384` | `0.2203` | `0.1308` | `0.0929` |
| `obj_only` | `0.9036` | `0.8916` | `0.8778` | `0.5060` | `0.2788` | `0.8589` | `0.7418` | `0.2926` | `0.1377` | `0.0929` |
| `PB_T50_RS` | `0.8706` | `0.8636` | `0.8439` | `0.4119` | `0.2078` | `0.8324` | `0.6589` | `0.2044` | `0.0968` | `0.0708` |

## ShapeNetPart Support

Metrics are computed on unique retained original point indices. When retained support is resampled to the fixed input size, duplicated forward logits are averaged back to each retained original point before scoring.

| condition | inst mIoU | clean subset inst mIoU | damage inst mIoU | retained unique pts | repeated forward pts |
|---|---:|---:|---:|---:|---:|
| `clean` | `0.8188` | `0.8188` | `0.0000` | `2048.0` | `0.0` |
| `random_keep20` | `0.7174` | `0.8260` | `0.1086` | `410.0` | `1640.8` |
| `structured_keep20` | `0.6252` | `0.8660` | `0.2408` | `410.0` | `1640.8` |
| `part_drop_largest` | `0.4909` | `0.6158` | `0.1249` | `769.0` | `1366.3` |
| `part_keep20_per_part` | `0.7136` | `0.8255` | `0.1120` | `409.6` | `1641.1` |
| `xyz_zero` | `0.3398` | `0.8194` | `0.4796` | `2048.0` | `0.0` |

Full support rows include `80/50/20/10` for random, structured, local jitter, local replace, and per-part keep.

## ShapeNetPart Grouping

| group mode | clean inst mIoU | random20 inst mIoU | structured20 inst mIoU | part-drop inst mIoU | xyz-zero inst mIoU |
|---|---:|---:|---:|---:|---:|
| `fps_knn` | `0.8188` | `0.7158` | `0.6263` | `0.4878` | `0.3400` |
| `random_center_knn` | `0.7812` | `0.6860` | `0.6246` | `0.4808` | `0.3400` |
| `voxel_center_knn` | `0.8114` | `0.7207` | `0.6260` | `0.4854` | `0.3402` |
| `radius_fps` | `0.8181` | `0.7126` | `0.6262` | `0.4840` | `0.3401` |
| `random_group` | `0.4714` | `0.4624` | `0.5387` | `0.3612` | `0.3397` |

## Blockers

- PointGPT ShapeNetPart no-mask and train-time grouping checkpoints were not available locally. The local directories only contain `pt.py`, not checkpoint weights.
- Because those checkpoints are unavailable, this refresh reruns official-checkpoint support/grouping and records the missing rows as blocked rather than fabricating new no-mask or train-time grouping numbers.

## Commands

Main entry points:

```bash
PointGPT/tools/eval_scanobjectnn_support_stress.py
PointGPT/tools/eval_scanobjectnn_readout_audit.py
PointGPT/tools/eval_scanobjectnn_grouping_ablation.py
PointGPT/segmentation/eval_shapenetpart_support_stress.py
PointGPT/segmentation/eval_shapenetpart_grouping_ablation.py
```

Checkpoints used:

```text
PointGPT/checkpoints/official/PointGPT-S/finetune_scan_objbg.pth
PointGPT/checkpoints/official/PointGPT-S/finetune_scan_objonly.pth
PointGPT/checkpoints/official/PointGPT-S/finetune_scan_hardest.pth
PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth
```

Dataset roots:

```text
3D-NEPA/data/ScanObjectNN
3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal
```
