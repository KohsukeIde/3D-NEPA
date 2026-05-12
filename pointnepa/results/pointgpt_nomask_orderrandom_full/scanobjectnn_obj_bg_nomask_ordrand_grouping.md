# ScanObjectNN Grouping Ablation

- config: `cfgs/PointGPT-S/finetune_scan_objbg.yaml`
- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300_20260504/ckpt-best.pth`
- split: `test`
- radius: `0.22`
- voxel grid: `6`

| group mode | condition | acc |
|---|---|---:|
| `fps_knn` | `clean` | `0.8950` |
| `fps_knn` | `random_keep20` | `0.5731` |
| `fps_knn` | `structured_keep20` | `0.2169` |
| `fps_knn` | `xyz_zero` | `0.0929` |
| `random_center_knn` | `clean` | `0.8606` |
| `random_center_knn` | `random_keep20` | `0.5129` |
| `random_center_knn` | `structured_keep20` | `0.2083` |
| `random_center_knn` | `xyz_zero` | `0.0929` |
| `voxel_center_knn` | `clean` | `0.8761` |
| `voxel_center_knn` | `random_keep20` | `0.5972` |
| `voxel_center_knn` | `structured_keep20` | `0.2117` |
| `voxel_center_knn` | `xyz_zero` | `0.0929` |
| `radius_fps` | `clean` | `0.8933` |
| `radius_fps` | `random_keep20` | `0.5577` |
| `radius_fps` | `structured_keep20` | `0.2410` |
| `radius_fps` | `xyz_zero` | `0.0929` |
| `random_group` | `clean` | `0.0654` |
| `random_group` | `random_keep20` | `0.0757` |
| `random_group` | `structured_keep20` | `0.0947` |
| `random_group` | `xyz_zero` | `0.0929` |

## Notes

- `fps_knn` is the trained/default patchization.
- `random_center_knn` keeps local kNN neighborhoods but changes center selection.
- `voxel_center_knn` keeps local kNN neighborhoods but chooses grid-distributed centers.
- `radius_fps` keeps FPS centers but changes neighborhood construction to a radius query with nearest fallback.
- `random_group` destroys local neighborhoods and is a destructive architecture sanity check.
- These are inference-time grouping perturbations, not retrained architectures.
