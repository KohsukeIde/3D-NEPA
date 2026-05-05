# ScanObjectNN Grouping Ablation

- config: `cfgs/PointGPT-S/finetune_scan_hardest.yaml`
- ckpt: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_hardest/PointGPT-S/pgpt_s_nomask_ordrand_hardest_e300_20260504/ckpt-best.pth`
- split: `test`
- radius: `0.22`
- voxel grid: `6`

| group mode | condition | acc |
|---|---|---:|
| `fps_knn` | `clean` | `0.8373` |
| `fps_knn` | `random_keep20` | `0.3855` |
| `fps_knn` | `structured_keep20` | `0.2405` |
| `fps_knn` | `xyz_zero` | `0.0708` |
| `random_center_knn` | `clean` | `0.8112` |
| `random_center_knn` | `random_keep20` | `0.3806` |
| `random_center_knn` | `structured_keep20` | `0.2196` |
| `random_center_knn` | `xyz_zero` | `0.0708` |
| `voxel_center_knn` | `clean` | `0.8289` |
| `voxel_center_knn` | `random_keep20` | `0.3956` |
| `voxel_center_knn` | `structured_keep20` | `0.2339` |
| `voxel_center_knn` | `xyz_zero` | `0.0708` |
| `radius_fps` | `clean` | `0.8432` |
| `radius_fps` | `random_keep20` | `0.3803` |
| `radius_fps` | `structured_keep20` | `0.2356` |
| `radius_fps` | `xyz_zero` | `0.0708` |
| `random_group` | `clean` | `0.1253` |
| `random_group` | `random_keep20` | `0.1214` |
| `random_group` | `structured_keep20` | `0.1069` |
| `random_group` | `xyz_zero` | `0.0708` |

## Notes

- `fps_knn` is the trained/default patchization.
- `random_center_knn` keeps local kNN neighborhoods but changes center selection.
- `voxel_center_knn` keeps local kNN neighborhoods but chooses grid-distributed centers.
- `radius_fps` keeps FPS centers but changes neighborhood construction to a radius query with nearest fallback.
- `random_group` destroys local neighborhoods and is a destructive architecture sanity check.
- These are inference-time grouping perturbations, not retrained architectures.
