# ScanObjectNN Grouping Ablation

- config: `cfgs/PointGPT-S/finetune_scan_objbg.yaml`
- ckpt: `checkpoints/official/PointGPT-S/finetune_scan_objbg.pth`
- split: `test`
- radius: `0.22`
- voxel grid: `6`

| group mode | condition | acc |
|---|---|---:|
| `fps_knn` | `clean` | `0.9002` |
| `fps_knn` | `random_keep20` | `0.4303` |
| `fps_knn` | `structured_keep20` | `0.2478` |
| `fps_knn` | `xyz_zero` | `0.0929` |
| `random_center_knn` | `clean` | `0.8812` |
| `random_center_knn` | `random_keep20` | `0.3941` |
| `random_center_knn` | `structured_keep20` | `0.1876` |
| `random_center_knn` | `xyz_zero` | `0.0929` |
| `voxel_center_knn` | `clean` | `0.8881` |
| `voxel_center_knn` | `random_keep20` | `0.4372` |
| `voxel_center_knn` | `structured_keep20` | `0.2100` |
| `voxel_center_knn` | `xyz_zero` | `0.0929` |
| `radius_fps` | `clean` | `0.9071` |
| `radius_fps` | `random_keep20` | `0.4544` |
| `radius_fps` | `structured_keep20` | `0.2220` |
| `radius_fps` | `xyz_zero` | `0.0929` |
| `random_group` | `clean` | `0.0740` |
| `random_group` | `random_keep20` | `0.0826` |
| `random_group` | `structured_keep20` | `0.0706` |
| `random_group` | `xyz_zero` | `0.0929` |

## Notes

- `fps_knn` is the trained/default patchization.
- `random_center_knn` keeps local kNN neighborhoods but changes center selection.
- `voxel_center_knn` keeps local kNN neighborhoods but chooses grid-distributed centers.
- `radius_fps` keeps FPS centers but changes neighborhood construction to a radius query with nearest fallback.
- `random_group` destroys local neighborhoods and is a destructive architecture sanity check.
- These are inference-time grouping perturbations, not retrained architectures.
