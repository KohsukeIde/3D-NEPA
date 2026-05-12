# ScanObjectNN Grouping Ablation

- config: `cfgs/PointGPT-S/finetune_scan_objonly.yaml`
- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_objonly/PointGPT-S/pgpt_s_nomask_ordrand_objonly_e300_20260504/ckpt-best.pth`
- split: `test`
- radius: `0.22`
- voxel grid: `6`

| group mode | condition | acc |
|---|---|---:|
| `fps_knn` | `clean` | `0.8847` |
| `fps_knn` | `random_keep20` | `0.5198` |
| `fps_knn` | `structured_keep20` | `0.3270` |
| `fps_knn` | `xyz_zero` | `0.0929` |
| `random_center_knn` | `clean` | `0.8657` |
| `random_center_knn` | `random_keep20` | `0.5611` |
| `random_center_knn` | `structured_keep20` | `0.3012` |
| `random_center_knn` | `xyz_zero` | `0.0929` |
| `voxel_center_knn` | `clean` | `0.8709` |
| `voxel_center_knn` | `random_keep20` | `0.5749` |
| `voxel_center_knn` | `structured_keep20` | `0.3305` |
| `voxel_center_knn` | `xyz_zero` | `0.0929` |
| `radius_fps` | `clean` | `0.8830` |
| `radius_fps` | `random_keep20` | `0.5456` |
| `radius_fps` | `structured_keep20` | `0.3081` |
| `radius_fps` | `xyz_zero` | `0.0929` |
| `random_group` | `clean` | `0.1325` |
| `random_group` | `random_keep20` | `0.1480` |
| `random_group` | `structured_keep20` | `0.1308` |
| `random_group` | `xyz_zero` | `0.0929` |

## Notes

- `fps_knn` is the trained/default patchization.
- `random_center_knn` keeps local kNN neighborhoods but changes center selection.
- `voxel_center_knn` keeps local kNN neighborhoods but chooses grid-distributed centers.
- `radius_fps` keeps FPS centers but changes neighborhood construction to a radius query with nearest fallback.
- `random_group` destroys local neighborhoods and is a destructive architecture sanity check.
- These are inference-time grouping perturbations, not retrained architectures.
