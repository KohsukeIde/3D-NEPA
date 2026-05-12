# ShapeNetPart Grouping Ablation

- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_ordrand_e300_20260504/checkpoints/best_model.pth`
- root: `/home/minesawa/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

| group mode | condition | accuracy | class avg IoU | instance avg IoU | clean subset inst IoU | damage inst IoU | retained unique pts | repeated forward pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fps_knn` | `clean` | `0.9443` | `0.8281` | `0.8555` | `0.8555` | `0.0000` | `2048.0` | `0.0` |
| `fps_knn` | `random_keep20` | `0.8858` | `0.6847` | `0.7447` | `0.8636` | `0.1189` | `410.0` | `1640.7` |
| `fps_knn` | `structured_keep20` | `0.8407` | `0.6063` | `0.6642` | `0.8909` | `0.2267` | `410.0` | `1640.7` |
| `fps_knn` | `part_drop_largest` | `0.8020` | `0.4518` | `0.5450` | `0.6453` | `0.1003` | `768.8` | `1366.3` |
| `fps_knn` | `part_keep20_per_part` | `0.8867` | `0.6846` | `0.7416` | `0.8595` | `0.1179` | `409.6` | `1641.1` |
| `fps_knn` | `xyz_zero` | `0.4308` | `0.2261` | `0.2831` | `0.8568` | `0.5736` | `2048.0` | `0.0` |
| `random_center_knn` | `clean` | `0.9380` | `0.7980` | `0.8344` | `0.8344` | `0.0000` | `2048.0` | `0.0` |
| `random_center_knn` | `random_keep20` | `0.8692` | `0.6556` | `0.7201` | `0.8455` | `0.1253` | `410.0` | `1640.7` |
| `random_center_knn` | `structured_keep20` | `0.8261` | `0.5871` | `0.6580` | `0.8726` | `0.2146` | `410.0` | `1640.7` |
| `random_center_knn` | `part_drop_largest` | `0.7917` | `0.4531` | `0.5428` | `0.6248` | `0.0820` | `769.0` | `1366.4` |
| `random_center_knn` | `part_keep20_per_part` | `0.8677` | `0.6551` | `0.7138` | `0.8405` | `0.1267` | `409.6` | `1641.1` |
| `random_center_knn` | `xyz_zero` | `0.4303` | `0.2264` | `0.2835` | `0.8355` | `0.5520` | `2048.0` | `0.0` |
| `voxel_center_knn` | `clean` | `0.9414` | `0.8222` | `0.8497` | `0.8497` | `0.0000` | `2048.0` | `0.0` |
| `voxel_center_knn` | `random_keep20` | `0.8823` | `0.6736` | `0.7366` | `0.8581` | `0.1215` | `410.0` | `1640.7` |
| `voxel_center_knn` | `structured_keep20` | `0.8378` | `0.5896` | `0.6655` | `0.8838` | `0.2183` | `410.0` | `1640.7` |
| `voxel_center_knn` | `part_drop_largest` | `0.7990` | `0.4470` | `0.5402` | `0.6442` | `0.1039` | `769.0` | `1366.5` |
| `voxel_center_knn` | `part_keep20_per_part` | `0.8809` | `0.6748` | `0.7351` | `0.8516` | `0.1165` | `409.6` | `1641.1` |
| `voxel_center_knn` | `xyz_zero` | `0.4303` | `0.2258` | `0.2831` | `0.8488` | `0.5657` | `2048.0` | `0.0` |
| `radius_fps` | `clean` | `0.9441` | `0.8295` | `0.8552` | `0.8552` | `0.0000` | `2048.0` | `0.0` |
| `radius_fps` | `random_keep20` | `0.8847` | `0.6816` | `0.7436` | `0.8649` | `0.1213` | `410.0` | `1640.7` |
| `radius_fps` | `structured_keep20` | `0.8405` | `0.6036` | `0.6658` | `0.8885` | `0.2227` | `410.0` | `1640.7` |
| `radius_fps` | `part_drop_largest` | `0.8005` | `0.4545` | `0.5458` | `0.6465` | `0.1007` | `768.8` | `1366.6` |
| `radius_fps` | `part_keep20_per_part` | `0.8833` | `0.6794` | `0.7377` | `0.8574` | `0.1197` | `409.6` | `1641.1` |
| `radius_fps` | `xyz_zero` | `0.4303` | `0.2263` | `0.2838` | `0.8575` | `0.5737` | `2048.0` | `0.0` |
| `random_group` | `clean` | `0.7271` | `0.4650` | `0.5342` | `0.5342` | `0.0000` | `2048.0` | `0.0` |
| `random_group` | `random_keep20` | `0.7262` | `0.4528` | `0.5277` | `0.5418` | `0.0141` | `410.0` | `1640.7` |
| `random_group` | `structured_keep20` | `0.7734` | `0.5345` | `0.5972` | `0.6287` | `0.0315` | `410.0` | `1640.7` |
| `random_group` | `part_drop_largest` | `0.5892` | `0.3222` | `0.4344` | `0.4370` | `0.0026` | `768.8` | `1366.8` |
| `random_group` | `part_keep20_per_part` | `0.7263` | `0.4536` | `0.5264` | `0.5408` | `0.0144` | `409.6` | `1641.1` |
| `random_group` | `xyz_zero` | `0.4303` | `0.2258` | `0.2828` | `0.5362` | `0.2534` | `2048.0` | `0.0` |

## Notes

- The checkpoint/head are fixed. Only grouping center/neighborhood construction is changed at inference time.
- ShapeNetPart support metrics are computed on unique retained original point indices; fixed-size forward resampling is aggregated back by original point.
- `random_group` destroys local neighborhoods and is a destructive sanity check.
- This is a diagnostic ablation, not a retrained architecture comparison.
