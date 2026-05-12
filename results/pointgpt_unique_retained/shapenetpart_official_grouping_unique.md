# ShapeNetPart Grouping Ablation

- ckpt: `PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth`
- root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

| group mode | condition | accuracy | class avg IoU | instance avg IoU | clean subset inst IoU | damage inst IoU | retained unique pts | repeated forward pts |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `fps_knn` | `clean` | `0.9357` | `0.8038` | `0.8188` | `0.8188` | `0.0000` | `2048.0` | `0.0` |
| `fps_knn` | `random_keep20` | `0.8844` | `0.6785` | `0.7158` | `0.8289` | `0.1131` | `410.0` | `1640.8` |
| `fps_knn` | `structured_keep20` | `0.8380` | `0.6034` | `0.6263` | `0.8637` | `0.2374` | `410.0` | `1640.8` |
| `fps_knn` | `part_drop_largest` | `0.7615` | `0.4044` | `0.4878` | `0.6119` | `0.1241` | `768.6` | `1366.6` |
| `fps_knn` | `part_keep20_per_part` | `0.8861` | `0.6833` | `0.7162` | `0.8252` | `0.1090` | `409.6` | `1641.1` |
| `fps_knn` | `xyz_zero` | `0.5552` | `0.2934` | `0.3400` | `0.8179` | `0.4779` | `2048.0` | `0.0` |
| `random_center_knn` | `clean` | `0.9215` | `0.7547` | `0.7812` | `0.7812` | `0.0000` | `2048.0` | `0.0` |
| `random_center_knn` | `random_keep20` | `0.8649` | `0.6461` | `0.6860` | `0.7916` | `0.1057` | `410.0` | `1640.8` |
| `random_center_knn` | `structured_keep20` | `0.8265` | `0.5919` | `0.6246` | `0.8431` | `0.2185` | `410.0` | `1640.8` |
| `random_center_knn` | `part_drop_largest` | `0.7474` | `0.3986` | `0.4808` | `0.5892` | `0.1084` | `768.2` | `1367.0` |
| `random_center_knn` | `part_keep20_per_part` | `0.8637` | `0.6463` | `0.6830` | `0.7888` | `0.1058` | `409.6` | `1641.1` |
| `random_center_knn` | `xyz_zero` | `0.5552` | `0.2931` | `0.3400` | `0.7842` | `0.4442` | `2048.0` | `0.0` |
| `voxel_center_knn` | `clean` | `0.9287` | `0.7823` | `0.8114` | `0.8114` | `0.0000` | `2048.0` | `0.0` |
| `voxel_center_knn` | `random_keep20` | `0.8816` | `0.6710` | `0.7207` | `0.8232` | `0.1025` | `410.0` | `1640.8` |
| `voxel_center_knn` | `structured_keep20` | `0.8344` | `0.5916` | `0.6260` | `0.8609` | `0.2348` | `410.0` | `1640.8` |
| `voxel_center_knn` | `part_drop_largest` | `0.7615` | `0.3981` | `0.4854` | `0.6107` | `0.1252` | `768.4` | `1366.6` |
| `voxel_center_knn` | `part_keep20_per_part` | `0.8831` | `0.6728` | `0.7171` | `0.8161` | `0.0990` | `409.6` | `1641.2` |
| `voxel_center_knn` | `xyz_zero` | `0.5549` | `0.2934` | `0.3402` | `0.8138` | `0.4736` | `2048.0` | `0.0` |
| `radius_fps` | `clean` | `0.9339` | `0.7983` | `0.8181` | `0.8181` | `0.0000` | `2048.0` | `0.0` |
| `radius_fps` | `random_keep20` | `0.8824` | `0.6745` | `0.7126` | `0.8296` | `0.1170` | `410.0` | `1640.8` |
| `radius_fps` | `structured_keep20` | `0.8380` | `0.6104` | `0.6262` | `0.8644` | `0.2382` | `410.0` | `1640.8` |
| `radius_fps` | `part_drop_largest` | `0.7611` | `0.3946` | `0.4840` | `0.6165` | `0.1325` | `768.7` | `1366.6` |
| `radius_fps` | `part_keep20_per_part` | `0.8812` | `0.6749` | `0.7110` | `0.8253` | `0.1143` | `409.6` | `1641.1` |
| `radius_fps` | `xyz_zero` | `0.5551` | `0.2932` | `0.3401` | `0.8191` | `0.4790` | `2048.0` | `0.0` |
| `random_group` | `clean` | `0.6603` | `0.4394` | `0.4714` | `0.4714` | `0.0000` | `2048.0` | `0.0` |
| `random_group` | `random_keep20` | `0.6607` | `0.4287` | `0.4624` | `0.4754` | `0.0130` | `410.0` | `1640.8` |
| `random_group` | `structured_keep20` | `0.7120` | `0.5108` | `0.5387` | `0.5938` | `0.0551` | `410.0` | `1640.8` |
| `random_group` | `part_drop_largest` | `0.5476` | `0.2346` | `0.3612` | `0.4136` | `0.0524` | `769.0` | `1366.7` |
| `random_group` | `part_keep20_per_part` | `0.6600` | `0.4233` | `0.4587` | `0.4756` | `0.0169` | `409.6` | `1641.1` |
| `random_group` | `xyz_zero` | `0.5552` | `0.2934` | `0.3397` | `0.4723` | `0.1326` | `2048.0` | `0.0` |

## Notes

- The checkpoint/head are fixed. Only grouping center/neighborhood construction is changed at inference time.
- ShapeNetPart support metrics are computed on unique retained original point indices; fixed-size forward resampling is aggregated back by original point.
- `random_group` destroys local neighborhoods and is a destructive sanity check.
- This is a diagnostic ablation, not a retrained architecture comparison.
