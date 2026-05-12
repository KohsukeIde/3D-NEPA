# ShapeNetPart Support Stress

- ckpt: `PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth`
- root: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

| condition | accuracy | class avg IoU | instance avg IoU | clean subset inst IoU | damage inst IoU | retained unique pts | repeated forward pts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clean` | `0.9357` | `0.8038` | `0.8188` | `0.8188` | `0.0000` | `2048.0` | `0.0` |
| `random_keep80` | `0.7866` | `0.5615` | `0.6157` | `0.8194` | `0.2038` | `1638.0` | `878.6` |
| `random_keep50` | `0.8605` | `0.6473` | `0.6975` | `0.8221` | `0.1246` | `1024.0` | `1162.3` |
| `random_keep20` | `0.8860` | `0.6835` | `0.7174` | `0.8260` | `0.1086` | `410.0` | `1640.8` |
| `random_keep10` | `0.8171` | `0.5724` | `0.6237` | `0.8366` | `0.2129` | `205.0` | `1843.0` |
| `structured_keep80` | `0.7804` | `0.5527` | `0.6066` | `0.8224` | `0.2158` | `1638.0` | `879.1` |
| `structured_keep50` | `0.8416` | `0.6075` | `0.6564` | `0.8305` | `0.1740` | `1024.0` | `1162.6` |
| `structured_keep20` | `0.8381` | `0.6024` | `0.6252` | `0.8660` | `0.2408` | `410.0` | `1640.8` |
| `structured_keep10` | `0.8153` | `0.6643` | `0.6822` | `0.8946` | `0.2124` | `205.0` | `1843.0` |
| `local_jitter80` | `0.6480` | `0.4454` | `0.3951` | `0.8153` | `0.4202` | `2048.0` | `0.0` |
| `local_jitter50` | `0.7252` | `0.5013` | `0.4641` | `0.8214` | `0.3573` | `2048.0` | `0.0` |
| `local_jitter20` | `0.8345` | `0.5933` | `0.5787` | `0.8189` | `0.2402` | `2048.0` | `0.0` |
| `local_jitter10` | `0.8735` | `0.6563` | `0.6431` | `0.8179` | `0.1748` | `2048.0` | `0.0` |
| `local_replace80` | `0.4774` | `0.3322` | `0.2525` | `0.8183` | `0.5658` | `2048.0` | `0.0` |
| `local_replace50` | `0.5856` | `0.3988` | `0.3325` | `0.8194` | `0.4869` | `2048.0` | `0.0` |
| `local_replace20` | `0.7530` | `0.5211` | `0.4802` | `0.8180` | `0.3378` | `2048.0` | `0.0` |
| `local_replace10` | `0.8394` | `0.6056` | `0.5805` | `0.8176` | `0.2371` | `2048.0` | `0.0` |
| `part_drop_largest` | `0.7592` | `0.4055` | `0.4909` | `0.6158` | `0.1249` | `769.0` | `1366.3` |
| `part_keep80_per_part` | `0.7851` | `0.5611` | `0.6141` | `0.8161` | `0.2020` | `1638.4` | `878.9` |
| `part_keep50_per_part` | `0.8600` | `0.6477` | `0.6958` | `0.8231` | `0.1273` | `1024.0` | `1162.7` |
| `part_keep20_per_part` | `0.8846` | `0.6769` | `0.7136` | `0.8255` | `0.1120` | `409.6` | `1641.1` |
| `part_keep10_per_part` | `0.8185` | `0.5751` | `0.6245` | `0.8316` | `0.2071` | `204.8` | `1843.2` |
| `xyz_zero` | `0.5552` | `0.2931` | `0.3398` | `0.8194` | `0.4796` | `2048.0` | `0.0` |

## Notes

- Metrics are computed on unique retained original point indices.
- When retained support is resampled for the fixed-size forward pass, repeated logits are averaged back to the original retained point before scoring.
- `clean subset inst IoU` evaluates clean full-input predictions on the same retained point set; `damage inst IoU` is the matched retained-subset delta.
- `part_drop_largest` removes the largest ground-truth part within each object before fixed-size forward resampling.
- `part_keepXX_per_part` keeps XX% of each ground-truth part before fixed-size forward resampling. These are support-stress probes, not official ShapeNetPart scores.
