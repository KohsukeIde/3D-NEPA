# ShapeNetPart Support Stress

- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_ordrand_e300_20260504/checkpoints/best_model.pth`
- root: `/home/minesawa/ssl/3D-NEPA/data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

| condition | accuracy | class avg IoU | instance avg IoU | clean subset inst IoU | damage inst IoU | retained unique pts | repeated forward pts |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clean` | `0.9443` | `0.8281` | `0.8555` | `0.8555` | `0.0000` | `2048.0` | `0.0` |
| `random_keep80` | `0.7933` | `0.5802` | `0.6471` | `0.8567` | `0.2096` | `1638.0` | `879.1` |
| `random_keep50` | `0.8699` | `0.6681` | `0.7324` | `0.8590` | `0.1265` | `1024.0` | `1162.4` |
| `random_keep20` | `0.8866` | `0.6850` | `0.7461` | `0.8640` | `0.1179` | `410.0` | `1640.7` |
| `random_keep10` | `0.7874` | `0.5454` | `0.6139` | `0.8683` | `0.2544` | `205.0` | `1843.0` |
| `structured_keep80` | `0.7903` | `0.5799` | `0.6428` | `0.8592` | `0.2164` | `1638.0` | `879.1` |
| `structured_keep50` | `0.8573` | `0.6430` | `0.7023` | `0.8647` | `0.1624` | `1024.0` | `1162.6` |
| `structured_keep20` | `0.8381` | `0.6040` | `0.6664` | `0.8896` | `0.2231` | `410.0` | `1640.7` |
| `structured_keep10` | `0.8203` | `0.6154` | `0.7035` | `0.9103` | `0.2068` | `205.0` | `1843.0` |
| `local_jitter80` | `0.7110` | `0.4641` | `0.4259` | `0.8550` | `0.4290` | `2048.0` | `0.0` |
| `local_jitter50` | `0.7822` | `0.5221` | `0.5126` | `0.8562` | `0.3436` | `2048.0` | `0.0` |
| `local_jitter20` | `0.8573` | `0.6224` | `0.6349` | `0.8551` | `0.2202` | `2048.0` | `0.0` |
| `local_jitter10` | `0.8925` | `0.6823` | `0.7025` | `0.8567` | `0.1541` | `2048.0` | `0.0` |
| `local_replace80` | `0.4726` | `0.3303` | `0.2611` | `0.8549` | `0.5937` | `2048.0` | `0.0` |
| `local_replace50` | `0.5986` | `0.3983` | `0.3440` | `0.8555` | `0.5115` | `2048.0` | `0.0` |
| `local_replace20` | `0.7696` | `0.5251` | `0.4982` | `0.8566` | `0.3583` | `2048.0` | `0.0` |
| `local_replace10` | `0.8480` | `0.6225` | `0.6135` | `0.8556` | `0.2421` | `2048.0` | `0.0` |
| `part_drop_largest` | `0.8001` | `0.4496` | `0.5442` | `0.6471` | `0.1029` | `768.8` | `1366.5` |
| `part_keep80_per_part` | `0.7933` | `0.5816` | `0.6456` | `0.8568` | `0.2112` | `1638.4` | `878.5` |
| `part_keep50_per_part` | `0.8694` | `0.6685` | `0.7306` | `0.8572` | `0.1266` | `1024.0` | `1162.3` |
| `part_keep20_per_part` | `0.8861` | `0.6835` | `0.7421` | `0.8593` | `0.1173` | `409.6` | `1641.1` |
| `part_keep10_per_part` | `0.7879` | `0.5387` | `0.6082` | `0.8612` | `0.2530` | `204.8` | `1843.2` |
| `xyz_zero` | `0.4304` | `0.2259` | `0.2822` | `0.8557` | `0.5735` | `2048.0` | `0.0` |

## Notes

- Metrics are computed on unique retained original point indices.
- When retained support is resampled for the fixed-size forward pass, repeated logits are averaged back to the original retained point before scoring.
- `clean subset inst IoU` evaluates clean full-input predictions on the same retained point set; `damage inst IoU` is the matched retained-subset delta.
- `part_drop_largest` removes the largest ground-truth part within each object before fixed-size forward resampling.
- `part_keepXX_per_part` keeps XX% of each ground-truth part before fixed-size forward resampling. These are support-stress probes, not official ShapeNetPart scores.
