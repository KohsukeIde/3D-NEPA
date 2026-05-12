# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_hardest.yaml`
- ckpt: `/mnt/urashima/users/minesawa/home-offload/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_hardest/PointGPT-S/pgpt_s_nomask_ordrand_hardest_e300_20260504/ckpt-best.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.8397`
- top2 hit: `0.9310`
- top5 hit: `0.9847`

## Hardest Pair

- pair: `bed (10) -> sofa (13)`
- off-diagonal count: `17`
- normalized confusion: `0.1545`
- pair direct top1 acc: `0.8844`
- `bed -> sofa`: `0.1545`
- `sofa -> bed`: `0.0190`
- mean logit margin (bed - sofa): `-4.3327`
- binary probe acc: `0.9250`
- binary probe bal acc: `0.8996`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.6867` |
| `bin` | `0.8492` |
| `box` | `0.6842` |
| `cabinet` | `0.8172` |
| `chair` | `0.9462` |
| `desk` | `0.8200` |
| `display` | `0.8922` |
| `door` | `0.9190` |
| `shelf` | `0.8423` |
| `table` | `0.7333` |
| `bed` | `0.8182` |
| `pillow` | `0.7714` |
| `sink` | `0.7667` |
| `sofa` | `0.9190` |
| `toilet` | `0.8824` |
