# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_hardest.yaml`
- ckpt: `checkpoints/official/PointGPT-S/finetune_scan_hardest.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.8668`
- top2 hit: `0.9486`
- top5 hit: `0.9868`

## Hardest Pair

- pair: `bag (0) -> box (2)`
- off-diagonal count: `14`
- normalized confusion: `0.1687`
- pair direct top1 acc: `0.7639`
- `bag -> box`: `0.1687`
- `box -> bag`: `0.0226`
- mean logit margin (bag - box): `-3.4270`
- binary probe acc: `0.8750`
- binary probe bal acc: `0.8623`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.7349` |
| `bin` | `0.8693` |
| `box` | `0.7820` |
| `cabinet` | `0.8414` |
| `chair` | `0.9667` |
| `desk` | `0.8533` |
| `display` | `0.9363` |
| `door` | `0.9238` |
| `shelf` | `0.9004` |
| `table` | `0.7185` |
| `bed` | `0.8364` |
| `pillow` | `0.8095` |
| `sink` | `0.7917` |
| `sofa` | `0.9476` |
| `toilet` | `0.8824` |
