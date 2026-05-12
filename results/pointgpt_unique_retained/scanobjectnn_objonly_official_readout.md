# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_objonly.yaml`
- ckpt: `checkpoints/official/PointGPT-S/finetune_scan_objonly.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.9053`
- top2 hit: `0.9656`
- top5 hit: `0.9931`

## Hardest Pair

- pair: `pillow (11) -> bag (0)`
- off-diagonal count: `3`
- normalized confusion: `0.1429`
- pair direct top1 acc: `0.8947`
- `pillow -> bag`: `0.1429`
- `bag -> pillow`: `0.0000`
- mean logit margin (pillow - bag): `0.0776`
- binary probe acc: `0.9211`
- binary probe bal acc: `0.9286`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.9412` |
| `bin` | `0.9500` |
| `box` | `0.8571` |
| `cabinet` | `0.8667` |
| `chair` | `1.0000` |
| `desk` | `0.8667` |
| `display` | `0.9286` |
| `door` | `0.9286` |
| `shelf` | `0.8980` |
| `table` | `0.8704` |
| `bed` | `0.9091` |
| `pillow` | `0.8571` |
| `sink` | `0.7500` |
| `sofa` | `0.9524` |
| `toilet` | `0.8235` |
