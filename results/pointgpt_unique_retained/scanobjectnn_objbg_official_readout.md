# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_objbg.yaml`
- ckpt: `checkpoints/official/PointGPT-S/finetune_scan_objbg.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.9036`
- top2 hit: `0.9690`
- top5 hit: `0.9966`

## Hardest Pair

- pair: `bag (0) -> box (2)`
- off-diagonal count: `3`
- normalized confusion: `0.1765`
- pair direct top1 acc: `0.8000`
- `bag -> box`: `0.1765`
- `box -> bag`: `0.0000`
- mean logit margin (bag - box): `-3.0848`
- binary probe acc: `0.9111`
- binary probe bal acc: `0.8939`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.7059` |
| `bin` | `0.9250` |
| `box` | `0.8571` |
| `cabinet` | `0.8267` |
| `chair` | `1.0000` |
| `desk` | `0.9333` |
| `display` | `0.9762` |
| `door` | `0.9286` |
| `shelf` | `0.9592` |
| `table` | `0.7778` |
| `bed` | `0.8636` |
| `pillow` | `0.8571` |
| `sink` | `0.8750` |
| `sofa` | `0.9762` |
| `toilet` | `0.9412` |
