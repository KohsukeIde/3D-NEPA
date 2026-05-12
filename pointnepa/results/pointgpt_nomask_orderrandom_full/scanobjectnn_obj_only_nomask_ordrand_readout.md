# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_objonly.yaml`
- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_objonly/PointGPT-S/pgpt_s_nomask_ordrand_objonly_e300_20260504/ckpt-best.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.8830`
- top2 hit: `0.9398`
- top5 hit: `0.9845`

## Hardest Pair

- pair: `sink (12) -> table (9)`
- off-diagonal count: `3`
- normalized confusion: `0.1250`
- pair direct top1 acc: `0.7949`
- `sink -> table`: `0.1250`
- `table -> sink`: `0.0185`
- mean logit margin (sink - table): `-4.7429`
- binary probe acc: `0.9103`
- binary probe bal acc: `0.8657`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.7059` |
| `bin` | `0.9750` |
| `box` | `0.8214` |
| `cabinet` | `0.8400` |
| `chair` | `1.0000` |
| `desk` | `0.9000` |
| `display` | `0.8571` |
| `door` | `0.9286` |
| `shelf` | `0.8776` |
| `table` | `0.8519` |
| `bed` | `0.8636` |
| `pillow` | `0.8571` |
| `sink` | `0.6667` |
| `sofa` | `0.9524` |
| `toilet` | `0.8235` |
