# ScanObjectNN Readout Audit

- config: `cfgs/PointGPT-S/finetune_scan_objbg.yaml`
- ckpt: `/home/minesawa/ssl/3D-NEPA/PointGPT/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300_20260504/ckpt-best.pth`
- train split: `train`
- test split: `test`

## Global

- top1 acc: `0.9053`
- top2 hit: `0.9639`
- top5 hit: `0.9931`

## Hardest Pair

- pair: `bag (0) -> box (2)`
- off-diagonal count: `3`
- normalized confusion: `0.1765`
- pair direct top1 acc: `0.8222`
- `bag -> box`: `0.1765`
- `box -> bag`: `0.0714`
- mean logit margin (bag - box): `-1.9980`
- binary probe acc: `0.8667`
- binary probe bal acc: `0.8466`

## Per-Class Acc

| class | acc |
|---|---:|
| `bag` | `0.8235` |
| `bin` | `0.9000` |
| `box` | `0.8214` |
| `cabinet` | `0.9200` |
| `chair` | `0.9615` |
| `desk` | `0.8333` |
| `display` | `0.9524` |
| `door` | `0.9048` |
| `shelf` | `0.9184` |
| `table` | `0.8519` |
| `bed` | `0.9091` |
| `pillow` | `0.8571` |
| `sink` | `0.9167` |
| `sofa` | `0.9524` |
| `toilet` | `0.8824` |
