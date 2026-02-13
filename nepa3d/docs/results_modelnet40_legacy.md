# ModelNet40 Legacy Results

This file summarizes the ModelNet40-era results that were previously mixed into `nepa3d/README.md`.

For full historical context and additional old notes, see:

- `nepa3d/docs/legacy_full_history.md`

## Scope

- Dataset focus: ModelNet40 (v0/v1 cache era)
- Goal at that time: backend transfer under shared query-token interface
- This is archived context, not the current main track

## v0 single-run summary

| Setting | Pretrain Backend | Finetune Backend | Final Acc (ep=99) | Best Acc (epoch) |
|---|---|---|---:|---:|
| Base | mesh | mesh | 0.8643 | 0.8643 (99) |
| Transfer A | mesh | pointcloud | 0.8468 | 0.8663 (73) |
| Transfer B | pointcloud | mesh | 0.8205 | 0.8278 (98) |

## v1 single-run summary (`EVAL_SEED=0`, `MC_EVAL_K=4`)

| Setting | Pretrain Backend | Finetune Backend | Final Acc (ep=99) | Best Acc (epoch) |
|---|---|---|---:|---:|
| Base | mesh | mesh | 0.8602 | 0.8756 (79) |
| Transfer A | mesh | pointcloud | 0.8679 | 0.8740 (89) |
| Transfer B | pointcloud | mesh | 0.8578 | 0.8728 (81) |
| Base-PC | pointcloud | pointcloud | 0.8562 | 0.8740 (98) |

## v1 multi-seed summary (`seed=0,1,2`)

### Pretrained initialization

| Setting | Final mean +- std | Best mean +- std |
|---|---:|---:|
| mesh -> mesh | 0.8594 +- 0.0057 | 0.8752 +- 0.0104 |
| mesh -> pointcloud | 0.8499 +- 0.0115 | 0.8749 +- 0.0074 |
| pointcloud -> mesh | 0.8635 +- 0.0114 | 0.8755 +- 0.0043 |
| pointcloud -> pointcloud | 0.8563 +- 0.0101 | 0.8718 +- 0.0033 |

### From-scratch baseline

| Setting | Final mean +- std | Best mean +- std |
|---|---:|---:|
| scratch mesh | 0.8498 +- 0.0025 | 0.8671 +- 0.0042 |
| scratch pointcloud | 0.8549 +- 0.0072 | 0.8705 +- 0.0040 |

## Legacy notes

- These tables are preserved for historical comparison only.
- Current main paper track has moved to ShapeNet/ScanObjectNN M1.
