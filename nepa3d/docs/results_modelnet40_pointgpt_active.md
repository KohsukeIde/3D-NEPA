# ModelNet40 PointGPT-Style Protocol (Active)

Run root: `runs/modelnet40_pointgpt_protocol/`

Artifacts:
- raw: `results/modelnet40_pointgpt_protocol_raw.csv`
- summary: `results/modelnet40_pointgpt_protocol_summary.csv`

Completeness:
- full: `15/15`
- few-shot LP: `200/200`
- total: `215/215`

## Full Fine-tune (n=3 seeds)

| Method | n(seed) | test_acc mean +- std |
|---|---:|---:|
| `scratch` | 3 | 0.8589 +- 0.0042 |
| `shapenet_mesh_udf_nepa` | 3 | 0.8633 +- 0.0059 |
| `shapenet_mix_mae` | 3 | 0.8563 +- 0.0031 |
| `shapenet_mix_nepa` | 3 | 0.8587 +- 0.0040 |
| `shapenet_nepa` | 3 | 0.8598 +- 0.0064 |

## Few-shot Linear Probe (n=10 trials)

### N=5, K=10

| Method | n(trial) | test_acc mean +- std |
|---|---:|---:|
| `scratch` | 10 | 0.3342 +- 0.0419 |
| `shapenet_mesh_udf_nepa` | 10 | 0.4889 +- 0.1208 |
| `shapenet_mix_mae` | 10 | 0.5675 +- 0.1459 |
| `shapenet_mix_nepa` | 10 | 0.4682 +- 0.0971 |
| `shapenet_nepa` | 10 | 0.5893 +- 0.1759 |

### N=5, K=20

| Method | n(trial) | test_acc mean +- std |
|---|---:|---:|
| `scratch` | 10 | 0.3349 +- 0.0421 |
| `shapenet_mesh_udf_nepa` | 10 | 0.5232 +- 0.1247 |
| `shapenet_mix_mae` | 10 | 0.5899 +- 0.1525 |
| `shapenet_mix_nepa` | 10 | 0.4834 +- 0.1072 |
| `shapenet_nepa` | 10 | 0.6031 +- 0.1612 |

### N=10, K=10

| Method | n(trial) | test_acc mean +- std |
|---|---:|---:|
| `scratch` | 10 | 0.1654 +- 0.0229 |
| `shapenet_mesh_udf_nepa` | 10 | 0.3395 +- 0.0591 |
| `shapenet_mix_mae` | 10 | 0.3734 +- 0.1045 |
| `shapenet_mix_nepa` | 10 | 0.3304 +- 0.0797 |
| `shapenet_nepa` | 10 | 0.4091 +- 0.0667 |

### N=10, K=20

| Method | n(trial) | test_acc mean +- std |
|---|---:|---:|
| `scratch` | 10 | 0.1861 +- 0.0308 |
| `shapenet_mesh_udf_nepa` | 10 | 0.5031 +- 0.0554 |
| `shapenet_mix_mae` | 10 | 0.3823 +- 0.1159 |
| `shapenet_mix_nepa` | 10 | 0.4286 +- 0.0590 |
| `shapenet_nepa` | 10 | 0.5151 +- 0.0824 |

## Best-by-Setting

- Full: `shapenet_mesh_udf_nepa` (0.8633 +- 0.0059)
- N=5, K=10: `shapenet_nepa` (0.5893 +- 0.1759)
- N=5, K=20: `shapenet_nepa` (0.6031 +- 0.1612)
- N=10, K=10: `shapenet_nepa` (0.4091 +- 0.0667)
- N=10, K=20: `shapenet_nepa` (0.5151 +- 0.0824)
