# NEPA-3D Active Track (M1)

This file tracks only the current active experiment track.

- Active track: ShapeNet/ScanObjectNN M1
- As-of snapshot date: February 13, 2026

Legacy ModelNet40-era notes/results were moved out of this file.
See:

- `nepa3d/docs/results_index.md`
- `nepa3d/docs/results_modelnet40_legacy.md`
- `nepa3d/docs/legacy_full_history.md`

## 1) Current M1 definition

Pretrain datasets:

- ShapeNet mesh
- ShapeNet UDF (`udfgrid`)
- ScanObjectNN pointcloud no-ray

Downstream:

- ScanObjectNN classification
- full + few-shot: `K=0,1,5,10,20`
- seeds: `0,1,2`

Methods in table:

- `scratch`
- `shapenet_nepa` (mesh-only pretrain)
- `shapenet_mesh_udf_nepa`
- `shapenet_mix_nepa`
- `shapenet_mix_mae`

## 2) Launch commands

M1 pretrain (3 jobs on 2 GPUs):

```bash
bash scripts/pretrain/launch_shapenet_m1_pretrains_local.sh
```

M1 fine-tune table:

```bash
bash scripts/finetune/launch_scanobjectnn_m1_table_local.sh
```

Optional chain (start fine-tune after pretrain success):

```bash
bash scripts/finetune/launch_scanobjectnn_m1_after_pretrain.sh
```

## 3) Resume behavior

Pretrain:

- `pretrain.py` supports `--resume` and `--auto_resume`.
- Launchers pass `--resume <save_dir>/last.pt`.
- Interrupted pretrain resumes from `last.pt` if present.

Fine-tune table:

- Each fine-tune job is skipped when `runs/scan_<method>_k<K>_s<seed>/last.pt` exists.
- Relaunch command resumes remaining jobs only.

## 4) Logs and outputs

- pretrain logs: `logs/pretrain/m1/`
- fine-tune logs: `logs/finetune/scan_m1_table/`
- fine-tune job logs: `logs/finetune/scan_m1_table/jobs/*.log`
- outputs: `runs/scan_<method>_k<K>_s<seed>/`

Helper scripts:

```bash
bash scripts/logs/show_pipeline_status.sh
bash scripts/logs/cleanup_stale_pids.sh
```

## 5) Current result snapshot (partial)

Status (as of February 13, 2026):

- completed jobs: `30 / 75`
- completion by method:
  - `scratch`: `10/15`
  - `shapenet_nepa`: `10/15`
  - `shapenet_mesh_udf_nepa`: `3/15`
  - `shapenet_mix_nepa`: `4/15`
  - `shapenet_mix_mae`: `3/15`

Table below is computed from completed `runs/*/last.pt` only.
`n(seed)` is the number of finished seeds for each `(method, K)`.

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 2 | 0.8221 +- 0.0038 |
| `scratch` | 1 | 2 | 0.1414 +- 0.0211 |
| `scratch` | 5 | 2 | 0.1456 +- 0.0033 |
| `scratch` | 10 | 2 | 0.1115 +- 0.0000 |
| `scratch` | 20 | 2 | 0.1115 +- 0.0000 |
| `shapenet_nepa` | 0 | 2 | 0.8075 +- 0.0028 |
| `shapenet_nepa` | 1 | 2 | 0.1630 +- 0.0328 |
| `shapenet_nepa` | 5 | 2 | 0.2283 +- 0.0117 |
| `shapenet_nepa` | 10 | 2 | 0.2618 +- 0.0155 |
| `shapenet_nepa` | 20 | 2 | 0.3247 +- 0.0042 |
| `shapenet_mesh_udf_nepa` | 0 | 1 | 0.8178 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 1 | 1 | 0.1345 +- 0.0000 |
| `shapenet_mesh_udf_nepa` | 10 | 1 | 0.2827 +- 0.0000 |
| `shapenet_mix_nepa` | 0 | 1 | 0.8285 +- 0.0000 |
| `shapenet_mix_nepa` | 1 | 1 | 0.1293 +- 0.0000 |
| `shapenet_mix_nepa` | 5 | 1 | 0.2546 +- 0.0000 |
| `shapenet_mix_nepa` | 20 | 1 | 0.3793 +- 0.0000 |
| `shapenet_mix_mae` | 0 | 1 | 0.7856 +- 0.0000 |
| `shapenet_mix_mae` | 1 | 1 | 0.1399 +- 0.0000 |
| `shapenet_mix_mae` | 10 | 1 | 0.2662 +- 0.0000 |

## 6) Notes

- `scratch K=10/20` shows majority-class collapse behavior in current seeds.
- Do not finalize claims until all `75` jobs are complete.
