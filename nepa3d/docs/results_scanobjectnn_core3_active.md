# ScanObjectNN Core3 Active Results

This file stores the paper-safe ScanObjectNN protocol-variant results (OBJ-BG / OBJ-ONLY / PB-T50-RS).

Status note (Feb 17, 2026):

- This file is a **causal-attention baseline snapshot** (legacy classification behavior).
- New bidirectional reruns (`cls_is_causal=0`) are running in:
  - `runs/scan_variants_review_ft_bidir_nray0`
  - `runs/scan_variants_review_lp_bidir_nray0`
- Use `nepa3d/docs/results_scanobjectnn_review_active.md` as the primary active table page.

- Run root: `runs/scan_variants_core3/`
- Log root: `logs/finetune/scan_variants_core3/`
- Status snapshot date: February 15, 2026

## Paper-safe ScanObjectNN protocol variants (`core3`, complete)

Status (as of February 15, 2026):

- completed jobs: `135 / 135`
- variants: `obj_bg`, `obj_only`, `pb_t50_rs`
- methods in this core3 sweep:
  - `scratch`
  - `shapenet_nepa`
  - `shapenet_mesh_udf_nepa`

Artifacts:

- run root: `runs/scan_variants_core3/`
- logs: `logs/finetune/scan_variants_core3/`
- chain log: `logs/finetune/scan_variants_chain/pipeline.log`

`n(seed)=3` for all rows.

### OBJ-BG (`obj_bg`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.4607 +- 0.0353 |
| `scratch` | 1 | 3 | 0.1572 +- 0.0016 |
| `scratch` | 5 | 3 | 0.1629 +- 0.0232 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1320 +- 0.0032 |
| `shapenet_nepa` | 0 | 3 | 0.6391 +- 0.0106 |
| `shapenet_nepa` | 1 | 3 | 0.1807 +- 0.0123 |
| `shapenet_nepa` | 5 | 3 | 0.2800 +- 0.0187 |
| `shapenet_nepa` | 10 | 3 | 0.3276 +- 0.0321 |
| `shapenet_nepa` | 20 | 3 | 0.4033 +- 0.0513 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6575 +- 0.0092 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1819 +- 0.0114 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.3242 +- 0.0255 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.4005 +- 0.0080 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4687 +- 0.0239 |

### OBJ-ONLY (`obj_only`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5594 +- 0.0184 |
| `scratch` | 1 | 3 | 0.1578 +- 0.0181 |
| `scratch` | 5 | 3 | 0.1618 +- 0.0377 |
| `scratch` | 10 | 3 | 0.1343 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1411 +- 0.0097 |
| `shapenet_nepa` | 0 | 3 | 0.6334 +- 0.0074 |
| `shapenet_nepa` | 1 | 3 | 0.1899 +- 0.0118 |
| `shapenet_nepa` | 5 | 3 | 0.2725 +- 0.0086 |
| `shapenet_nepa` | 10 | 3 | 0.3614 +- 0.0115 |
| `shapenet_nepa` | 20 | 3 | 0.4366 +- 0.0141 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6621 +- 0.0165 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1899 +- 0.0081 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.3184 +- 0.0292 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.4039 +- 0.0165 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4968 +- 0.0162 |

### PB-T50-RS (`pb_t50_rs`)

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5133 +- 0.0038 |
| `scratch` | 1 | 3 | 0.1297 +- 0.0246 |
| `scratch` | 5 | 3 | 0.1607 +- 0.0272 |
| `scratch` | 10 | 3 | 0.1353 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1353 +- 0.0000 |
| `shapenet_nepa` | 0 | 3 | 0.5202 +- 0.0108 |
| `shapenet_nepa` | 1 | 3 | 0.1460 +- 0.0214 |
| `shapenet_nepa` | 5 | 3 | 0.1886 +- 0.0140 |
| `shapenet_nepa` | 10 | 3 | 0.2412 +- 0.0162 |
| `shapenet_nepa` | 20 | 3 | 0.2716 +- 0.0146 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.5228 +- 0.0092 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1449 +- 0.0011 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2002 +- 0.0109 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.2407 +- 0.0101 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.2898 +- 0.0215 |

### Best-by-K (core3)

| Variant | K | best method | test_acc mean +- std |
|---|---:|---|---:|
| `obj_bg` | 0 | `shapenet_mesh_udf_nepa` | 0.6575 +- 0.0092 |
| `obj_bg` | 1 | `shapenet_mesh_udf_nepa` | 0.1819 +- 0.0114 |
| `obj_bg` | 5 | `shapenet_mesh_udf_nepa` | 0.3242 +- 0.0255 |
| `obj_bg` | 10 | `shapenet_mesh_udf_nepa` | 0.4005 +- 0.0080 |
| `obj_bg` | 20 | `shapenet_mesh_udf_nepa` | 0.4687 +- 0.0239 |
| `obj_only` | 0 | `shapenet_mesh_udf_nepa` | 0.6621 +- 0.0165 |
| `obj_only` | 1 | `shapenet_nepa` | 0.1899 +- 0.0118 |
| `obj_only` | 5 | `shapenet_mesh_udf_nepa` | 0.3184 +- 0.0292 |
| `obj_only` | 10 | `shapenet_mesh_udf_nepa` | 0.4039 +- 0.0165 |
| `obj_only` | 20 | `shapenet_mesh_udf_nepa` | 0.4968 +- 0.0162 |
| `pb_t50_rs` | 0 | `shapenet_mesh_udf_nepa` | 0.5228 +- 0.0092 |
| `pb_t50_rs` | 1 | `shapenet_nepa` | 0.1460 +- 0.0214 |
| `pb_t50_rs` | 5 | `shapenet_mesh_udf_nepa` | 0.2002 +- 0.0109 |
| `pb_t50_rs` | 10 | `shapenet_nepa` | 0.2412 +- 0.0162 |
| `pb_t50_rs` | 20 | `shapenet_mesh_udf_nepa` | 0.2898 +- 0.0215 |
