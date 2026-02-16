# ScanObjectNN Review Tables (paper-safe core3)

Source run roots:

- `runs/scan_variants_review_ft_nray0` (full fine-tune)

- `runs/scan_variants_review_lp_nray0` (linear probe / freeze backbone)


Artifacts:

- raw: `results/scan_variants_review_raw.csv`

- summary: `results/scan_variants_review_summary.csv`


## Full Fine-tune

### obj_bg

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.4859 +- 0.0334 |
| `scratch` | 1 | 3 | 0.1595 +- 0.0066 |
| `scratch` | 5 | 3 | 0.1790 +- 0.0281 |
| `scratch` | 10 | 3 | 0.1532 +- 0.0268 |
| `scratch` | 20 | 3 | 0.1566 +- 0.0198 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6454 +- 0.0049 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1807 +- 0.0024 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.3006 +- 0.0247 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.3901 +- 0.0187 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4727 +- 0.0126 |
| `shapenet_mix_mae` | 0 | 3 | 0.6190 +- 0.0021 |
| `shapenet_mix_mae` | 1 | 3 | 0.1721 +- 0.0138 |
| `shapenet_mix_mae` | 5 | 3 | 0.2691 +- 0.0261 |
| `shapenet_mix_mae` | 10 | 3 | 0.3356 +- 0.0056 |
| `shapenet_mix_mae` | 20 | 3 | 0.3844 +- 0.0191 |
| `shapenet_mix_nepa` | 0 | 3 | 0.6718 +- 0.0077 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1945 +- 0.0092 |
| `shapenet_mix_nepa` | 5 | 3 | 0.3264 +- 0.0189 |
| `shapenet_mix_nepa` | 10 | 3 | 0.4177 +- 0.0176 |
| `shapenet_mix_nepa` | 20 | 3 | 0.4997 +- 0.0131 |
| `shapenet_nepa` | 0 | 3 | 0.6397 +- 0.0091 |
| `shapenet_nepa` | 1 | 3 | 0.1738 +- 0.0138 |
| `shapenet_nepa` | 5 | 3 | 0.2869 +- 0.0154 |
| `shapenet_nepa` | 10 | 3 | 0.3190 +- 0.0199 |
| `shapenet_nepa` | 20 | 3 | 0.4148 +- 0.0221 |

### obj_only

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5841 +- 0.0071 |
| `scratch` | 1 | 3 | 0.1572 +- 0.0142 |
| `scratch` | 5 | 3 | 0.1503 +- 0.0358 |
| `scratch` | 10 | 3 | 0.1503 +- 0.0227 |
| `scratch` | 20 | 3 | 0.1480 +- 0.0183 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.6609 +- 0.0088 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1836 +- 0.0150 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.3219 +- 0.0208 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.4200 +- 0.0304 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.4682 +- 0.0267 |
| `shapenet_mix_mae` | 0 | 3 | 0.6179 +- 0.0098 |
| `shapenet_mix_mae` | 1 | 3 | 0.2232 +- 0.0043 |
| `shapenet_mix_mae` | 5 | 3 | 0.2811 +- 0.0186 |
| `shapenet_mix_mae` | 10 | 3 | 0.3408 +- 0.0263 |
| `shapenet_mix_mae` | 20 | 3 | 0.4200 +- 0.0316 |
| `shapenet_mix_nepa` | 0 | 3 | 0.6690 +- 0.0029 |
| `shapenet_mix_nepa` | 1 | 3 | 0.2129 +- 0.0135 |
| `shapenet_mix_nepa` | 5 | 3 | 0.3127 +- 0.0077 |
| `shapenet_mix_nepa` | 10 | 3 | 0.4165 +- 0.0246 |
| `shapenet_mix_nepa` | 20 | 3 | 0.4945 +- 0.0045 |
| `shapenet_nepa` | 0 | 3 | 0.6351 +- 0.0037 |
| `shapenet_nepa` | 1 | 3 | 0.1767 +- 0.0197 |
| `shapenet_nepa` | 5 | 3 | 0.2846 +- 0.0231 |
| `shapenet_nepa` | 10 | 3 | 0.3523 +- 0.0316 |
| `shapenet_nepa` | 20 | 3 | 0.4464 +- 0.0128 |

### pb_t50_rs

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.5034 +- 0.0068 |
| `scratch` | 1 | 3 | 0.1641 +- 0.0197 |
| `scratch` | 5 | 3 | 0.1781 +- 0.0335 |
| `scratch` | 10 | 3 | 0.1523 +- 0.0240 |
| `scratch` | 20 | 3 | 0.1404 +- 0.0067 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.5202 +- 0.0057 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1266 +- 0.0168 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.2060 +- 0.0193 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.2445 +- 0.0100 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.2927 +- 0.0201 |
| `shapenet_mix_mae` | 0 | 3 | 0.5113 +- 0.0061 |
| `shapenet_mix_mae` | 1 | 3 | 0.1616 +- 0.0127 |
| `shapenet_mix_mae` | 5 | 3 | 0.2106 +- 0.0064 |
| `shapenet_mix_mae` | 10 | 3 | 0.2407 +- 0.0065 |
| `shapenet_mix_mae` | 20 | 3 | 0.2681 +- 0.0007 |
| `shapenet_mix_nepa` | 0 | 3 | 0.5501 +- 0.0023 |
| `shapenet_mix_nepa` | 1 | 3 | 0.1398 +- 0.0193 |
| `shapenet_mix_nepa` | 5 | 3 | 0.2223 +- 0.0087 |
| `shapenet_mix_nepa` | 10 | 3 | 0.2591 +- 0.0026 |
| `shapenet_mix_nepa` | 20 | 3 | 0.3038 +- 0.0068 |
| `shapenet_nepa` | 0 | 3 | 0.5056 +- 0.0086 |
| `shapenet_nepa` | 1 | 3 | 0.1378 +- 0.0185 |
| `shapenet_nepa` | 5 | 3 | 0.2003 +- 0.0078 |
| `shapenet_nepa` | 10 | 3 | 0.2155 +- 0.0083 |
| `shapenet_nepa` | 20 | 3 | 0.2666 +- 0.0097 |

## Linear Probe

### obj_bg

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1457 +- 0.0107 |
| `scratch` | 1 | 3 | 0.1067 +- 0.0155 |
| `scratch` | 5 | 3 | 0.1067 +- 0.0129 |
| `scratch` | 10 | 3 | 0.1394 +- 0.0085 |
| `scratch` | 20 | 3 | 0.1222 +- 0.0208 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.2989 +- 0.0049 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.0964 +- 0.0098 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1119 +- 0.0097 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.1308 +- 0.0249 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.1928 +- 0.0212 |
| `shapenet_mix_mae` | 0 | 3 | 0.1371 +- 0.0008 |
| `shapenet_mix_mae` | 1 | 3 | 0.1153 +- 0.0061 |
| `shapenet_mix_mae` | 5 | 3 | 0.1538 +- 0.0192 |
| `shapenet_mix_mae` | 10 | 3 | 0.0924 +- 0.0317 |
| `shapenet_mix_mae` | 20 | 3 | 0.1779 +- 0.0288 |
| `shapenet_mix_nepa` | 0 | 3 | 0.2737 +- 0.0129 |
| `shapenet_mix_nepa` | 1 | 3 | 0.0734 +- 0.0292 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1153 +- 0.0112 |
| `shapenet_mix_nepa` | 10 | 3 | 0.1188 +- 0.0152 |
| `shapenet_mix_nepa` | 20 | 3 | 0.1589 +- 0.0148 |
| `shapenet_nepa` | 0 | 3 | 0.2851 +- 0.0021 |
| `shapenet_nepa` | 1 | 3 | 0.1067 +- 0.0155 |
| `shapenet_nepa` | 5 | 3 | 0.1308 +- 0.0184 |
| `shapenet_nepa` | 10 | 3 | 0.1383 +- 0.0094 |
| `shapenet_nepa` | 20 | 3 | 0.1773 +- 0.0042 |

### obj_only

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1532 +- 0.0186 |
| `scratch` | 1 | 3 | 0.1061 +- 0.0168 |
| `scratch` | 5 | 3 | 0.1044 +- 0.0175 |
| `scratch` | 10 | 3 | 0.1348 +- 0.0008 |
| `scratch` | 20 | 3 | 0.1371 +- 0.0041 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.3098 +- 0.0074 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1033 +- 0.0134 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1343 +- 0.0106 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.1348 +- 0.0211 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.1916 +- 0.0197 |
| `shapenet_mix_mae` | 0 | 3 | 0.1365 +- 0.0008 |
| `shapenet_mix_mae` | 1 | 3 | 0.1216 +- 0.0181 |
| `shapenet_mix_mae` | 5 | 3 | 0.1589 +- 0.0176 |
| `shapenet_mix_mae` | 10 | 3 | 0.0906 +- 0.0365 |
| `shapenet_mix_mae` | 20 | 3 | 0.1865 +- 0.0260 |
| `shapenet_mix_nepa` | 0 | 3 | 0.3414 +- 0.0065 |
| `shapenet_mix_nepa` | 1 | 3 | 0.0964 +- 0.0212 |
| `shapenet_mix_nepa` | 5 | 3 | 0.1182 +- 0.0150 |
| `shapenet_mix_nepa` | 10 | 3 | 0.1256 +- 0.0061 |
| `shapenet_mix_nepa` | 20 | 3 | 0.1956 +- 0.0118 |
| `shapenet_nepa` | 0 | 3 | 0.3184 +- 0.0092 |
| `shapenet_nepa` | 1 | 3 | 0.1165 +- 0.0138 |
| `shapenet_nepa` | 5 | 3 | 0.1325 +- 0.0146 |
| `shapenet_nepa` | 10 | 3 | 0.1383 +- 0.0086 |
| `shapenet_nepa` | 20 | 3 | 0.1939 +- 0.0152 |

### pb_t50_rs

| Method | K | n(seed) | test_acc mean +- std |
|---|---:|---:|---:|
| `scratch` | 0 | 3 | 0.1690 +- 0.0105 |
| `scratch` | 1 | 3 | 0.1071 +- 0.0170 |
| `scratch` | 5 | 3 | 0.1070 +- 0.0185 |
| `scratch` | 10 | 3 | 0.1353 +- 0.0000 |
| `scratch` | 20 | 3 | 0.1406 +- 0.0075 |
| `shapenet_mesh_udf_nepa` | 0 | 3 | 0.3287 +- 0.0011 |
| `shapenet_mesh_udf_nepa` | 1 | 3 | 0.1058 +- 0.0075 |
| `shapenet_mesh_udf_nepa` | 5 | 3 | 0.1143 +- 0.0141 |
| `shapenet_mesh_udf_nepa` | 10 | 3 | 0.1260 +- 0.0116 |
| `shapenet_mesh_udf_nepa` | 20 | 3 | 0.1718 +- 0.0154 |
| `shapenet_mix_mae` | 0 | 3 | 0.2429 +- 0.0011 |
| `shapenet_mix_mae` | 1 | 3 | 0.1012 +- 0.0253 |
| `shapenet_mix_mae` | 5 | 3 | 0.1268 +- 0.0249 |
| `shapenet_mix_mae` | 10 | 3 | 0.0939 +- 0.0350 |
| `shapenet_mix_mae` | 20 | 3 | 0.1457 +- 0.0264 |
| `shapenet_mix_nepa` | 0 | 3 | 0.3014 +- 0.0002 |
| `shapenet_mix_nepa` | 1 | 3 | 0.0744 +- 0.0077 |
| `shapenet_mix_nepa` | 5 | 3 | 0.0840 +- 0.0057 |
| `shapenet_mix_nepa` | 10 | 3 | 0.0999 +- 0.0020 |
| `shapenet_mix_nepa` | 20 | 3 | 0.1433 +- 0.0045 |
| `shapenet_nepa` | 0 | 3 | 0.2984 +- 0.0017 |
| `shapenet_nepa` | 1 | 3 | 0.1114 +- 0.0303 |
| `shapenet_nepa` | 5 | 3 | 0.1106 +- 0.0185 |
| `shapenet_nepa` | 10 | 3 | 0.1170 +- 0.0093 |
| `shapenet_nepa` | 20 | 3 | 0.1530 +- 0.0083 |

## Best-by-K (Full Fine-tune)

| Variant | K | best method | test_acc mean +- std |
|---|---:|---|---:|
| `obj_bg` | 0 | `shapenet_mix_nepa` | 0.6718 +- 0.0077 |
| `obj_bg` | 1 | `shapenet_mix_nepa` | 0.1945 +- 0.0092 |
| `obj_bg` | 5 | `shapenet_mix_nepa` | 0.3264 +- 0.0189 |
| `obj_bg` | 10 | `shapenet_mix_nepa` | 0.4177 +- 0.0176 |
| `obj_bg` | 20 | `shapenet_mix_nepa` | 0.4997 +- 0.0131 |
| `obj_only` | 0 | `shapenet_mix_nepa` | 0.6690 +- 0.0029 |
| `obj_only` | 1 | `shapenet_mix_mae` | 0.2232 +- 0.0043 |
| `obj_only` | 5 | `shapenet_mesh_udf_nepa` | 0.3219 +- 0.0208 |
| `obj_only` | 10 | `shapenet_mesh_udf_nepa` | 0.4200 +- 0.0304 |
| `obj_only` | 20 | `shapenet_mix_nepa` | 0.4945 +- 0.0045 |
| `pb_t50_rs` | 0 | `shapenet_mix_nepa` | 0.5501 +- 0.0023 |
| `pb_t50_rs` | 1 | `scratch` | 0.1641 +- 0.0197 |
| `pb_t50_rs` | 5 | `shapenet_mix_nepa` | 0.2223 +- 0.0087 |
| `pb_t50_rs` | 10 | `shapenet_mix_nepa` | 0.2591 +- 0.0026 |
| `pb_t50_rs` | 20 | `shapenet_mix_nepa` | 0.3038 +- 0.0068 |

## Best-by-K (Linear Probe)

| Variant | K | best method | test_acc mean +- std |
|---|---:|---|---:|
| `obj_bg` | 0 | `shapenet_mesh_udf_nepa` | 0.2989 +- 0.0049 |
| `obj_bg` | 1 | `shapenet_mix_mae` | 0.1153 +- 0.0061 |
| `obj_bg` | 5 | `shapenet_mix_mae` | 0.1538 +- 0.0192 |
| `obj_bg` | 10 | `scratch` | 0.1394 +- 0.0085 |
| `obj_bg` | 20 | `shapenet_mesh_udf_nepa` | 0.1928 +- 0.0212 |
| `obj_only` | 0 | `shapenet_mix_nepa` | 0.3414 +- 0.0065 |
| `obj_only` | 1 | `shapenet_mix_mae` | 0.1216 +- 0.0181 |
| `obj_only` | 5 | `shapenet_mix_mae` | 0.1589 +- 0.0176 |
| `obj_only` | 10 | `shapenet_nepa` | 0.1383 +- 0.0086 |
| `obj_only` | 20 | `shapenet_mix_nepa` | 0.1956 +- 0.0118 |
| `pb_t50_rs` | 0 | `shapenet_mesh_udf_nepa` | 0.3287 +- 0.0011 |
| `pb_t50_rs` | 1 | `shapenet_nepa` | 0.1114 +- 0.0303 |
| `pb_t50_rs` | 5 | `shapenet_mix_mae` | 0.1268 +- 0.0249 |
| `pb_t50_rs` | 10 | `scratch` | 0.1353 +- 0.0000 |
| `pb_t50_rs` | 20 | `shapenet_mesh_udf_nepa` | 0.1718 +- 0.0154 |

## Completeness

- `full_ft`: 225/225 complete
- `linear_probe`: 225/225 complete
