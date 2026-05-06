# ShapeNetPart Semantic Per-Part Thinning Curve

These rows evaluate per-part thinning (`part_keepXX_per_part`) on ShapeNetPart part segmentation. Metrics are computed on unique retained original points; retained points are resampled only for fixed-size forward inference and logits are averaged back by original point index before scoring. Largest-part removal is not included in this semantic curve.

## Result Files
- Point-MAE: `results/object_ssl_pointmae_pcpmae/semantic_curve/pointmae_shapenetpart_semantic_curve.json`
- PCP-MAE: `results/object_ssl_pointmae_pcpmae/semantic_curve/pcpmae_shapenetpart_semantic_curve.json`
- PCP-MAE-ckpt300: `results/object_ssl_pointmae_pcpmae/semantic_curve/pcpmae_ckpt300_shapenetpart_semantic_curve.json`
- Summary CSV: `results/object_ssl_pointmae_pcpmae/semantic_curve/shapenetpart_semantic_curve_summary.csv`

## Instance mIoU (%)

| model | Semantic80 | Semantic50 | Semantic20 | Semantic10 |
|---|---:|---:|---:|---:|
| Point-MAE | 85.9729 | 85.7288 | 81.8297 | 68.8604 |
| PCP-MAE | 85.9239 | 85.8457 | 81.9898 | 67.8759 |
| PCP-MAE-ckpt300 | 85.6937 | 85.6757 | 81.6351 | 67.2090 |

## Matched Clean-Subset Damage (pp)

| model | Semantic80 | Semantic50 | Semantic20 | Semantic10 |
|---|---:|---:|---:|---:|
| Point-MAE | -0.0105 | 0.2912 | 4.4093 | 17.6509 |
| PCP-MAE | 0.1508 | 0.3362 | 4.3519 | 18.7611 |
| PCP-MAE-ckpt300 | 0.0280 | 0.1752 | 4.5761 | 19.1675 |

## Provenance

- metric scope: `unique_retained_original_points`
- logit aggregation: `mean_by_original_index`
- git commit: `ff82a2e6c73791514a945cb2ed2d6e7ee7d0f24e`
- SemanticR keeps R% of each ground-truth part label before fixed-size forward resampling.
