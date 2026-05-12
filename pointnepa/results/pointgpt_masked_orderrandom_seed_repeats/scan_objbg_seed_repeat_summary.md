# PointGPT-S Mask-On Order-Randomized ScanObjectNN obj_bg Seed Repeats

Date: 2026-05-07 JST.

Purpose:

- Track the completed ScanObjectNN `obj_bg` fine-tuning seed repeats for the mask-on, order-randomized PointGPT-S row.
- Keep this separate from the Q1 2x2 single-run table unless the paper switches to seed-averaged reporting.

Source logs:

- `3D-NEPA/logs/pointgpt_masked_orderrandom_seed_repeats/scan_objbg_seed1.log`
- `3D-NEPA/logs/pointgpt_masked_orderrandom_seed_repeats/scan_objbg_seed2.log`

## Summary

| row | seed | best epoch | best accuracy (%) | epoch300 accuracy (%) |
|---|---:|---:|---:|---:|
| seed1 | 1 | 238 | 91.0499 | 89.1566 |
| seed2 | 2 | 210 | 90.3614 | 89.3287 |
| mean |  | 224 | 90.70565 | 89.24265 |
| sample std |  | 19.79899 | 0.48684 | 0.12169 |

Notes:

- Accuracy values are copied from PointGPT validation logs and are percentages.
- These are stability repeats for the mask-on, order-randomized `obj_bg` fine-tuning row.
- The matched single-run Q1 table row remains a separate result unless the paper explicitly reports seed repeats.
