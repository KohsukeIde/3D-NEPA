# PointGPT-S Mask-On Order-Randomized ShapeNetPart Seed Repeats

Purpose:

- Track the completed ShapeNetPart fine-tuning seed repeats for the mask-on, order-randomized PointGPT-S row.
- These runs use the order-randomized PointGPT-S pretraining checkpoint and keep masks on.
- Values below are taken from the completed ShapeNetPart fine-tuning logs.

Relevant logs:

- `3D-NEPA/logs/pointgpt_masked_orderrandom_seed_repeats/shapenetpart_seed1.log`
- `3D-NEPA/logs/pointgpt_masked_orderrandom_seed_repeats/shapenetpart_seed2.log`

## Results

| row | seed | best accuracy | best class avg mIoU | best instance avg mIoU | epoch300 accuracy | epoch300 class avg mIoU | epoch300 instance avg mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed1 | 1 | 0.94470 | 0.83604 | 0.85604 | 0.944203 | 0.831859 | 0.854469 |
| seed2 | 2 | 0.94491 | 0.83341 | 0.85741 | 0.944360 | 0.829411 | 0.854190 |
| mean |  | 0.944805 | 0.834725 | 0.856725 | 0.9442815 | 0.830635 | 0.8543295 |
| sample std |  | 0.0001485 | 0.0018597 | 0.0009687 | 0.0001110 | 0.0017310 | 0.0001973 |

## Notes

- The paper table should use the matched single-run row unless it explicitly reports seed repeats.
- If using the seed-repeat mean, the mask-on order-randomized ShapeNetPart instance mIoU is `85.67%`.
- The completed single-run value already used in the Q1 2x2 table is `85.44%`; the seed-repeat mean is a stability check, not a replacement unless the table is converted to seed-averaged reporting.
