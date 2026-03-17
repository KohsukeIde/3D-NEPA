# CQA External Baseline Plan

Goal: compare a frozen point-only SSL encoder against the same `udf_distance`
CQA harness.

Recommended first baseline:
- Point-MAE frozen encoder -> lightweight CQA readout on `pc_bank -> udf_distance`

Minimal implementation options:
1. Reuse `nepa3d/data/cls_patch_dataset.py` style point sampling.
2. Build a wrapper model with:
   - `encode_context(ctx_xyz)` -> context tokens / pooled features
   - small query encoder `MLP(q_xyz) + type_emb`
   - 2-layer generator + answer head identical to CQA
3. Train only the readout/generator; keep the external backbone frozen.

Why this is still a plan rather than an implementation:
- the live repo does not yet include the Point-MAE / PointGPT checkpoint-loading
  glue for this harness,
- exact checkpoint key adaptation is repo-specific and should be validated in
  the live repo rather than copied from an archive.

First evaluation path once a loader exists:
- same/offdiag eval through `eval_primitive_answering_controls.py`
- same completion through `completion_udfdist_cqa.py`
