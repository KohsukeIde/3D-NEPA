# PosetNEPA Lite Literature And Protocol Notes

Status: active notes for interpreting the `stage1_20260513_025903` mask/order preflight.

## Current Claim Boundary

Do not claim that copy-like AR directly causes better semantic classification.

Supported claim:

- Local geometric structure and local patch reconstruction/prediction are useful for point-cloud recognition.
- Therefore, a copy-friendly/local-continuity objective can still produce classification-useful features.

Inference, not yet directly proven:

- If a low-loss AR row has high `copy_win`, the low loss may be measuring local autocorrelation/easiness rather than better long-range AR causality.
- Full ScanObjectNN fine-tuning can hide or erase pretraining-order differences.

Current safest framing:

- A copy-friendly/local order being classification-strong is not paradoxical. Point-cloud recognition has a long local-to-global feature tradition, so local patch continuity can be a useful inductive bias.
- The problematic claim would be stronger: "lower `copy_win` should improve classification." Current local results do not support that yet.
- The research question should be stated as: **what causal/filtration structure makes AR pretraining learn transferable 3D representations, beyond simply exploiting local continuity?**

## Literature Support

Local geometry is recognition-relevant:

- PointNet++ argues that vanilla PointNet misses local structures and that exploiting metric-space neighborhoods learns local features at increasing contextual scales.
- DGCNN/EdgeConv explicitly models local geometric relationships and reports strong classification/segmentation performance.

Patch reconstruction / masked point modeling transfers to classification:

- FoldingNet is an older but clean autoencoder example: reconstruction pretraining learns a latent code that transfers to linear SVM classification. This supports reconstruction as useful representation learning, not copy as a sufficient mechanism.
- Point-BERT uses local point patches and masked point modeling, reporting strong ModelNet40 and ScanObjectNN results plus few-shot transfer.
- Point-MAE reconstructs masked point patches from visible patches and reports strong ScanObjectNN/ModelNet40 and few-shot gains. Its ablations are especially relevant: easier reconstruction can lower reconstruction loss while hurting downstream accuracy, so lower pretext loss is not automatically better representation.
- Point-M2AE reports strong frozen linear SVM performance, indicating that masked hierarchical reconstruction learns transferable representations.
- PointGPT converts point patches into an ordered sequence and uses autoregressive generation for downstream point-cloud tasks.

Ordering / serialization is known to matter for sequence-style point-cloud models:

- PointGPT explicitly orders point patches before AR generation.
- PointGPT also motivates dual masking from the fact that point clouds are unordered, sparse/low-information-density, and redundant enough that naive next-patch generation may be too easy without enough context pressure.
- PointGrow and other 3D AR/generation papers impose a linearization or traversal; they are relevant because they show AR requires an artificial sequence for unordered 3D samples.
- Autoregressive 3D shape generation via canonical mapping is relevant because it treats point-cloud sequentialization as ambiguous and first maps shapes to a canonical domain before spiral/group serialization.
- PointNSP / next-scale point prediction is especially close as a motivation: it criticizes artificial total ordering of unordered point sets and replaces next-token style AR with scale/level-of-detail prediction and intra-scale interaction.
- Point Cloud Mamba and Point Mamba adapt sequence/state-space models to point clouds by serialization/order design.
- Point Transformer V3 uses serialized neighbor mapping/space-filling-curve ideas for scalable point processing.
- In image AR, VAR moves from raster next-token prediction to next-scale prediction, and Mirai injects future/foresight signals. These are not point-cloud papers, but they support the broader concern that strict 1D next-token causality can be misaligned with visual/spatial structure.

What these papers do not directly prove:

- They do not prove that a high `copy_win` AR objective learns better semantic features.
- They do not prove that a total order is causally correct for 3D point-cloud AR.
- They do not evaluate partial-order/frontier NEPA objectives.
- They do not directly prove that PointGPT has a copy shortcut; that must come from local diagnostics such as previous-token baselines, order counterfactuals, and robustness probes.

Useful citation buckets:

- Direct point-cloud support: PointNet++, DGCNN/EdgeConv, FoldingNet, Point-BERT, Point-MAE, Point-M2AE, PointGPT, PointGrow, Point Mamba / Point Cloud Mamba.
- Direct AR-order critique support: PointGPT, PointGrow, canonical-mapping AR 3D generation, PointNSP / next-scale point prediction.
- Analogy support only: VAR, RandAR, Mirai, direction-aware/learned-order image AR papers.
- Shortcut support by analogy: automatic shortcut removal in SSL, reconstruction/downstream mismatch papers, and Point-MAE's location-leakage ablation.

## Why This May Not Have Been Done Directly

Likely reasons:

- Point clouds are unordered sets, so any 1D AR order is artificial and easy to criticize.
- Grouping/local patch construction often dominates downstream performance, making order effects hard to isolate.
- Standard classification benchmarks can be insensitive to AR causality because full fine-tuning and pooling can erase order-specific biases.
- Masked/reconstruction methods already work well and are easier to evaluate than partial-order AR.
- Sequence models for point clouds have mostly focused on efficient serialization/backbones, not on partial-order pretraining objectives.
- Some adjacent visual AR work, including VAR and Mirai, attacks the mismatch between visual structure and raster/strict next-token AR. That strengthens the motivation, but also raises the bar: PosetNEPA must show that its geometry-induced filtration solves an analogous mismatch in 3D rather than merely adding another order.

Ambiguity:

- The user mentioned "mirai"; this note does not identify a specific point-cloud method by that name. If this refers to a particular paper/system, add it explicitly before making related claims.

## Local Protocol Caveats

Stage 1 is not a clean AR-causality test yet.

- Pretrain uses `ckpt-last.pth`; there is no meaningful pretrain `ckpt-best.pth` in this path.
- Fine-tune reports `ckpt-best.pth`, selected on ScanObjectNN `test` as validation in the legacy protocol.
- Fine-tune uses the same `order_mode` and `group_mode` as pretrain, so it tests "pretrain plus same-order full fine-tune", not a common-readout representation probe.
- `order_mode=random` is stochastic every forward and should be treated as augmentation/noise. Use `fixed_random` for a fixed arbitrary total order.
- Pretext diagnostic extraction should be run-tag pinned and epoch-averaged, not mtime-selected or last-batch selected.
- Previous-token/copy baseline should be aggregated over the final logged epoch to align with pretext diagnostics.
- Frozen/readout comparison is required before calling an order effect a representation effect; full fine-tune can wash out or create order effects.

## Next Required Diagnostics

Before broad Stage 2:

1. Finish Stage 1 mask=0.7 rows.
2. Extract epoch-averaged pretext diagnostics.
3. Add previous-token/copy baseline.
4. Run fixed random and deterministic controls.
5. Run pretrain-order x fine-tune-order mismatch matrix.
6. Add scratch/frozen/early-epoch checks to test whether full fine-tuning erases order effects.

Current automation status:

- The post-Stage 1 chain waits for `stage1_20260513_025903`, then refreshes summaries with run-tag-pinned epoch-average diagnostics.
- It runs `fixed_random` controls for mask=0.0 and mask=0.7.
- It runs a small mask-off pretrain-order x fine-tune-order mismatch matrix, skipping diagonal same-order jobs.
- It runs scratch / early epoch / Stage 1 last / Stage 1 frozen comparisons for `simplified_morton` and `diffusion_shell`.

Current Stage 1 diagnostic implication:

- `simplified_morton` has the lowest pretrain loss, but the previous-token baseline is competitive or better under the cosine proxy.
- `diffusion_shell` reduces `copy_win` and, for mask=0.0, beats the previous-token baseline under the cosine proxy, but it does not yet beat `simplified_morton` on completed downstream fine-tune.
- Therefore, current evidence supports "filtration changes the shortcut profile"; it does not yet support "diffusion-shell filtration improves classification."
