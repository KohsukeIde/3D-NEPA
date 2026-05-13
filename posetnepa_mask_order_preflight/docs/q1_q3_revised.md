# Revised Q1–Q3 for PosetNEPA after mask-aware redesign

## Q1. Research theme

**PosetNEPA: Geometry-Induced Filtrations for 3D Next-Embedding Prediction.**

3D point clouds have no natural language-like total order, so latent autoregression requires a design choice: what counts as past and future? We test whether geometry-induced orderings/filtrations over point patches produce better and less shortcut-prone next-embedding prediction than arbitrary or proximity-only 1D linearizations.

## Q2. What is new?

- **Primary:** We frame 3D autoregressive pretraining as a filtration design problem, not merely a target-design problem. Language has a natural 1D filtration; point clouds do not.
- **Supporting:** We run mask-aware diagnostics to separate copy-shortcut reduction from masking effects. This prevents claiming PosetNEPA solves a problem already solved by PointGPT masking.
- **Supporting:** We fix grouping and change ordering only, so the experiment tests AR order rather than broken input tokenization.

## Q3. Closest prior work

1. **PointGPT** — AR pretraining over ordered point patches. It shows 3D AR is powerful, but it makes “next” depend on grouping and ordering choices.
2. **NEPA** — next-embedding prediction as a decoder-free latent AR objective. PosetNEPA asks what ordering/filtration this objective should use for unordered 3D data.
3. **Mirai** — visual AR needs foresight. It motivates the broader idea that visual/spatial AR must account for structure beyond immediate next-token supervision.

Point-MAE / PCP-MAE remain important baselines for downstream tables, but they are not the conceptual closest prior for the AR filtration question.
