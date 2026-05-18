# PosetNEPA Research Framing Notes

Status: active interpretation memo for the May 2026 pointNEPA / PointGPT
mask-order preflight.

## Core Thesis

Autoregressive pretraining is only as meaningful as the information schedule it
imposes. For language, the observed token stream already gives a natural 1D
filtration. For images and point clouds, a 1D sequence is a modeling choice.
For 3D point clouds it is especially artificial because the input is an
unordered set.

Therefore, the central question is not only "what target should NEPA predict?"
but also "what counts as context and what counts as future?"

The clean thesis is:

> 3D autoregressive pretraining is a filtration-design problem. PointGPT-style
> total-order next-patch prediction is a useful baseline, but the next state in
> 3D should be defined by a geometry-induced expansion over the shape rather
> than by an arbitrary linearization.

## Relation To Mirai

Mirai is a useful analogy, not a direct point-cloud predecessor.

Mirai's hypothesis for visual AR is:

- strict next-token causal supervision is natural for language;
- visual data has spatial structure that is not naturally a 1D stream;
- immediate next-token supervision can be too myopic for global visual
  coherence and convergence;
- injecting future/foresight signals into internal representations can improve
  visual AR training.

The 3D analogue here is:

- strict total-order next-patch prediction is not natural for unordered point
  clouds;
- the choice of order defines the AR context/future split;
- a model can optimize an easy local-continuity objective without proving that
  it learned transferable 3D structure;
- geometry-induced filtrations/frontiers are a more native way to define the
  context/future split.

Short version:

- Mirai: 2D visual AR needs foresight beyond strict raster next-token training.
- PosetNEPA: 3D point-cloud AR needs geometry-induced filtrations beyond strict
  total-order next-patch training.

## PointGPT / Point-MAE / PCP-MAE Boundary

The current results should not be read as "copy is good."

PointGPT, Point-MAE, and PCP-MAE all support the opposite caution: point clouds
are redundant, local geometry is highly predictive, and positional/center
side-channels can make reconstruction or next-patch prediction too easy.

The safer interpretation is:

- a copy-friendly/local order can still classify well because local geometric
  continuity is useful for recognition;
- low pretext loss plus high `copy_win` is a shortcut warning, not a success
  criterion;
- Point-MAE/PCP-MAE showing strong performance means point-cloud SSL benchmarks
  can be satisfied by local geometry and carefully controlled masking, so a
  pointNEPA-only accuracy story is weak unless controls rule out leakage and
  full-finetune washout.

In the current PointGPT-style code path, `mask_ratio=0.7` does not fully remove
the immediate predecessor from the shifted AR path. The diagnostic
`copy_win=0.6606` for `simplified_morton_m0p7` is therefore a warning that the
masked condition may still retain an immediate-local shortcut.

## Mathematical View

Language AR uses a natural filtration:

```text
F_0 subset F_1 subset ... subset F_T
```

where `F_t` is the first `t` observed tokens.

For point clouds, a total order over patches is a gauge choice. A more native
3D version is a shape-graph expansion:

```text
G = (V, E)
F_0, F_1, ..., F_K
```

where each `F_t` is a frontier or shell induced by diffusion, geodesic distance,
graph distance, or another geometry-aware rule.

The next-embedding objective becomes:

```text
h_t = Agg({z_i : i in F_t})
hat_h_{t+1} = g(h_{<=t})
L_frontier = 1 - cos(hat_h_{t+1}, stopgrad(h_{t+1}))
```

This is still NEPA-style next-embedding prediction, but the next state is a
frontier in a geometry-induced filtration rather than a single next token in a
linearized sequence.

## Loss Design

The loss choice matters, but it is not the whole contribution.

There are three distinct design layers:

1. **Filtration / schedule:** which patches are context and which patches are
   future.
2. **Target representation:** patch embedding, frontier embedding, teacher
   feature, center-free feature, etc.
3. **Set matching loss:** how to compare predicted frontier sets to target
   frontier sets.

Sinkhorn / optimal transport belongs mainly to layer 3. It is useful when the
next frontier is a set:

```text
C_ij = 1 - cos(hat_z_i, stopgrad(z_j))
L = SinkhornOT(C)
```

This avoids imposing an arbitrary order inside the frontier. Chamfer-cosine is
an easier baseline, but can allow many-to-one matching. Sinkhorn is more aligned
with the "frontier as unordered set" thesis.

Sinkhorn does not define the filtration; diffusion/geodesic/frontier
construction does.

## Current Experimental Reading

As of 2026-05-19 JST:

- `simplified_morton_m0p0` is strongest among completed Stage 1 rows:
  PB-T50-RS `82.3040`, `copy_win=0.5440`.
- `diffusion_shell_m0p0` reduces copy pressure:
  `copy_win=0.4466`, positive gap `0.0096`, but downstream is only `79.4587`.
- `simplified_morton_m0p7` remains strong:
  PB-T50-RS `81.8182`, but `copy_win=0.6606`, which is suspicious under the
  PointGPT / PCP-MAE shortcut framing.
- `random_m0p7` completed at PB-T50-RS `78.5912`.
- `diffusion_shell_m0p7` has pretrain diagnostics but its diagonal full
  fine-tune row did not complete in the original Stage 1 chain because the
  fine-tune wrapper failed after `random_m0p7`. Treat this row as missing from
  the full-finetune table, not negative.
- The `fixed_random` controls completed:
  - `fixed_random_m0p0`: loss `0.1654`, gap `-0.0026`, `copy_win=0.5098`.
  - `fixed_random_m0p7`: loss `0.1806`, gap `-0.0049`, `copy_win=0.5170`.
  - downstream PB-T50-RS: `79.2852` for mask-off and `78.2790` for mask-on.
- The small pretrain-order x fine-tune-order mismatch matrix completed for
  mask-off rows:
  - pretrain `simplified_morton` -> fine-tune `diffusion_shell`: `81.8182`.
  - pretrain `simplified_morton` -> fine-tune `random`: `81.8529`.
  - pretrain `random` -> fine-tune `diffusion_shell`: `78.7994`.
  - pretrain `random` -> fine-tune `simplified_morton`: `78.4178`.
  - pretrain `diffusion_shell` -> fine-tune `random`: `80.2568`.
  - pretrain `diffusion_shell` -> fine-tune `simplified_morton`: `79.3546`.
- The scratch/early/frozen chain completed for mask-off `simplified_morton` and
  `diffusion_shell` on ScanObjectNN hardest:
  - scratch `simplified_morton`: `73.3518`.
  - early `simplified_morton` e1/e5/e10: `74.1152`, `72.8661`, `79.9792`.
  - Stage 1 last `simplified_morton` e30: `82.4774`.
  - frozen-head `simplified_morton` e30: `50.0000`.
  - scratch `diffusion_shell`: `73.0049`.
  - early `diffusion_shell` e1/e5/e10: `75.5031`, `69.3615`, `75.9195`.
  - Stage 1 last `diffusion_shell` e30: `79.3199`.
  - frozen-head `diffusion_shell` e30: `43.0951`.
- The true frozen-feature linear probe also completed on ScanObjectNN hardest:
  - `simplified_morton_m0p0`: `51.8390`.
  - `random_m0p0`: `48.2998`.
  - `diffusion_shell_m0p0`: best `45.4892`, last `44.9341`.
  - `simplified_morton_m0p7`: best `51.1797`, last `51.0062`.
  - `random_m0p7`: `48.4039`.
  - `diffusion_shell_m0p7`: best `47.1201`, last `46.7384`.
  - `fixed_random_m0p0`: best `47.7099`, last `47.5711`.
  - `fixed_random_m0p7`: `48.0569`.
- The skip-k / center-leakage chain completed as
  `skipcenter_20260517_213653` on 2026-05-18 17:59 JST:
  - `skip2`: PB-T50-RS `81.0548`, linear probe `52.9146`,
    `copy_win=0.4829`, gap `0.0074`.
  - `skip4`: PB-T50-RS `80.1527`, linear probe `51.5961`,
    `copy_win=0.4819`, gap `0.0089`.
  - `skip8`: PB-T50-RS `78.9035`, linear probe `49.2019`,
    `copy_win=0.4970`, gap `0.0048`.
  - `poszero`: PB-T50-RS `76.5094`, linear probe `45.9056`,
    `copy_win=0.6224`, gap `-0.0090`.
  - `posshuffle`: PB-T50-RS `80.8813`, linear probe `51.4920`,
    `copy_win=0.6374`, gap `-0.0213`.
  - `centeraux`: PB-T50-RS `82.0264`, linear probe `53.5045`,
    `copy_win=0.5395`, gap `-0.0020`.
  - `skip4_poszero`: PB-T50-RS `77.3074`, linear probe `48.6468`,
    `copy_win=0.4899`, gap `0.0034`.

Current claim boundary:

- Supported: order/filtration changes pretext shortcut profile.
- Not supported: diffusion-shell improves classification in this
  PointGPT-style total-order lite setup.
- Supported after skip-center: the explanation is not just immediate
  previous-token copying. `skip2` remains competitive, but larger skip distance
  degrades both full fine-tune and frozen linear readout. The useful signal is
  local-neighborhood continuity, not only `z_{t-1}` copying.
- Supported after position controls: position/center side-channels matter.
  Zeroing pretrain positions collapses both PB-T50-RS and linear probe, while
  position shuffling can recover under full fine-tune but worsens copy/gap
  diagnostics.
- Not supported: long-range skip-k latent AR as a simple fix. No skip-k row
  gives a large frozen-readout improvement, and larger skips hurt.
- Strengthened caveat: the mismatch matrix suggests full fine-tune can recover
  much of the diagonal order performance even when fine-tune order differs from
  pretrain order. This weakens any claim that the current total-order
  PosetNEPA-Lite rows are learning a robust order-specific representation.
- New readout conclusion: full fine-tuning strongly amplifies the Stage 1
  pretraining advantage (`82.4774` vs scratch `73.3518` for
  `simplified_morton`), but frozen-head and linear-probe performance are only
  around `50-52%`. This means the current evaluation is not a clean
  representation-quality proof. It is mainly evidence that full fine-tune can
  exploit the initialization and rewrite/adapt features.
- Current action: PosetNEPA-Lite is killed as the main route, and the
  skip-k/center chain is now complete. The next method experiment should be
  PosetNEPA-Core smoke: frontier-set prediction with explicit random-frontier
  and position/center controls.
- Still required: frontier-set prediction, random-frontier controls,
  structured robustness/stress readouts, and a cleaner frozen/linear protocol
  if this becomes a paper claim.

The active decision plan is `docs/skip_center_decision_plan_active.md`.

## Next Kill Tests

Before claiming PosetNEPA as a method:

- **Previous-token blocked NEPA:** remove the immediate predecessor path or
  predict with a larger skip.
- **Center-leakage control:** PCP-MAE-style removal/prediction of target center
  information.
- **Fixed-random control:** compare geometry orders to a deterministic arbitrary
  order, not stochastic `random`.
- **Frozen/readout comparison:** verify that the order effect survives without
  full fine-tune rewriting the representation.
- **Frontier set prediction:** replace single next token with a frontier set and
  compare mean/attention pooling, Chamfer-cosine, and Sinkhorn/OT matching.
