# PatchNEPA Hypothesis Matrix

Last updated: 2026-03-06

## 1. Purpose

This file keeps the active hypotheses stable across runs so later analysis does
not restart from zero.

Status labels:

- `supported`: evidence is strong enough to guide the next implementation step.
- `rejected`: current evidence argues against the hypothesis.
- `open`: plausible, but not yet decided.
- `engineering pending`: not a scientific hypothesis; still needs a strict run.

## 2. Matrix

| ID | hypothesis | current status | current evidence | next minimal test |
|---|---|---|---|---|
| H1 | cosine target geometry on the v2 token path permits collapse (`cos_tgt ~= cos_prev`) | supported | repeated tiny-gap runs in v2 cosine branch; centered cosine and `skip_k` do not materially change the regime | keep as baseline diagnosis axis; do not spend more time on minor cosine-only tweaks |
| H2 | centered cosine alone fixes the collapse | rejected | `segment` enters negative-gap regime; `shape` converges to near-trivial high cosine without copy-margin improvement | none; treat as explored control |
| H3 | larger `skip_k` fixes the collapse by making the task longer-horizon | rejected | `k=1,2,4` short sweep ends with nearly identical gap | none; keep `skip_k=1` default |
| H4 | reconstruction-space objectives create real context use | supported | `recon_lift_q/a` become positive and stay positive in both `recon_mse` and `recon_chamfer` short runs | continue using reconstruction-aligned diagnostics as the primary probe |
| H5 | `recon_chamfer` is materially better than `recon_mse` in short pretrain diagnostics | rejected | short-run diagnostics are almost numerically identical | treat them as equivalent on short smoke; decide by FT or generator runs |
| H6 | current reconstruction branch already beats the historical v1-family baseline in FT | rejected | `recon_chamfer` best FT remains below the prior v1-family reference | move to generator tests instead of longer no-generator reconstruction only |
| H7 | current transfer gap is partly due to missing reconstruction-side generator depth | open but favored | reconstruction-space lift is healthy while FT remains below v1-family; PointGPT-style reference uses generator path | run `reconstruction + generator_depth>0` under strict logging |
| H8 | PointGPT-style loss parity is required for a strict apples-to-apples comparison | supported | PointGPT objective is context patch reconstruction; PatchNEPA composite loss is a different axis | keep dedicated `pointgpt_ctx_only` runs for strict comparison |
| H9 | dual-mask parity vs PointGPT was previously incomplete | supported | older PatchNEPA runs often used `near/far` regimes; PointGPT-like `column + keep_prefix` parity was added later | use only parity-configured reruns for direct PointGPT mask comparison |
| H10 | PointGPT should be compared to PatchNEPA with cosine probes as the main axis | rejected | PointGPT native objective is reconstruction; PatchNEPA cosine probes are off-objective for recon runs | compare in `recon/copy/lift` and latent-spread space |
| H11 | Point-MAE should be used as architectural evidence for PatchNEPA design choices | rejected | Point-MAE is a benchmark and split sanity control, not a direct causal reconstruction comparator | keep Point-MAE in benchmark pages only |
| H12 | Query-NEPA results can be merged directly into active PatchNEPA benchmark claims | rejected | Query line contains mixed-validity historical runs and obsolete benchmark conditions | use Query-NEPA only through explicit chronology and comparison docs |

## 3. Current Priority Order

1. `reconstruction + generator_depth`
2. strict PointGPT-axis compare (`pointgpt_ctx_only`, aligned mask, aligned diag)
3. FT decision against the historical v1-family reference

## 4. Primary Source Docs

- `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md`
- `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
- `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md`
- `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`
