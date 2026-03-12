# Insight Register

Last updated: 2026-03-12

## Purpose

This file records what each meaningful experiment family taught us.

It exists to answer:

- what new thing was learned from a run or run family,
- which experiments materially changed the project direction,
- which findings are positive, negative, or still only provisional,
- where the supporting evidence lives.

This is not a raw runlog. One row should represent one *decision-relevant*
insight, not one PBS job.

## How To Use

- Read this before opening raw ledgers.
- Use this to find the experiment family that matters.
- Then follow the linked canonical source for exact numbers or provenance.

## Status Labels

- `active`: still part of the current decision surface
- `historical`: useful background, but not a current decision axis
- `negative-result`: useful because it closed a branch
- `provisional`: interesting but not yet strong enough to guide the line
- `invalid`: kept only for provenance; should not guide conclusions

## PatchNEPA / QueryNEPA Insights

| ID | period / line | experiment family | what changed | what was newly learned | status | canonical evidence |
|---|---|---|---|---|---|---|
| I001 | QueryNEPA -> PatchNEPA transition | historical QueryNEPA audit | reviewed mixed historical QueryNEPA runs and benchmark usage | QueryNEPA still provides protocol lessons, but should not be used directly as the current benchmark headline source | historical | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/query_nepa_chronology_audit_202602_active.md` |
| I002 | PatchNEPA v2 cosine branch | centered-cosine / `skip_k` / mask controls | tried `center_mode`, `skip_k`, and later PointGPT-style dual-mask parity | the cosine-path collapse (`cos_tgt ~= cos_prev`) is not explained by these small controls; the bottleneck is deeper than missing column-mask parity alone | negative-result | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`, `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` |
| I003 | PatchNEPA v2 recon short runs | `recon_mse` / `recon_chamfer` short reruns with objective-aligned diags | switched away from cosine-space probes to `recon/copy/lift` diagnostics | reconstruction objectives clearly use context (`recon_lift_q/a > 0`), and short-run `recon_mse` vs `recon_chamfer` is almost identical on these diagnostics | active | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` |
| I004 | PatchNEPA v2 full300 `g0` | composite recon without generator | full300 pretrain + ScanObjectNN FT from no-generator recon | no-generator reconstruction is already a real transfer baseline; it beats the old v1 line on `2/3` ScanObjectNN variants | active | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` |
| I005 | PatchNEPA v2 full300 `g2` | composite recon with generator depth `2` | matched `g0` line but enabled generator depth `2` | `g2` improves over `g0` and the historical v1 line on all three ScanObjectNN variants; this is the current PatchNEPA mainline | active | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md`, `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md` |
| I006 | PointGPT parity branch | strict PointGPT-axis comparison | aligned ShapeNet domain, dual-mask parity, and added `pointgpt_ctx_only` loss mode | strict PointGPT comparison is scientifically important, but still not a fully closed branch because the parity path has had engineering/runtime interruptions | provisional | `nepa3d/docs/patch_nepa/hypothesis_matrix_active.md`, `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` |
| I007 | translation-loss / mini-CPAC screen | `composite` vs `answer_only` vs `context_plus_answer` | short translation-centric screen with mini-CPAC readout | `answer_only` can help thresholded CPAC IoU while hurting `recon_lift_q`; this is interesting, but not enough yet to replace `composite` as the mainline | provisional | `nepa3d/docs/patch_nepa/storyline_query_to_patch_v2_active.md`, `nepa3d/docs/patch_nepa/execution_backlog_active.md`, `nepa3d/docs/patch_nepa/restart_plan_patchnepa_data_v2_20260303.md` |
| I008 | current local phase | local execution backlog + visibility-first branch | moved active next-run policy into a local backlog and staged the visibility/occupancy branch | execution order and scientific conclusions are now intentionally separated; queue state should come from the backlog, not from ad hoc notes or stdout | active | `nepa3d/docs/patch_nepa/execution_backlog_active.md`, `nepa3d/docs/operations/README.md` |

## Current Highest-Value Insights

If a collaborator reads only three things from this register, they should keep:

1. `I002`: the cosine path failed in a robust, informative way
2. `I005`: `recong2` is the current best PatchNEPA line
3. `I007`: translation-side signals are promising but not yet strong enough to change the mainline

## Maintenance Rule

- Add a new row only when an experiment family changes the decision surface.
- If a finding is superseded, do not delete it; change its `status` and point to
  the newer canonical evidence.
- Keep raw job-level history in:
  - `nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md`
  - `nepa3d/docs/query_nepa/runlog_202602.md`
