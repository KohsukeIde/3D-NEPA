# Ablation Timeline (All Docs, Active)

Last updated: 2026-02-25

## 1. Purpose

This document consolidates **all docs under `nepa3d/docs/`** into one ablation-oriented timeline.

Use this as the primary "what happened, when, and why it matters" memo before launching new ablations on this machine.

## 2. Coverage (All Docs)

| Doc | Role | Main time window in content | Ablation value |
|---|---|---|---|
| `nepa3d/docs/results_index.md` | navigation hub | updated through 2026-02-25 | entry-point map; prevents active/legacy mixups |
| `nepa3d/docs/history/legacy_full_history.md` | full legacy archive | up to 2026-02-19 memo | canonical source for transfer matrix, DDA metrics, early UCPR/CPAC + MUST runs |
| `nepa3d/docs/classification/results_modelnet40_legacy.md` | ModelNet legacy summary | v0/v1 era snapshot | historical transfer baseline table |
| `nepa3d/docs/classification/results_scanobjectnn_m1_legacy.md` | ScanObjectNN M1 legacy table | 2026-02-14 snapshot | old cache/protocol reference (`scanobjectnn_cache_v2`) |
| `nepa3d/docs/classification/results_scanobjectnn_core3_active.md` | protocol-variant causal baseline | 2026-02-15 to 2026-02-17 status note | pre-bidir reference for `obj_bg/obj_only/pb_t50_rs` |
| `nepa3d/docs/classification/results_scanobjectnn_review_legacy.md` | legacy review snapshot | pre-bidir causal + interim notes | comparison target for active v0->v3 audit |
| `nepa3d/docs/classification/results_scanobjectnn_review_active.md` | active Scan review | 2026-02-18 to 2026-02-20 | core source for v0/v1/v2/v3, D1/D2, G1/G2, protocol-integrity findings |
| `nepa3d/docs/classification/results_modelnet40_pointgpt_active.md` | ModelNet PointGPT protocol page | 2026-02-17 status note | causal baseline + bidir rerun context |
| `nepa3d/docs/completion/eccv_ucpr_cpac_tables.md` | stable table plan template | planning doc (no single snapshot date) | run-matrix contracts for UCPR/CPAC/few-shot |
| `nepa3d/docs/completion/results_ucpr_cpac_active.md` | active UCPR/CPAC log | mainly 2026-02-15 to 2026-02-20 | QA cycle, MAE parity, A/B/C/D/E/6 blocks, core completion metrics |
| `nepa3d/docs/completion/results_ucpr_cpac_active_mixed_archive.md` | raw archival backup | same era as active UCPR/CPAC | deep command/result provenance backup |
| `nepa3d/docs/completion/results_ucpr_cpac_plane_baselines_active.md` | plane-baseline isolate | 2026-02-15 to 2026-02-17 | tri-plane/k-plane branches separated from mainline |
| `nepa3d/docs/completion/plan_completion_ae6.md` | completion execution policy | 2026-02-16 with 2026-02-20 updates | priority and policy decisions (objective-preserving mainline, UCPR diagnostic-only) |
| `nepa3d/docs/completion/results_completion_ae6_active.md` | completion ablation ledger | 2026-02-16 to 2026-02-20 | detailed A/B/C/D/E/6 outcomes, including scale retries |
| `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md` | 1024 A/B/C/D + multi-node operations | 2026-02-23 to 2026-02-25 | current infra/protocol hotfixes, A/B-only resource strategy, fps/rfps provenance policy |
| `nepa3d/docs/history/answer_token_expansion_crazy_ideas.md` | answer-token idea memo | 2026-02-19 to 2026-02-20 notes | objective-preserving feature-side expansion candidates |
| `nepa3d/docs/history/ablation_transfer_dda_active.md` | condensed storyline | 2026-02-25 | bridge summary (Transfer -> DDA -> Scan review), now superseded by this full-doc timeline |

## 3. Chronological Summary (Ablation-Focused)

### 2026-02-13 to 2026-02-14: Legacy transfer baseline phase

Sources:

- `nepa3d/docs/history/legacy_full_history.md`
- `nepa3d/docs/classification/results_modelnet40_legacy.md`
- `nepa3d/docs/classification/results_scanobjectnn_m1_legacy.md`

Key points:

- ModelNet transfer matrix established early confidence in shared interface transfer under multiple backends.
- Legacy ScanObjectNN M1 (`scanobjectnn_cache_v2`) was completed but is explicitly non-current protocol.
- This phase is now reference-only and should not be mixed into current fair-comparison claims.

### 2026-02-15: UCPR/CPAC and plane-baseline expansion

Sources:

- `nepa3d/docs/completion/results_ucpr_cpac_active.md`
- `nepa3d/docs/completion/results_ucpr_cpac_plane_baselines_active.md`
- `nepa3d/docs/completion/eccv_ucpr_cpac_tables.md`

Key points:

- QA/dualmask follow-ups, CPAC non-trans settings, and MAE-vs-NEPA comparisons were logged in detail.
- Plane baselines (k-plane / tri-plane) were branched as an isolated track to avoid contaminating NEPA/MAE mainline interpretations.
- Table-plan contracts (UCPR/CPAC/few-shot) were formalized.

### 2026-02-16: Completion A/B/C/D/E/6 rapid ablation wave

Sources:

- `nepa3d/docs/completion/results_completion_ae6_active.md`
- `nepa3d/docs/completion/plan_completion_ae6.md`

Key points:

- Progress-8 merge introduced scaling hooks (`max_len`, pos-emb resize, schedules).
- A/B/C/D/E/6 probes were executed quickly in seed0-first style.
- `B-2 + C-2` became the strongest completion-side candidate in the fixed-size line.
- Scale-long attempts showed instability/regression versus short scale quick wins.

### 2026-02-17: Core3/PointGPT status + completion refinements

Sources:

- `nepa3d/docs/classification/results_scanobjectnn_core3_active.md`
- `nepa3d/docs/classification/results_modelnet40_pointgpt_active.md`
- `nepa3d/docs/completion/results_completion_ae6_active.md`

Key points:

- ScanObjectNN core3 causal baseline snapshot was fixed as pre-bidir reference.
- PointGPT-style ModelNet page remained causal baseline while bidirectional rerun was pending.
- Completion side added B-3 full fill and A-1 coarse-to-fine query checks.

### 2026-02-18 to 2026-02-20: ScanObjectNN review correction era

Sources:

- `nepa3d/docs/classification/results_scanobjectnn_review_active.md`
- `nepa3d/docs/classification/results_scanobjectnn_review_legacy.md`
- `nepa3d/docs/completion/plan_completion_ae6.md`
- `nepa3d/docs/completion/results_completion_ae6_active.md`

Key points:

- v1/v2 were audited as provisional due to `pt_xyz_key` fallback and FPS-key misalignment.
- v3 fixed key propagation and aligned FPS semantics (`pt_fps_key=auto` path).
- Dist-enabled D1/D2 and fixed-grid diagnostics G1/G2 were completed and separated as diagnostic branches.
- Policy shifted: completion main judgment should remain objective-preserving, and UCPR should be diagnostic-only.

### 2026-02-19 to 2026-02-20: Idea and interpretation consolidation

Sources:

- `nepa3d/docs/history/answer_token_expansion_crazy_ideas.md`
- `nepa3d/docs/completion/plan_completion_ae6.md`

Key points:

- Feature-side answer-token expansion direction was documented while keeping NEPA objective unchanged.
- EncDec/plusgut chain interpretation was explicitly marked diagnostic where needed.
- Reporting hygiene (rounding, config checks, JSON/ckpt cross-checks) was codified.

### 2026-02-23 to 2026-02-25: 1024 A/B/C/D multi-node operational phase

Source:

- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`

Key points:

- Multi-node pretrain/eval pipelines were run and repeatedly hotfixed (QA token forwarding, CPAC type-hint issue, protocol split propagation).
- Fine-tune regularization branches were explicitly executed:
  - knobs: `llrd`, `drop_path`, `use_fc_norm` (LayerNorm head), `label_smoothing`, `weight_decay_norm` split.
  - logs/results were recorded as dedicated ablation sets (`base`, `llrd`, `dp`, `llrd_dp` and pool/LS/reg tables).
  - observed direction in that cycle: `drop_path` was near-tie with base, `llrd`/`llrd_dp` were clearly worse, and `fc_norm` helped under SOTA-fair settings.
- Several interpretation risks were identified and addressed:
  - `qa_tokens` forwarding mismatch,
  - LR scaling formula/default behavior,
  - split leakage (group-aware val split introduction),
  - mixed protocol reporting (`main_split` vs variant split),
  - sampling provenance ambiguity (`fps` vs `rfps`).
- Policy outcome by 2026-02-25:
  - mandatory metadata for every result row,
  - resource scope reduced to A/B for faster loops,
  - LR rollback to `1e-4` default for controlled comparability,
  - explicit separate handling for `fps` and `rfps` claims.

## 4. Cross-Doc Conclusions (What is stable)

1. Protocol integrity dominates outcome interpretation.
- `pt_xyz_key`, FPS key alignment, sampling mode provenance, and split policy affected results more than many architecture tweaks.

2. DDA quality itself is not the primary current blocker.
- Legacy DDA quality checks were strong; recent regressions were mostly protocol/reporting mismatches.

3. Keep objective-preserving mainline and objective-side probes separated.
- B-2/C-2 style objective-side gains can be useful, but paper-main narrative is currently objective-preserving.

4. Completion and classification should be reported with explicit branch boundaries.
- Do not merge diagnostic tracks (fixed-grid, plusgut diagnostic variants, mixed protocol runs) into headline fair-comparison tables.

## 5. Ablation Checklist for New Runs (Use This Before Launch)

1. Declare protocol signature up front.
- `pt_xyz_key`, `pt_dist_key`, `pt_fps_key`, `pt_sample_mode_train/eval`, `pt_rfps_m`, `point_order_mode`, split/cache root.

2. Declare provenance.
- pretrain ckpt path + job IDs + whether pretrain is `fps` or `rfps`.

3. Declare comparison class.
- objective-preserving mainline vs objective-side diagnostic.

4. Enforce split hygiene.
- prefer group-aware validation (`group_auto` / ScanObjectNN group mode) for internal model selection.

5. Keep branch separation in final tables.
- `main_split` mixed runs, variant-split runs (`obj_bg/obj_only/pb_t50_rs`), and diagnostic sweeps must be stratified.

## 6. Recommended Next Ablation Slice (this machine)

Given the full timeline, the lowest-risk high-signal next slice is:

1. A/B-only controlled comparisons with strict provenance labels (`fps` vs `rfps` separated).
2. Variant-split ScanObjectNN reporting (`obj_bg/obj_only/pb_t50_rs`) under one fixed eval protocol.
3. Completion-side objective-preserving improvements first; keep objective-side auxiliary tracks as explicit diagnostics.
