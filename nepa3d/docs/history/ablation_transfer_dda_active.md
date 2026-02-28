# Transfer-to-DDA Ablation Story (Active)

Last updated: 2026-02-25

## 1. Purpose

This note consolidates how this repo moved from:

- backend transfer validation (ModelNet40-era),
- to DDA quality checks and ray-path robustness,
- to ScanObjectNN protocol debugging and current ablation policy.

Goal: keep one "why we changed what" timeline for future ablation design.

## 2. Source Documents

Primary sources used for this summary:

- `nepa3d/docs/history/legacy_full_history.md`
- `nepa3d/docs/classification/results_scanobjectnn_review_active.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`

## 3. Timeline (Transfer -> DDA -> ScanObjectNN)

### 3.1 ModelNet40-era transfer baseline (legacy)

Context:

- Interface-first design was validated under multiple backends (`mesh`, `pointcloud`, `pointcloud_meshray`) with a shared query-answer token interface.
- Pointcloud ray answers were upgraded using occupancy-based 3D DDA pools (`ray_hit_pc_pool`, `ray_t_pc_pool`, `ray_n_pc_pool`).

Key result (3x3 pretrained fine-tune transfer matrix):

- Aggregate mean over 27 runs: `test_acc=0.8584 +- 0.0101`
- Interpretation at that time: transfer across backends looked viable under one token interface.

### 3.2 DDA quality checkpoint (legacy SHOULD-2)

DDA metrics were explicitly measured on ModelNet40 test cache:

- `hit_acc=0.9583`, `precision=0.8609`, `recall=0.9464`, `f1=0.9016`
- `depth_abs_mean=0.1008`, `depth_abs_p90=0.2349`, `depth_abs_p99=0.9866`

Interpretation:

- DDA geometry itself was not the primary bottleneck in the legacy transfer track.
- This motivated moving attention to protocol/domain mismatch in ScanObjectNN classification behavior.

### 3.3 ScanObjectNN review chain (v0 -> v1 -> v2 -> v3)

Observed issue path:

- `v1/v2` were intended as fair `pc_xyz` runs but were later audited as provisional due to key propagation mismatch:
  - backend did not expose `pc_xyz` in pools,
  - dataset fell back to `pt_xyz_pool`,
  - `v2` eval FPS key was aligned to `pt_xyz_pool`, not `pc_xyz`.

Fix + rerun:

- `v3` corrected key propagation and FPS-key alignment (`pt_xyz_key=pc_xyz`, `pt_fps_key=auto`, `pt_sample_mode_eval=fps`).
- Best-by-K improved versus provisional `v1/v2`:
  - K=0: `v3 0.4859` vs `v1 0.4550` vs `v2 0.2215`
  - K=20: `v3 0.2851` vs `v1 0.2266` vs `v2 0.1698`

Interpretation:

- A major part of earlier degradation came from protocol inconsistency, not only model capacity.

### 3.4 Post-v3 fixed-grid diagnostic (G1/G2)

Diagnostic objective:

- under v0-style query-token setup (`pt_xyz_pool`, `n_point=256`), test deterministic fixed-grid query behavior.

Results (`obj_bg`, K=0 only):

- `G1` (`fixed_grid + mean_no_special`) best: `0.4808`
- `G2` (`fixed_grid + bos`) best: `0.4750`
- both below v0 best (`0.6644`)

Interpretation:

- fixed-grid and BOS pooling did not beat v0 baseline in this first sweep.
- keep as diagnostic track, not mainline claim.

### 3.5 Fine-tune regularization branch (LLRD / drop_path / LayerNorm head)

In the 1024 A/B/C/D operational cycle, fine-tune-side regularization knobs were also evaluated:

- `llrd` (layer-wise learning-rate decay),
- `drop_path` (stochastic depth),
- `use_fc_norm` (LayerNorm on classifier pooled feature),
- plus `label_smoothing` and `weight_decay_norm` split.

Readout from that cycle:

- SOTA-fair branch:
  - `drop_path` was near base (no clear stable gain),
  - `llrd` and `llrd_dp` were clearly lower than base,
  - `fc_norm` improved over base in the regularization sweep.
- Interpretation:
  - regularization knobs mattered, but did not overturn the larger protocol-integrity effects (sampling/split/key alignment).

## 4. DDA-vs-Transfer Reading (Current)

What looks "mostly solved" at this stage:

- DDA feature generation path itself had explicit quality checks in legacy phase.
- 1024 A/B/C/D active pretrain docs also show DDA fallback robustness improvements around preprocessing and cache handling.

What repeatedly dominated outcomes:

- protocol/data-key integrity (`pt_xyz_key`, `pt_fps_key`, sampling mode provenance),
- train/eval mode mismatch and reporting ambiguity (`fps` vs `rfps`),
- split policy and ablation comparability control.

Working conclusion:

- Recent failures are better explained as protocol and transfer-condition mismatches than as pure DDA geometry failure.

## 5. Ablation Episodes and Outcomes

| Episode | Date | Question | Outcome |
|---|---|---|---|
| Transfer matrix (legacy) | 2026-02-19 snapshot in doc | Does interface transfer across backends hold? | Yes on ModelNet40-era setup (`0.8584 +- 0.0101` aggregate). |
| DDA metrics (legacy) | 2026-02-19 snapshot in doc | Is pointcloud DDA ray quality acceptable? | Largely yes (`hit_acc 0.9583`), not the dominant blocker there. |
| Scan review v1/v2 audit | 2026-02-20 | Did fair `pc_xyz` protocol actually run? | No, marked provisional due to key fallback/misalignment. |
| Scan review v3 correction | 2026-02-20 | Does corrected `pc_xyz + auto FPS key` recover? | Yes vs v1/v2, but still below v0 K=0 headline. |
| Fixed-grid G1/G2 | 2026-02-20 | Does deterministic query grid beat v0-style random? | No in first sweep; diagnostic only. |
| Fine-tune regularization sweep (`llrd/drop_path/fc_norm`) | 2026-02-24 to 2026-02-25 | Do scheduler/regularization knobs explain the main variance? | Partially; `fc_norm` helped some SOTA-fair runs, but `llrd` degraded and protocol factors still dominated. |
| 1024 A/B/C/D pretrain audit | 2026-02-25 | Are fps/rfps and protocol labels cleanly tracked? | Policy updated; provenance metadata now mandatory in reports. |

## 6. Current Ablation Policy (for next runs)

Mandatory metadata per result block:

1. pretrain checkpoint path + job IDs
2. `pt_sample_mode_train` (+ `pt_rfps_m` if used)
3. `pt_fps_key`, `pt_xyz_key`, `pt_dist_key`
4. fine-tune `pt_sample_mode_train/eval`
5. cache protocol split (`main_split` mixed vs `obj_bg/obj_only/pb_t50_rs`)

Separation rule:

- do not mix `fps`-pretrained and `rfps`-pretrained outcomes in one conclusion table unless explicitly stratified.

## 7. Suggested Next Ablation Blocks (this machine focus)

1. Keep A/B-only controlled comparisons first (resource-efficient), then expand to C/D only after clear signal.
2. Continue protocol-integrity-first ablations (key/sampling/split correctness) before adding new architecture knobs.
3. Keep fixed-grid/BOS as secondary diagnostic branch until a reproducible win appears against v0/v3 references.
