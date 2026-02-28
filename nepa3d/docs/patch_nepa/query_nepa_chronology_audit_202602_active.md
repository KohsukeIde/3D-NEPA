# Query-NEPA Chronology Audit (for Patch-NEPA Porting)

Last updated: 2026-02-28

## 1. Purpose

This document defines which Query-NEPA findings are still valid for porting and which are obsolete.
Scope is pretrain-related parity only (excluding `split/dual_mask/QA` family).

Primary sources:

- `nepa3d/docs/query_nepa/pretrain_abcd_1024_multinode_active.md`
- `nepa3d/docs/query_nepa/pretrain_abcd_1024_variant_reval_active.md`
- `nepa3d/docs/query_nepa/runlog_202602.md`

## 2. Chronology (Validity Boundary)

### Era A: early Query-NEPA runs (historical, mixed validity)

Common issues repeatedly recorded in ledgers:

- argument forwarding mismatches (`qa_tokens`, env propagation),
- protocol mismatch between intended run setting and actual launch setting,
- max-length and mesh-eval mismatch failures in some bundles.

Status:

- keep as provenance only; not benchmark-grade evidence.

### Era B: protocol hardening in Query line (valid engineering lessons)

Confirmed good practices that remain valid:

- explicit run-time config print in launcher logs,
- fail-fast for data/key mismatch (no silent zero-fallback in non-ablate mode),
- split-policy clarity (`file` vs `group_*` vs legacy `pointmae`),
- strict separation of headline benchmark vs historical logs.

Status:

- valid and should be carried into Patch-NEPA line.

### Era C: Patch migration start (current active)

Decision now fixed in docs:

- Query-NEPA is historical reference line.
- Patch-NEPA is active implementation line.
- port only validated ingredients with explicit runlog evidence.

## 3. Query Variable/Feature Effects (Pretrain) and Porting Status

| Item | Effect in Query-NEPA | Port status to Patch-NEPA | Current policy |
|---|---|---|---|
| `pt_sample_mode_train=random` | high stochasticity; weaker run-to-run comparability | supported | not default |
| `pt_sample_mode_train=fps` | deterministic FPS subset; depends on FPS-order availability | supported | not default |
| `pt_sample_mode_train=rfps` | randomized FPS-like subset; robust compromise | supported | **default** |
| `pt_rfps_m` | RFPS candidate pool size; affects diversity/speed | supported | default `4096` |
| `pt_fps_key` | cache key for precomputed FPS order | supported | default `auto` |
| `pt_xyz_key` / `pt_dist_key` | selects point source/aux distance channel | supported | point-only baseline uses `pc_xyz` + dist ablate |
| `ablate_point_dist` | disables dist channel to avoid dist-mismatch artifacts | supported | point-only baseline `1` |
| `point_order_mode` | ordering after sampling (`morton/fps/random`) | supported | default `morton` |
| `lr_scheduler=none` | fixed LR recipe used in many Query baselines | supported | **default** |
| `auto_resume` + `resume_optimizer` | robustness for long runs/restarts | supported | **default on** |
| `qk_norm_affine/bias`, `layerscale`, `rope_theta` | backbone stability/compatibility controls | supported | aligned defaults (`0/0`, `1e-5`, `100`) |

## 4. Query Features Marked Out-of-Scope for Current Patch Baseline

These are Query-era features and are intentionally not in current Patch baseline pretrain:

- `qa_layout` (`interleave/split/split_sep`),
- `sequence_mode` (`block/event`),
- `event_order_mode`, `ray_order_mode`,
- dual-mask controls and Query auxiliary objective family (`B2/B3/C/D/E`).

These are ported only in dedicated milestones after point-only patch baseline is stable.

## 5. Obsolete or Restricted Practices

- Treat `scanobjectnn_main_split_v2` mixed-cache headline reporting as obsolete for fair benchmark.
- Treat legacy `test-as-val` (`val_split_mode=pointmae`) as reproduction-only.
- Treat run sets with known config mismatch or DDP/eval-path mismatch as invalid for comparison.

## 6. Active Patch-NEPA Pretrain Baseline (frozen defaults)

- `patch_embed=fps_knn`
- `group_size=32`, `num_groups=64`
- `n_point=1024`, `n_ray=0`, `use_ray_patch=0`
- `pt_xyz_key=pc_xyz`, `ablate_point_dist=1`
- `pt_sample_mode=rfps`, `pt_rfps_m=4096`
- `batch(per-proc)=16`, global batch from topology
- `lr=3e-4`, `weight_decay=0.05`, `lr_scheduler=none`
- `auto_resume=1`, `resume_optimizer=1`

## 7. Operational Rule

For any new Patch-NEPA pretrain/eval claim:

1. prove run config from node log header,
2. prove no key/fallback warnings affecting sample path,
3. classify each referenced Query-era item as either:
   - valid carryover, or
   - historical/outdated.

## 8. CLI Surface Diff Snapshot (Query pretrain vs Patch pretrain)

Static argparse diff (2026-02-28):

- Query pretrain args: `109`
- Patch pretrain args: `66`
- shared args: `32`

Main Query-only clusters (expected for current phase):

- `qa_*` (`--qa_tokens`, `--qa_layout`)
- dual-mask family (`--dual_mask_*`)
- auxiliary family (`--aux_b2_*`, `--aux_b3_*`, `--aux_e_weight`, `--d_hard_*`)
- teacher/cycle family (`--teacher_*`, `--cycle_*`)
- sequence/event/ray-order (`--sequence_mode`, `--event_order_mode`, `--ray_order_mode`)
- objective variants (`--objective`, `--mask_ratio`)
- token-length controls (`--max_len`)

Interpretation:

- This diff confirms current Patch baseline is intentionally narrower and does not yet include Query-specific objective/layout branches.
- Porting work should reduce this gap only through milestone-based additions, not by ad-hoc launcher drift.
