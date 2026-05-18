# PosetNEPA mask/order pre-flight pack

This pack replaces the earlier pre-flight design. The earlier version tried to use `copy_win` under the default PointGPT mask-on setting; that is not a clean test of the PosetNEPA motivation because PointGPT masking already suppresses adjacent-token copy shortcuts.

## What this pack tests

The pack is designed to answer three precise questions before implementing full frontier-level PosetNEPA:

1. **Kill Test 0: Is copy-like shortcut real in naive mask-off latent AR?**
   - Run pointNEPA with `mask_ratio=0.0` and the existing order.
   - Measure `copy_win`, `gap = cos_tgt - cos_prev`, and downstream PB-T50-RS.
   - If mask-off has low `copy_win`, copy shortcut is not the right motivation.

2. **Does ordering matter once grouping is fixed?**
   - Keep `group_mode=fps_knn` fixed.
   - Change only `order_mode` across `simplified_morton`, `fixed_random`, `axis_x`, `radial`, `bfs_shell`, `geodesic_shell`, `diffusion_shell`.
   - This avoids the trivial “bad tokenization gives bad representation” critique.

3. **Does order interact with masking?**
   - Sweep `mask_ratio in {0.0, 0.3, 0.7}`.
   - If order only matters with mask-off, the story is shortcut reduction.
   - If order matters with mask-on as well, the story is stronger: geometry-induced filtration solves something masking does not.

## What this pack intentionally does NOT test yet

- No full frontier set-level PosetNEPA loss.
- No vit-shift; it is not central to the initial hypothesis.
- No multi-seed sweep; single-run is aligned with point-cloud SSL reporting conventions.
- No grouping ablation as a main claim. Grouping is fixed by default.

## Active Interpretation Notes

- Research framing, Mirai connection, PointGPT/Point-MAE/PCP-MAE boundary, and
  Sinkhorn/OT loss positioning:
  `docs/research_framing_active.md`
- Literature and protocol caveats for the current Stage 1 run:
  `docs/literature_protocol_notes_active.md`
- Post-Lite skip-k / center leakage decision plan:
  `docs/skip_center_decision_plan_active.md`
- Short Q1-Q3 paper framing:
  `docs/q1_q3_revised.md`

## Local fixes in this repo

This checked-in copy adds a few guardrails beyond the original zip:

- Stage scripts pin and export one `RUN_TAG`, so fine-tune resolves the exact
  pretrain checkpoints from the same stage.
- `DRY_RUN=1` works through pretrain and fine-tune command construction.
- fine-tune checkpoints/logs are summarized into
  `generated/<RUN_TAG>/finetune_results.csv`.
- `05_summarize_pf3.py` consumes downstream PB_T50_RS / obj_bg / obj_only
  values instead of reporting pretext diagnostics only.
- `04_extract_pretext_diag.py` reports final-epoch averaged pretext
  diagnostics in the legacy metric columns and includes explicit `*_mean`,
  `n_diag`, `epoch`, and `*_last` CSV columns.
- `07_verify_chain.py` checks that generated configs really carry the intended
  `order_mode`, `group_mode`, and pretrain `mask_ratio`.
- stage outputs are written under `generated/<RUN_TAG>/`, so Stage 1/2/3 runs
  do not overwrite each other's manifests or summaries.
- Stage 2 includes `morton`, `fixed_random`, and `farthest_greedy` controls,
  so stochastic `random` is no longer needed as the main random-order evidence.

## Minimal first run

From the root of `KohsukeIde/3D-NEPA`:

```bash
# If starting from the zip only:
# unzip posetnepa_mask_order_preflight.zip -d .
#
# In this repo the checked-in directory already includes local fixes, so do not
# overwrite it with the zip unless you intend to replace those fixes.
python3 posetnepa_mask_order_preflight/scripts/00_patch_pointgpt_order_modes.py --apply
python3 posetnepa_mask_order_preflight/scripts/06_verify_patch.py

# Generate mask/order configs. Default is mask-off first.
python3 posetnepa_mask_order_preflight/scripts/01_make_mask_order_configs.py \
  --orders simplified_morton,fixed_random,diffusion_shell \
  --masks 0.0,0.7 \
  --max-epoch 30 \
  --ft-max-epoch 50 \
  --manifest posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json \
  --out-dir PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/manual_stage1

# Pretrain short runs on 2 GPUs.
MANIFEST="$PWD/posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json" \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 \
  bash posetnepa_mask_order_preflight/scripts/02_pretrain_mask_order_matrix.sh

# Fine-tune PB-T50-RS only.
MANIFEST="$PWD/posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json" \
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 FT_SPLITS=hardest \
  bash posetnepa_mask_order_preflight/scripts/03_finetune_scan_pb_t50.sh

# Summarize logs and go/no-go.
python3 posetnepa_mask_order_preflight/scripts/04_extract_pretext_diag.py \
  --manifest posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json \
  --out-csv posetnepa_mask_order_preflight/generated/manual_stage1/pretext_diag.csv \
  --out-md posetnepa_mask_order_preflight/generated/manual_stage1/pretext_diag.md

python3 posetnepa_mask_order_preflight/scripts/04b_extract_finetune_results.py \
  --manifest posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json \
  --splits hardest \
  --out-csv posetnepa_mask_order_preflight/generated/manual_stage1/finetune_results.csv \
  --out-md posetnepa_mask_order_preflight/generated/manual_stage1/finetune_results.md

python3 posetnepa_mask_order_preflight/scripts/05_summarize_pf3.py \
  --manifest posetnepa_mask_order_preflight/generated/manual_stage1/manifest.json \
  --diag-csv posetnepa_mask_order_preflight/generated/manual_stage1/pretext_diag.csv \
  --results-csv posetnepa_mask_order_preflight/generated/manual_stage1/finetune_results.csv \
  --out-md posetnepa_mask_order_preflight/generated/manual_stage1/pf3_summary.md
```

The same sequence is wrapped as:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 \
  ORDERS=simplified_morton,fixed_random,diffusion_shell \
  MASKS=0.0,0.7 \
  MAX_EPOCH=30 \
  FT_MAX_EPOCH=50 \
  FT_SPLITS=hardest \
  bash posetnepa_mask_order_preflight/scripts/run_stage1_local.sh
```

Use `DRY_RUN=1` to print the full chain without launching training.
Stage summaries are written to `posetnepa_mask_order_preflight/generated/<RUN_TAG>/`.

## Representation Probes

Full fine-tuning can erase the very order/filtration differences this pack is
trying to diagnose. The follow-up chain therefore has two readout controls:

- `freeze_backbone` fine-tune: freezes the pretrained PointTransformer and
  trains the repo's normal classification head.
- frozen linear probe: extracts frozen `PointTransformer` features and trains
  only one linear classifier on cached features.

The linear probe can be queued after the current post-stage follow-up:

```bash
POST_WAIT_PID=<post_followup_pid> \
STAGE1_TAG=stage1_20260513_025903 \
PROBE_EPOCHS=200 PROBE_SEEDS=0 SPLITS=hardest CUDA_VISIBLE_DEVICES=0 \
  bash posetnepa_mask_order_preflight/scripts/18_run_rep_probe_after_post.sh
```

For a fresh post-stage chain, it can also be run inline:

```bash
RUN_LINEAR_PROBE=1 LINEAR_PROBE_SPLITS=hardest \
  bash posetnepa_mask_order_preflight/scripts/16_run_post_stage1_followup.sh
```

It writes:

```text
posetnepa_mask_order_preflight/generated/rep_probe_after_<STAGE1_TAG>/linear_probe_scanobjectnn.md
```

## Skip-k / Center Leakage Chain

After PosetNEPA-Lite, do not jump straight to frontier-level Core. The next
diagnostic chain tests local-continuity dependence and position side-channels:

```bash
USE_WANDB=0 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash posetnepa_mask_order_preflight/scripts/20_run_skip_center_chain.sh
```

Default variants are `skip2`, `skip4`, `skip8`, `poszero`, `posshuffle`,
`centeraux`, and `skip4_poszero`, all on `simplified_morton`, `mask=0.0`,
`group_mode=fps_knn`.

Outputs are written under:

```text
posetnepa_mask_order_preflight/generated/<RUN_TAG>/
```

The completed local run `skipcenter_20260517_213653` is summarized in
`docs/skip_center_decision_plan_active.md`. The short read is:

- `skip2` remains competitive, so the shortcut is not only immediate
  previous-token copy.
- `skip4` and `skip8` degrade, so local-neighborhood continuity is important.
- zero/shuffled position controls show position and center side-channels are
  part of the current objective.
- pretrain curves decrease for all rows, but raw loss is not comparable across
  variants with different targets or side information.
- the recommended next method test is frontier-set PosetNEPA-Core, not another
  total-order Lite sweep.

## Full pre-flight chain

Stage 1 is the required diagnostic lock. Stage 2/3 are intentionally opt-in
because they are expensive.

```bash
# Stage 1 only.
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 \
  bash posetnepa_mask_order_preflight/scripts/run_full_preflight_chain.sh

# Wider order sweep after Stage 1 is promising.
RUN_STAGE2=1 CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 \
  bash posetnepa_mask_order_preflight/scripts/run_full_preflight_chain.sh

# Confirmation sweep after Stage 2 selects candidate geometry orders.
RUN_STAGE1=0 RUN_STAGE2=0 RUN_STAGE3=1 \
  ORDERS=simplified_morton,morton,fixed_random,diffusion_shell,geodesic_shell \
  MASKS=0.0,0.7 \
  FT_SPLITS=hardest,objbg,objonly \
  MAX_EPOCH=100 FT_MAX_EPOCH=100 \
  CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 USE_WANDB=0 \
  bash posetnepa_mask_order_preflight/scripts/run_full_preflight_chain.sh
```

## Interpretation

- `mask=0.0, simplified_morton` is the naive 1D latent-AR baseline.
- `mask=0.7, simplified_morton` is the PointGPT/pointNEPA-style masked condition.
- `mask=0.0, diffusion_shell` asks whether geometry-induced order reduces copy shortcut without relying on masking.
- `mask=0.7, diffusion_shell` asks whether geometry-induced order is complementary to masking.

The first publishable story should be chosen only after these four conditions are compared.
