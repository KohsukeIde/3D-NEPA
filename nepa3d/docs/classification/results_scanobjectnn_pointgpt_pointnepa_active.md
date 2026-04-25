# ScanObjectNN PointGPT / pointNEPA Sidecar Results (Active)

Snapshot time: `2026-04-25 JST` (includes the 2026-04-22 local rebuild chain, the 2026-04-24 order-randomized downstream partial, and the 2026-04-25 order-randomized readout/support audits)

## 2026-04-22 local rebuild status

Rebuild intent:

- Recreate the PointGPT / ScanObjectNN object-side environment on this machine without relying on the previously used external host.

Environment and data status:

- Local venv rebuilt at `3D-NEPA/.venv-pointgpt`.
- `ScanObjectNN` official HDF5 files are present under `3D-NEPA/PointGPT/data/ScanObjectNN/h5_files/...`.
- `ShapeNet55` processed point clouds are present under `3D-NEPA/PointGPT/data/ShapeNet55-34/shapenet_pc`.
- `ShapeNet-55/train.txt` and `test.txt` were restored locally from the Point-BERT canonical split because the Google Drive archive only contained `ShapeNet55/shapenet_pc`.
- Official PointGPT-S checkpoints are present under `3D-NEPA/PointGPT/checkpoints/official/`.

Compatibility fix applied during rebuild:

- `PointGPT/tools/builder.py` now handles both `cycle_decay` and `decay_rate` signatures of `timm.scheduler.CosineLRScheduler`. This was required for the current local environment and unblocked both pretrain and finetune.

Smoke verification completed on ABCI:

- no-mask pretrain smoke:
  - experiment: `3D-NEPA/PointGPT/experiments/pretrain_nomask_smoke/PointGPT-S/pgpt_s_nomask_smoke_e1_fix1`
  - result: one-epoch smoke completed and saved `ckpt-last.pth`
- ScanObjectNN `obj_bg` finetune smoke from official PointGPT-S pretrain:
  - experiment: `3D-NEPA/PointGPT/experiments/finetune_scan_objbg_smoke/PointGPT-S/pgpt_s_objbg_smoke_e2_fix1`
  - result: two-epoch smoke completed and saved both `ckpt-best.pth` and `ckpt-last.pth`
  - validation trace: epoch-2 `acc = 49.2255`

Completed no-mask rebuild chain:

- full no-mask PointGPT-S pretrain (`300` epochs) completed:
  - former job: `134204.qjcm`
  - experiment: `3D-NEPA/PointGPT/experiments/pretrain_nomask/PointGPT-S/pgpt_s_nomask_e300`
  - artifacts: `ckpt-last.pth`, `ckpt-epoch-300.pth`
- dependent ScanObjectNN finetunes from that no-mask pretrain also completed:
  - former `obj_bg`: `134205.qjcm`
  - former `objonly`: `134206.qjcm`
  - former `hardest`: `134207.qjcm`

Best validation accuracy from the local rebuild chain:

- `obj_bg`: `89.2123`
- `objonly`: `89.3836`
- `hardest`: `84.1886`

This means the local PointGPT no-mask rebuild is no longer pending. It now
serves as a recorded official-like no-mask object-side comparator, separate
from the earlier pointNEPA-S mask-on/mask-off/vit-shift rows.

Pretext-side evidence from the completed no-mask pretrain log:

- source log:
  - `3D-NEPA/PointGPT/experiments/pretrain_nomask/PointGPT-S/pgpt_s_nomask_e300/20260422_021328.log`
- runtime:
  - start: `2026-04-22 02:13:28 JST`
  - end: `2026-04-22 07:22:17 JST`
  - wall-time: about `5.1 h` on `4 GPU`
- epoch-end total loss snapshots:
  - epoch `0`: `356.9453`
  - epoch `1`: `88.1461`
  - epoch `2`: `63.5821`
  - epoch `5`: `51.7855`
  - epoch `10`: `49.5130`
  - epoch `25`: `42.5258`
  - epoch `50`: `37.6923`
  - epoch `100`: `35.3499`
  - epoch `150`: `34.4119`
  - epoch `200`: `33.8658`
  - epoch `250`: `33.4264`
  - epoch `299`: `33.3267`
  - epoch `300`: `33.3550`
- epoch-300 batch diagnostics (mean over the logged diagnostics at epoch `300`):
  - `loss_main`: `0.03319`
  - `recon_cd_l1`: `0.02944`
  - `recon_cd_l2`: `0.00374`

Interpretation:

- the no-mask pretrain is not a null run; the pretext objective moves strongly
  early and continues to improve through the end of training.
- this gives the object-side no-mask row an explicit pretext-side movement
  anchor, analogous to the scene-side “pretext reacts, downstream only
  partially follows” framing.

Scope:

- This page tracks local PointGPT / pointNEPA sidecar experiments on ScanObjectNN.
- This page is **not** the canonical PatchNEPA benchmark headline ledger.
- Use `nepa3d/docs/patch_nepa/benchmark_scanobjectnn_variant.md` for current PatchNEPA headline tables.

Definitions used on this page:

- `pointNEPA`
  - PointGPT scaffold with `nepa_cosine` pretrain loss only
  - no decoder / Chamfer pretrain loss
  - fine-tune with `cls-only` (`ft_recon_weight=0`)
- `pointNEPA-S (mask-on)`
  - `PointGPT-S` + `nepa_cosine` + `mask_ratio=0.7` + `cls-only FT`
- `pointNEPA-S (mask-off)`
  - `PointGPT-S` + `nepa_cosine` + `mask_ratio=0.0` + `cls-only FT`
- `pointNEPA-S (vit-shift)`
  - `PointGPT-S` variant that keeps the PointGPT causal extractor but moves the one-step shift to the loss side to match `models/vit_nepa` more closely

Primary source summaries:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_b_20260312_summary.md`
- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_s_20260318_summary.md`
- `logs/local/pointgpt_ft_recipe_matrix_2x2/pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointnepa_s_maskoff_20260403_212525_summary.md`
- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointnepa_s_vitshift_maskoff_20260403_221453_summary.md`

## Official checkpoint protocol compare

### PointGPT-B official checkpoint

Source summary:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_b_20260312_summary.md`

Checkpoint:

- `PointGPT/checkpoints/official/pointgpt_b_post_pretrain_official.pth`

| Variant | `test-as-val` test_acc_plain | `strict(train->val)` test_acc_plain |
|---|---:|---:|
| `obj_bg` | `96.7298` | `96.7298` |
| `objonly` | `94.4923` | `94.4923` |
| `hardest` | `91.6031` | `90.5968` |

### PointGPT-S official checkpoint

Source summary:

- `logs/local/pointgpt_protocol_compare/pointgpt_protocol_compare_official_s_20260318_summary.md`

Checkpoint:

- `PointGPT/checkpoints/official/pointgpt_s_pretrain_official.pth`

| Variant | `test-as-val` test_acc_plain | `strict(train->val)` test_acc_plain |
|---|---:|---:|
| `obj_bg` | `90.0172` | `89.8451` |
| `objonly` | `91.222` | `87.2633` |
| `hardest` | `86.086` | `85.6697` |

## Local PointGPT-B objective x FT recipe matrix (complete)

Master summary:

- `logs/local/pointgpt_ft_recipe_matrix_2x2/pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md`

| Pretrain source | FT recipe | `obj_bg` best_acc | `objonly` best_acc | `hardest` best_acc | Arm summary |
|---|---|---:|---:|---:|---|
| `nepa_cosine` | `cls-only` | `89.17526245117188` | `89.0034408569336` | `84.76751708984375` | `logs/local/pointgpt_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `nepa_cosine` | `PointGPT FT (cls+recon)` | `89.86254119873047` | `89.5188980102539` | `84.6634292602539` | `logs/local/pointgpt_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `cdl12` | `cls-only` | `88.83161926269531` | `89.0034408569336` | `84.21234893798828` | `logs/local/pointgpt_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |
| `cdl12` | `PointGPT FT (cls+recon)` | `90.03436279296875` | `89.17526245117188` | `85.14920043945312` | `logs/local/pointgpt_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_ft_recipe_matrix_2x2_20260311_153835_summary.md` |

## Local PointGPT-S objective x FT recipe matrix (complete)

Run tag:

- `pointgpt_s_ft_recipe_matrix_2x2_20260318`

Completed arm summaries:

- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`
- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/cdl12_pointgptft_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`

| Pretrain source | FT recipe | `obj_bg` best_acc | `objonly` best_acc | `hardest` best_acc | Status |
|---|---|---:|---:|---:|---|
| `nepa_cosine` | `cls-only` | `90.03436279296875` | `90.20618438720703` | `85.39208984375` | complete |
| `nepa_cosine` | `PointGPT FT (cls+recon)` | `91.06529235839844` | `90.03436279296875` | `85.46147918701172` | complete |
| `cdl12` | `cls-only` | `90.03436279296875` | `89.5188980102539` | `83.83067321777344` | complete |
| `cdl12` | `PointGPT FT (cls+recon)` | `91.58075714111328` | `89.86254119873047` | `84.62872314453125` | complete |

## pointNEPA-S readout

### pointNEPA-S (mask-on) baseline

This is the already-completed `PointGPT-S + nepa_cosine + mask_ratio=0.7 + cls-only FT` run.

Source summary:

- `logs/local/pointgpt_s_ft_recipe_matrix_2x2/nepa_cosine_clsonly_pointgpt_s_ft_recipe_matrix_2x2_20260318_summary.md`

Results:

- `obj_bg`: `90.03436279296875`
- `objonly`: `90.20618438720703`
- `hardest`: `85.39208984375`

### pointNEPA-S (mask-off, current-shift)

Config:

- `PointGPT/cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0_nomask.yaml`

Runtime metadata:

- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525.meta.env`

Experiment path:

- `PointGPT/experiments/pretrain_nepa_cosine_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525`

Pretrain log:

- `PointGPT/experiments/pretrain_nepa_cosine_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_shapenet_cache_v0_nomask_pointnepa_s_maskoff_20260403_212525/20260403_212535.log`

Source summary:

- `logs/local/pointgpt_s_pointnepa_mask_ablation/pointnepa_s_maskoff_20260403_212525_summary.md`

Results:

- `obj_bg`: `90.37801361083984`
- `objonly`: `90.37801361083984`
- `hardest`: `84.52462768554688`

Relative to the mask-on baseline:

- `obj_bg`: `+0.34365081787109`
- `objonly`: `+0.17182922363281`
- `hardest`: `-0.86746215820312`

### pointNEPA-S (vit-shift, mask-off)

Intent:

- keep the PointGPT patch encoder and causal extractor
- remove the input-side one-step shift
- move the one-step shift into the loss to match `models/vit_nepa` more closely

Config:

- `PointGPT/cfgs/PointGPT-S/pretrain_nepa_cosine_vitshift_shapenet_cache_v0_nomask.yaml`

Runtime metadata:

- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointgpt_s_nepa_cosine_vitshift_shapenet_cache_v0_nomask_pointnepa_s_vitshift_maskoff_20260403_221453.meta.env`

Experiment path:

- `PointGPT/experiments/pretrain_nepa_cosine_vitshift_shapenet_cache_v0_nomask/PointGPT-S/pointgpt_s_nepa_cosine_vitshift_shapenet_cache_v0_nomask_pointnepa_s_vitshift_maskoff_20260403_221453`

Source summary:

- `logs/local/pointgpt_s_pointnepa_vitshift_ablation/pointnepa_s_vitshift_maskoff_20260403_221453_summary.md`

Results:

- `obj_bg`: `90.72164916992188`
- `objonly`: `89.0034408569336`
- `hardest`: `84.21234893798828`

Relative to the mask-off current-shift arm:

- `obj_bg`: `+0.34363555908204`
- `objonly`: `-1.37457275390624`
- `hardest`: `-0.31227874755860`

## Current readout

- `PointGPT-S + nepa_cosine` is operational on ScanObjectNN; the `mask-on` `cls-only` arm is already at:
  - `obj_bg=90.03436279296875`
  - `objonly=90.20618438720703`
  - `hardest=85.39208984375`
- `pointNEPA-S (mask-off, current-shift)` is complete:
  - `obj_bg=90.37801361083984`
  - `objonly=90.37801361083984`
  - `hardest=84.52462768554688`
- `pointNEPA-S (vit-shift, mask-off)` is complete:
  - `obj_bg=90.72164916992188`
  - `objonly=89.0034408569336`
  - `hardest=84.21234893798828`
- `PointGPT-S` official checkpoint protocol compare is complete and recorded here.
- `PointGPT-B` local 2x2 matrix is complete and recorded here.
- `PointGPT-S` local 2x2 matrix is complete; `cdl12 x PointGPT FT` reached `obj_bg=91.58075714111328`, `objonly=89.86254119873047`, `hardest=84.62872314453125`.
- Do not treat this page as the PatchNEPA benchmark headline; use it as the active PointGPT / pointNEPA sidecar ledger.

## PointGPT-S official-vs-no-mask object audit (`obj_bg`)

Full-object readout audit completed:

- official downstream checkpoint:
  - `PointGPT/checkpoints/official/pointgpt_s_scan_objbg_official.pth`
  - summary:
    - `3D-NEPA/results/ptgpt_readout_official_objbg_full.md`
- local no-mask downstream checkpoint:
  - `PointGPT/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_objbg_e300/ckpt-best.pth`
  - summary:
    - `3D-NEPA/results/ptgpt_readout_nomask_objbg_full.md`

Readout-decomposition headline:

| row | top1 acc | top2 hit | top5 hit | hardest pair | pair direct acc | pair probe bal acc |
|---|---:|---:|---:|---|---:|---:|
| official `obj_bg` | `0.9105` | `0.9776` | `0.9948` | `bag -> box` | `0.7778` | `0.8466` |
| no-mask `obj_bg` | `0.8726` | `0.9466` | `0.9914` | `pillow -> bag` | `0.7105` | `0.8754` |

Object-side readout interpretation:

- both rows retain a large gap between top-1 and top-2 / top-5, so
  “information exists but the fixed multiclass decision does not fully realize
  it” is no longer scene-only.
- the hardest pair is object-family specific rather than wall-adjacent, which
  is useful: the object-side audit is showing the same kind of actionability
  gap through a different confusion geometry.

## PointGPT-S official-vs-no-mask support stress (`obj_bg`)

Full-object support stress completed:

- official summary:
  - `3D-NEPA/results/ptgpt_stress_official_objbg_full.md`
- no-mask summary:
  - `3D-NEPA/results/ptgpt_stress_nomask_objbg_full.md`

Overall accuracy:

| row | clean | random keep50 | random keep20 | random keep10 | structured keep50 | structured keep20 | structured keep10 | xyz-zero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official `obj_bg` | `0.9122` | `0.8950` | `0.4269` | `0.2702` | `0.7401` | `0.2272` | `0.1308` | `0.0929` |
| no-mask `obj_bg` | `0.8744` | `0.8778` | `0.5215` | `0.2341` | `0.6730` | `0.2186` | `0.1153` | `0.0929` |

Stress interpretation:

- object-side support stress now mirrors the scene-side logic:
  - mild random keep is a weak stress;
  - structured removal is much more decisive;
  - `xyz_zero` collapses to near-chance.
- the no-mask row is not simply “more robust everywhere”:
  - it is stronger than the official baseline at `random_keep20`,
  - but weaker under `structured_keep50`,
  - so the effect is support-shape dependent rather than a uniform gain.

## ShapeNetPart key rows

The local ShapeNetPart path is now unblocked end-to-end and the key
official-vs-no-mask rows are complete.

Compatibility fix applied:

- `PointGPT/segmentation/main.py`
  - replaced the deprecated `np.float` alias with builtin `float`

One-epoch smoke pair completed successfully:

- official PointGPT-S pretrain checkpoint:
  - log dir:
    - `PointGPT/segmentation/log/part_seg/pgpt_s_partseg_smoke_official_fix6`
  - epoch-1 test:
    - accuracy `0.7934`
    - class-avg mIoU `0.4172`
    - instance-avg mIoU `0.5991`
- local no-mask PointGPT-S pretrain checkpoint:
  - log dir:
    - `PointGPT/segmentation/log/part_seg/pgpt_s_partseg_smoke_nomask_fix6`
  - epoch-1 test:
    - accuracy `0.7747`
    - class-avg mIoU `0.4035`
    - instance-avg mIoU `0.5818`

Smoke interpretation:

- ShapeNetPart is no longer blocked by environment / import / NumPy issues.
- The official-vs-no-mask key rows can now be treated as real downstream runs
  rather than speculative placeholders.

Full 300-epoch key rows:

- official PointGPT-S pretrain checkpoint:
  - log dir:
    - `PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300`
  - best:
    - accuracy `0.94500`
    - class-avg mIoU `0.83608`
    - instance-avg mIoU `0.85656`
  - final epoch-300:
    - accuracy `0.944541`
    - class-avg mIoU `0.832223`
    - instance-avg mIoU `0.854844`
- local no-mask PointGPT-S pretrain checkpoint:
  - log dir:
    - `PointGPT/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_e300`
  - best:
    - accuracy `0.94475`
    - class-avg mIoU `0.83267`
    - instance-avg mIoU `0.85598`
  - final epoch-300:
    - accuracy `0.944164`
    - class-avg mIoU `0.830073`
    - instance-avg mIoU `0.854362`

ShapeNetPart interpretation:

- This closes the simplest “classification-only” objection against the
  object-side no-mask audit.
- The no-mask row is slightly below the official PointGPT-S row, but only
  marginally, so the weak-binding reading is not confined to global object
  classification alone.

## PointGPT-S no-mask + order-randomized pretrain

The `300`-epoch no-mask + order-randomized PointGPT-S pretrain is now complete.

- experiment:
  - `PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300`
- checkpoints:
  - `ckpt-last.pth`
  - `ckpt-epoch-300.pth`

Pretext-side movement from the pretrain log:

- epoch `0`: `356.8673`
- epoch `1`: `89.7123`
- epoch `10`: `54.1389`
- epoch `50`: `44.4492`
- epoch `100`: `41.8430`
- epoch `150`: `40.5978`
- epoch `200`: `39.8553`
- epoch `250`: `39.2888`
- epoch `299`: `39.1702`
- epoch `300`: `39.1911`

Interpretation:

- order randomization is not a null run; the pretext objective still optimizes
  strongly under the modified AR mechanism.
- downstream is partially available through the `obj_bg` fine-tune below, but
  the readout/stress follow-up audits for that checkpoint are still missing.

### Order-randomized no-mask `obj_bg` downstream fine-tune

The `obj_bg` fine-tune from the no-mask + order-randomized pretrain produced a
valid best checkpoint, but did not complete the planned `300` epochs.

- job:
  - former `134867.qjcm`
- experiment:
  - `PointGPT/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300`
- log:
  - `PointGPT/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300/20260424_005342.log`
- checkpoint:
  - `ckpt-best.pth`
- termination:
  - walltime limit at epoch `215`
  - PBS message: `job killed: walltime 18035 exceeded limit 18000`
- best validation accuracy observed before termination:
  - epoch `170`: `88.8699`

Object-side interpretation:

- the order-randomized no-mask downstream row is not a failed/null downstream
  run; it has a usable best checkpoint.
- it is slightly below the completed local no-mask rebuild `obj_bg` best
  (`89.2123`) and below the official PointGPT-S protocol-compare row
  (`~89.85-90.02`, depending on split convention), so the current evidence
  does not show an order-randomization downstream gain.
- because the run ended early, report it as a walltime-limited partial row
  unless a continuation is launched. The best checkpoint was saved at epoch
  `170` and remained the best observed checkpoint through epoch `215`, so the
  follow-up audits below use it as the early-stop checkpoint.

### Order-randomized no-mask `obj_bg` readout audit

Readout decomposition is now complete for the early-stop best checkpoint.

- result files:
  - `results/ptgpt_nomask_ordrand_objbg_readout_full.md`
  - `results/ptgpt_nomask_ordrand_objbg_readout_full.json`
- global:
  - top-1 accuracy `0.8726`
  - top-2 hit `0.9587`
  - top-5 hit `0.9931`
- hardest pair:
  - `pillow -> bag`
  - normalized confusion `0.2857`
  - direct pair top-1 accuracy `0.7368`
  - binary probe balanced accuracy `0.8221`

Interpretation:

- the order-randomized no-mask row retains high top-k coverage even though its
  top-1 accuracy is below the official PointGPT-S and completed no-mask rows.
- this supports the current conservative claim: causal order randomization is
  not a collapse-inducing perturbation for ScanObjectNN `obj_bg`, but it is
  more damaging than mask removal alone.

### Order-randomized no-mask `obj_bg` support stress

Support-stress audit is also complete for the same early-stop best checkpoint.

- result files:
  - `results/ptgpt_nomask_ordrand_objbg_stress_full.md`
  - `results/ptgpt_nomask_ordrand_objbg_stress_full.json`

| condition | accuracy |
|---|---:|
| clean | `0.8933` |
| random keep50 | `0.8692` |
| random keep20 | `0.4509` |
| random keep10 | `0.1945` |
| structured keep50 | `0.7177` |
| structured keep20 | `0.2427` |
| structured keep10 | `0.1325` |
| xyz zero | `0.0723` |

Interpretation:

- the order-randomized row follows the same object-side stress pattern as the
  official/no-mask rows: mild random drop is a weak stress, structured removal
  is much harsher, and xyz-zero collapses.
- this completes the minimal readout + support-stress audit for the
  order-randomized checkpoint. A full 300-epoch continuation remains optional,
  not required for the current early-stop interpretation.

## Current pending object-side runs

Current live queue:

- none observed on `2026-04-25 JST`.

Still missing:

- optional continuation of the order-randomized `obj_bg` fine-tune to the full
  `300` epochs, if a complete matched row is required

Recently completed:

- `134864.qjcm`
  - PointGPT-S `no-mask + order-randomized` full pretrain (`300 epoch`)
- `134873.qjcm`
  - ShapeNetPart full fine-tune from the official PointGPT-S pretrain
- `134874.qjcm`
  - ShapeNetPart full fine-tune from the local no-mask PointGPT-S pretrain
- `134867.qjcm`
  - PointGPT-S `no-mask + order-randomized` `obj_bg` fine-tune reached a valid
    best checkpoint at epoch `170` with validation accuracy `88.8699`, then
    stopped by walltime at epoch `215`.
