# Patchified Transformer scratch baseline (Step 1)

目的: **NEPA の工夫（順序 / Q/A / DualMask etc）を議論する前に**、
Point-MAE/Point-BERT 等で言う *"Transformer (rand)"* 相当の **基礎体力** を確認する。

このベースラインは以下を満たす:

- 1024 点入力
- FPS + kNN + mini-PointNet による **patchify** (例: 64 patch × 32 points)
- Transformer を **scratch** で分類学習（ラベルのみ）
- decoder / MAE などは入れない（貢献を散らさない）

## 実装

- `nepa3d/models/point_patch_embed.py`
  - FPS + kNN + mini-PointNet で patch token を作る
- `nepa3d/models/patch_classifier.py`
  - patchify + Transformer (+ pooling) + linear head
- `nepa3d/train/finetune_patch_cls.py`
  - scratch / finetune で分類を回す（NEPA token 系とは独立）

## 実行例

ScanObjectNN (variant cache, example `pb_t50_rs`):

```bash
scripts/finetune/patchcls_scanobjectnn_scratch.sh \
  CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v3_nonorm \
  RUN_NAME=patchcls_scan_scratch_pb_t50_rs \
  N_POINT=1024 NUM_GROUPS=64 GROUP_SIZE=32 \
  EPOCHS=300 BATCH=64
```

ScanObjectNN (serialization-based patch grouping, Morton -> chunk):

```bash
scripts/finetune/patchcls_scanobjectnn_scratch_serial.sh \
  CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v3_nonorm \
  RUN_NAME=patchcls_scan_serial_pb_t50_rs \
  N_POINT=1024 NUM_GROUPS=64 GROUP_SIZE=16 \
  EPOCHS=300 BATCH=64
```

ScanObjectNN (serialization-based patch grouping, PTv3 trans variant):

```bash
scripts/finetune/patchcls_scanobjectnn_scratch_serial_trans.sh \
  CACHE_ROOT=data/scanobjectnn_pb_t50_rs_v3_nonorm \
  RUN_NAME=patchcls_scan_serial_ztrans_pb_t50_rs \
  N_POINT=1024 NUM_GROUPS=64 GROUP_SIZE=16 \
  EPOCHS=300 BATCH=64
```

ModelNet40:

```bash
scripts/finetune/patchcls_modelnet40_scratch.sh \
  CACHE_ROOT=data/modelnet40_cache_v2 \
  RUN_NAME=patchcls_modelnet_scratch \
  N_POINT=1024 NUM_GROUPS=64 GROUP_SIZE=32
```

## 注意

- まずは `--is_causal 0` (bidirectional) で **Transformer baseline の土俵** に乗る。
- patch grouping backend は `PATCH_EMBED=fps_knn`（既定）と `PATCH_EMBED=serial` を切り替え可能。
- serial の順序は `SERIAL_ORDER` で切り替え可能（`morton`, `morton_trans`, `z`, `z-trans`, `random`, `identity`）。
- Point-MAE分類の標準は `64 patches x 32 points`（`N_POINT=1024, NUM_GROUPS=64, GROUP_SIZE=32`）。
  ただし現行 serial 実装は「非重複 contiguous chunk」のため、
  `N_POINT=1024` で `NUM_GROUPS=64` を維持するには `GROUP_SIZE=16` を使う運用になる。
  `64x32` を serial で厳密再現するには、overlap window（例: `group_size=32, stride=16`）の実装追加が必要。
- その後、NEPA の story に寄せて `--is_causal 1` (causal) の ablation を追加する。
- `mc_eval_k_test` / `aug_eval` は必要なときだけ（Point-MAE Table の比較では通常 no-vote）。
- Scan benchmark では variant-split cache (`obj_bg` / `obj_only` / `pb_t50_rs`) を使い、
  `main_split_v2` は使わない。
- 現在の `patchcls_scanobjectnn_scratch.sh` 既定は `VAL_SPLIT_MODE=file`（Point-MAE strict 方針に合わせた train 内 val 分割）。
- `test-as-val` は既定で無効（必要時のみ明示指定で有効化）。
- `scripts/sanity/pointmae_scan_scratch_qf.sh` も既定で `NO_TEST_AS_VAL=1`。
  Point-MAE scratch 実行時は `subset=train` から層化分割で `subset=val` を作る。
- 現行ポリシーでは `scanobjectnn_*_v2` (uni-scale) は使わない。
  `scanobjectnn_*_v3_nonorm` を使う（必要時のみ `ALLOW_SCAN_UNISCALE_V2=1` で明示 override）。

## Latest results (2026-02-27)

### v3_nonorm (`npz`) scratch baseline

Run set:

- `logs/sanity/patchcls/patchcls_scan3_scratch_v3nonorm_20260227_055706`
- jobs: `98390` (`pb_t50_rs`), `98391` (`obj_bg`), `98392` (`obj_only`) all `Exit_status=0`

| variant | val_acc (ep300) | test_acc |
|---|---:|---:|
| `obj_bg` | 0.7803 | 0.7108 |
| `obj_only` | 0.8161 | 0.7676 |
| `pb_t50_rs` | 0.8801 | 0.6877 |

### Official H5 (`scan_h5`) reference run

Run set:

- `logs/sanity/patchcls/patchcls_scan3_scratch_h5_20260227_061351`
- jobs: `98407` (`pb_t50_rs`), `98408` (`obj_bg`), `98409` (`obj_only`) all `Exit_status=0`

| variant | val_acc (ep300) | test_acc |
|---|---:|---:|
| `obj_bg` | 0.7783 | 0.7074 |
| `obj_only` | 0.7957 | 0.7797 |
| `pb_t50_rs` | 0.8815 | 0.6731 |

### Obj-only random-seed parity check

Run set:

- `logs/sanity/patchcls/patchcls_objonly_parity_randomseed_20260227_063811`
- jobs: `98413` (`obj_only_v3_random`), `98414` (`obj_only_h5_random`) both `Exit_status=0`

| setting | test_acc |
|---|---:|
| `obj_only_v3_random` | 0.7728 |
| `obj_only_h5_random` | 0.7745 |

Parity note:

- difference is small (`+0.0017`, h5 - v3).
- val split mode is still path-dependent (`group_scanobjectnn(auto)` for `npz`, `stratified_label(h5)` for `scan_h5`).

### Obj-only parity (Point-MAE split unified)

Run set:

- `logs/sanity/patchcls/patchcls_objonly_parity_pointmae_20260227_070409`
- jobs: `98427` (`obj_only_v3_pointmae`), `98428` (`obj_only_h5_pointmae`) both `Exit_status=0`

| setting | test_acc |
|---|---:|
| `obj_only_v3_pointmae` | 0.7900 |
| `obj_only_h5_pointmae` | 0.7900 |

Unified-split note:

- both runs used `val_split_mode=pointmae(test-as-val)`.
- parity is exact at reported precision (`delta=0.0000`).

### DDP (4GPU) support + reproducibility check

Code updates:

- `scripts/finetune/patchcls_scanobjectnn_scratch.sh`
  - added `NPROC_PER_NODE` and `BATCH_MODE` (`global` / `per_proc`)
  - `NPROC_PER_NODE>1` launches `torch.distributed.run` (torchrun)
- `nepa3d/train/finetune_patch_cls.py`
  - added `--batch_mode`
  - startup log now prints `world_size`, `batch_mode`, `batch_effective`

#### 300-epoch DDP run (obj_only, pointmae split)

Run set:

- `logs/sanity/patchcls/patchcls_objonly_parity_pointmae_ddp4_20260227_143511`
- jobs: `98847` (`obj_only_v3_ddp4`), `98848` (`obj_only_h5_ddp4`) both `Exit_status=0`

| setting | world_size | test_acc |
|---|---:|---:|
| `obj_only_v3_ddp4` | 4 | 0.7750 |
| `obj_only_h5_ddp4` | 4 | 0.7750 |

#### Same-ckpt eval parity (single vs ddp4)

Run set:

- `logs/sanity/patchcls/patchcls_objonly_eval_repro_ddp_vs_single_cleanup_20260227_144836`
- jobs:
  - single: `98876` (`v3`), `98878` (`h5`) -> `Exit_status=0`
  - ddp4: `98877` (`v3`), `98879` (`h5`) -> output reached `TEST`, then manually stopped to avoid queue hold (`Exit_status=271`)

| setting | test_acc |
|---|---:|
| `v3_single` | 0.7625 |
| `v3_ddp4` | 0.7625 |
| `h5_single` | 0.7625 |
| `h5_ddp4` | 0.7625 |

Repro conclusion:

- for the same checkpoint and eval settings, single and ddp4 are identical at reported precision.

Interpretation note:

- `0.7625` in this section is **eval-only parity** (same ckpt loaded, `EPOCHS=0`), not a new scratch-training result.
- scratch-training headline for obj_only remains the 300-epoch run in
  `patchcls_objonly_parity_pointmae_20260227_070409` (`test_acc=0.7900`).
- `0.7900` (single) vs `0.7750` (ddp4) is **not** enough evidence that single is intrinsically better.
  this pair differs in full training trajectory (batch partition/order, distributed RNG stream, optimizer step path).
  practical conclusion so far is:
  - inference path parity is confirmed (same ckpt -> same test acc in single/ddp4).
  - training-path sensitivity exists; use multi-seed mean/std before claiming single-vs-ddp superiority.

### DDP batch-size comparison run (completed)

Run set:

- `logs/sanity/patchcls/patchcls_objonly_scratch_ddp4_bs64_vs_bs128_20260227_164159`
- jobs:
  - `98993` (`pcs_bs64`) : `BATCH=64`, `NPROC_PER_NODE=4`, `BATCH_MODE=global`, `EPOCHS=300`, `Exit_status=0`
  - `98994` (`pcs_bs128`) : `BATCH=128`, `NPROC_PER_NODE=4`, `BATCH_MODE=global`, `EPOCHS=300`, `Exit_status=0`

Final metrics:

| setting | test_acc | test_loss |
|---|---:|---:|
| `ddp4_bs64` | 0.8003 | 1.4273 |
| `ddp4_bs128` | 0.7952 | 1.3677 |

Quick read:

- under this DDP4 setup, `BATCH=64` was slightly higher (`+0.0051`) than `BATCH=128`.
- this is one run-pair result, not yet a multi-seed conclusion.

### Point-MAE scratch ckpt test-accuracy extraction

Test-from-ckpt script:

- `scripts/sanity/pointmae_scan_test_from_ckpt_qf.sh`

Clean test run set:

- `logs/sanity/pointmae_scratch_tests/pointmae_scan3_scratch_test_from_ckpt_fixlog_20260227_164410`
- jobs:
  - `98998` (`pm_test_objbg2`) `Exit_status=0`
  - `98999` (`pm_test_objonly2`) `Exit_status=0`

Test metrics:

| setting | source ckpt | TEST acc |
|---|---|---:|
| `Point-MAE scratch / obj_bg` | `Point-MAE/experiments/finetune_scan_objbg/cfgs/pm_obj_bg_pointmae_scan3_scratch_stdbs32_20260227_142324/ckpt-best.pth` | 0.867470 |
| `Point-MAE scratch / obj_only` | `Point-MAE/experiments/finetune_scan_objonly/cfgs/pm_obj_only_pointmae_scan3_scratch_stdbs32_20260227_142324/ckpt-best.pth` | 0.864028 |

Pending:

- `pb_t50_rs` test is already dependency-submitted as `98997` (`pm_test_pbt50`, `afterok:98824`).
