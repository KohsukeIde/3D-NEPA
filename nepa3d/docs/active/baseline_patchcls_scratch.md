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

ModelNet40:

```bash
scripts/finetune/patchcls_modelnet40_scratch.sh \
  CACHE_ROOT=data/modelnet40_cache_v2 \
  RUN_NAME=patchcls_modelnet_scratch \
  N_POINT=1024 NUM_GROUPS=64 GROUP_SIZE=32
```

## 注意

- まずは `--is_causal 0` (bidirectional) で **Transformer baseline の土俵** に乗る。
- その後、NEPA の story に寄せて `--is_causal 1` (causal) の ablation を追加する。
- `mc_eval_k_test` / `aug_eval` は必要なときだけ（Point-MAE Table の比較では通常 no-vote）。
- Scan benchmark では variant-split cache (`obj_bg` / `obj_only` / `pb_t50_rs`) を使い、
  `main_split_v2` は使わない。
- 現行ポリシーでは `scanobjectnn_*_v2` (uni-scale) は使わない。
  `scanobjectnn_*_v3_nonorm` を使う（必要時のみ `ALLOW_SCAN_UNISCALE_V2=1` で明示 override）。
