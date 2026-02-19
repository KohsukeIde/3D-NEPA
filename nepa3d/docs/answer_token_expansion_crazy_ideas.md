# Answer トークン拡張で completion を伸ばすアイデア集（+ 実装済み）

本プロジェクトの制約（貢献を散らさない / completion 主戦場 / 2048 対応）に合わせ、**目的関数（NEPA）自体は維持**したまま、
「Answer トークン（=教師信号側トークン）の情報量」を増やして表現学習を強くする方向のメモです。

---

## 0) 前提（今回の実装方針）

- **NEPA objective は変更しない**（予測対象の埋め込みは変えない）。
- 追加するのは **Answer トークンの feature**（入力側の表現）で、loss はそのまま。
- 既存 15 次元 feature を壊さず、**未使用スロットを type-dependent に再利用**する。
  - slot[11] は Ray では hit、Point では occupancy として使用
  - slot[12:15] は Ray では法線、Point では pseudo-normal として使用

---

## 1) 実装済み（この zip に含む）

### 1.1 Point Answer に Occupancy を追加
- **Point Answer token**: dist に加えて occ (0/1) を入れる
- `pt_occ_pool` があればそれを使用し、無ければ `dist < pt_occ_tau` で自動生成

狙い：
- completion のメッシュ化（isosurface）で重要な near-surface 判別を、表現の段階で強化

### 1.2 Point Answer に Pseudo-normal を追加
- **UDFGrid / Mesh backend**: UDF grid の有限差分から `-∇udf` を推定して正規化
- **PointCloud backend**: 観測点（pc_xyz）の最近傍の normal (pc_n) を query に転写

狙い：
- completion の局所幾何（面の向き）を Answer 側に含めて、表現を「メッシュ化しやすい形」に寄せる

---

## 2) 未実装だが筋が良さそうな拡張案（候補）

### 2.1 Answer の multi-head（距離 + 追加チャネル）
- dist に加えて `curvature / thickness / medialness` 等の近傍特徴を複数チャネルで入れる
- 例：`|∇udf|` の分布、局所 2nd derivative（Hessian の近似）

### 2.2 Ray Answer の情報を増やす
- 現状: hit / t / n
- 追加候補:
  - 交差点近傍の local surface patch embedding（数点サンプルの統計）
  - 深度の不確実性 proxy（近傍の hit consistency）

### 2.3 Answer-drop / multi-view consistency（目的関数は維持）
- Answer feature を確率的に落としても表現が安定するように、token-level augmentation
- loss 自体は NEPA のまま（入力側変換として扱う）

### 2.4 Adaptive querying（オンライン）
- 「不確実な query」を追加で投げる（active completion）
- ただし論文貢献が散るので、別スレッド推奨

---

## 3) 実装メモ（該当ファイル）

- tokenizer: `nepa3d/token/tokenizer.py`
  - Point Answer の slot[11]/[12:15] を occ / pseudo-normal に使用
  - `qa_layout=split`（後述の enc-dec と相性が良い）も追加
- backends:
  - `nepa3d/backends/udfgrid_backend.py` / `mesh_backend.py`: udf_grid から pseudo-normal
  - `nepa3d/backends/pointcloud_backend.py`: pc_n 最近傍転写
- util:
  - `nepa3d/utils/grid.py`: trilinear + finite-diff gradient + pseudo-normal

---

## 4) 実験解釈メモ（Feb 19, 2026）

- Answer 構造強化は pool completion に効く一方、grid 側に副作用が出る構成がある（例: C-2 単体）。
- grid 側の安定化は query 設計（A-1/A-2）と ray 系補助（B-2）を併用したほうが再現性が高い。
- したがって主張は「Answer を増やせば常に改善」ではなく、
  「Answer 拡張 + Query 設計 + 安定化の組合せで completion を押し上げる」が現時点の妥当な整理。

---

## 5) +grad/+unc/topo 連鎖の現状メモ（Feb 19, 2026）

- `plusgut` 連鎖（`causal_plusgut` / `encdec_plusgut_proj` / `encdec_plusgut_bbox`）は評価完了済み。
- 現時点の要点:
  - CPACでは encdec が以前の collapse から回復（pool/grid とも NN-copy 近傍〜上回る設定あり）。
  - ただし UCPR（hard pair）は encdec が依然として near-random で、causal を大きく下回る。
- 注意:
  - `encdec_plusgut_bbox` は現状「diagnostic ckpt-path variant」として扱う（独立再学習ラインではない）。
  - `causal_plusgut_ref` と `causal_plusgut` は同一 ckpt を参照しており、結果は一致する。

---

## 6) 記載運用メモ（Feb 19, 2026）

- `loss=0.0000` は表示丸め（4桁）を含むため、これ単体で改善判断しない。
- `b2/b3/e/d` は重み未指定時に 0 のままなので、ログ解釈時は実行引数と合わせて読む。
- docs へ結果を記載する際は、必ず以下を突合する:
  1. `logs/pretrain/*.log`（学習条件/挙動）
  2. `results/*.json`（最終評価値）
  3. JSON 内 `ckpt`（参照チェックポイントの一致）
