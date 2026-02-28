#!/usr/bin/env bash
set -euo pipefail

# Patchified Transformer scratch baseline for ScanObjectNN.
#
# Goal: reproduce the typical "Transformer" baseline range (~0.77-0.80+)
# before attributing gaps to NEPA pretraining details.

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
DATA_FORMAT="${DATA_FORMAT:-npz}"  # npz | scan_h5
SCAN_H5_ROOT="${SCAN_H5_ROOT:-data/ScanObjectNN/h5_files/main_split}"
SCAN_VARIANT="${SCAN_VARIANT:-pb_t50_rs}"  # auto|obj_bg|obj_only|pb_t50_rs
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"

# Training hyperparams (Point-MAE-ish defaults)
RUN_NAME="${RUN_NAME:-patchcls_scan_scratch}"
CKPT="${CKPT:-}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
LR="${LR:-5e-4}"
WD="${WD:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

N_POINT="${N_POINT:-1024}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PATCH_EMBED="${PATCH_EMBED:-fps_knn}"  # fps_knn | serial
MODEL_SOURCE="${MODEL_SOURCE:-patchcls}"  # patchcls | pointmae | patchnepa
USE_RAY_PATCH="${USE_RAY_PATCH:-0}"       # 0 | 1
N_RAY="${N_RAY:-256}"
RAY_SAMPLE_MODE_TRAIN="${RAY_SAMPLE_MODE_TRAIN:-random}"  # random | first
RAY_SAMPLE_MODE_EVAL="${RAY_SAMPLE_MODE_EVAL:-first}"     # random | first
RAY_POOL_MODE="${RAY_POOL_MODE:-max}"      # max | mean
RAY_FUSE_MODE="${RAY_FUSE_MODE:-concat}"   # concat | add
RAY_HIDDEN_DIM="${RAY_HIDDEN_DIM:-128}"
RAY_MISS_T="${RAY_MISS_T:-4.0}"
RAY_HIT_THRESHOLD="${RAY_HIT_THRESHOLD:-0.5}"
SERIAL_ORDER="${SERIAL_ORDER:-morton}"  # morton | morton_trans | z | z-trans | random | identity
SERIAL_BITS="${SERIAL_BITS:-10}"
SERIAL_SHUFFLE_WITHIN_PATCH="${SERIAL_SHUFFLE_WITHIN_PATCH:-0}"  # 0 | 1
BACKBONE_MODE="${BACKBONE_MODE:-nepa2d}"  # nepa2d | vanilla
QK_NORM="${QK_NORM:-1}"                   # 0 | 1 (nepa2d path)
QK_NORM_AFFINE="${QK_NORM_AFFINE:-0}"     # 0 | 1
QK_NORM_BIAS="${QK_NORM_BIAS:-0}"         # 0 | 1
LAYERSCALE_VALUE="${LAYERSCALE_VALUE:-1e-5}"
ROPE_THETA="${ROPE_THETA:-100.0}"         # <=0 disables RoPE in nepa2d path
ROPE_PREFIX_TOKENS="${ROPE_PREFIX_TOKENS:-1}"
USE_GATED_MLP="${USE_GATED_MLP:-0}"       # 0 | 1
HIDDEN_ACT="${HIDDEN_ACT:-gelu}"          # gelu | silu
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"

NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_MODE="${BATCH_MODE:-global}"  # global | per_proc
MASTER_PORT="${MASTER_PORT:-29577}"
POOLING="${POOLING:-cls_max}"  # mean | cls | cls_max
POS_MODE="${POS_MODE:-center_mlp}"  # learned | center_mlp
HEAD_MODE="${HEAD_MODE:-pointmae_mlp}"  # auto | linear | pointmae_mlp
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.5}"
INIT_MODE="${INIT_MODE:-default}"  # default | pointmae
AUG_PRESET="${AUG_PRESET:-pointmae}"  # none | default | strong | pointmae
AUG_EVAL="${AUG_EVAL:-1}"             # 0 | 1 (policy default: 1)
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}" # policy default: 10

SAVE_DIR="${SAVE_DIR:-runs/patchcls}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"  # file | group_auto | group_scanobjectnn | pointmae

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ "${DATA_FORMAT}" != "npz" ]] && [[ "${DATA_FORMAT}" != "scan_h5" ]]; then
  echo "[error] DATA_FORMAT must be one of: npz | scan_h5 (got: ${DATA_FORMAT})"
  exit 1
fi
if [[ "${DATA_FORMAT}" == "npz" ]]; then
  if [ ! -d "${CACHE_ROOT}" ]; then
    echo "[error] missing cache root: ${CACHE_ROOT}"
    exit 1
  fi
  if [[ "${CACHE_ROOT}" == *"scanobjectnn_"*"_v2" ]] && [[ "${ALLOW_SCAN_UNISCALE_V2}" != "1" ]]; then
    echo "[error] CACHE_ROOT=${CACHE_ROOT} is a uniscale v2 cache and is disallowed by policy."
    echo "        Use v3_nonorm variant caches, or set ALLOW_SCAN_UNISCALE_V2=1 for intentional legacy reruns."
    exit 2
  fi
else
  if [ ! -d "${SCAN_H5_ROOT}" ]; then
    echo "[error] missing ScanObjectNN h5 root: ${SCAN_H5_ROOT}"
    exit 1
  fi
fi

ARGS=(
  -m nepa3d.train.finetune_patch_cls
  --data_format "${DATA_FORMAT}"
  --cache_root "${CACHE_ROOT}"
  --scan_h5_root "${SCAN_H5_ROOT}"
  --scan_variant "${SCAN_VARIANT}"
  --run_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --ckpt "${CKPT}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --batch_mode "${BATCH_MODE}"
  --lr "${LR}"
  --weight_decay "${WD}"
  --lr_scheduler cosine
  --warmup_epochs "${WARMUP_EPOCHS}"
  --n_point "${N_POINT}"
  --use_ray_patch "${USE_RAY_PATCH}"
  --n_ray "${N_RAY}"
  --ray_sample_mode_train "${RAY_SAMPLE_MODE_TRAIN}"
  --ray_sample_mode_eval "${RAY_SAMPLE_MODE_EVAL}"
  --ray_pool_mode "${RAY_POOL_MODE}"
  --ray_fuse_mode "${RAY_FUSE_MODE}"
  --ray_hidden_dim "${RAY_HIDDEN_DIM}"
  --ray_miss_t "${RAY_MISS_T}"
  --ray_hit_threshold "${RAY_HIT_THRESHOLD}"
  --model_source "${MODEL_SOURCE}"
  --patch_embed "${PATCH_EMBED}"
  --pt_sample_mode_train "${PT_SAMPLE_MODE_TRAIN}"
  --pt_sample_mode_eval "${PT_SAMPLE_MODE_EVAL}"
  --serial_order "${SERIAL_ORDER}"
  --serial_bits "${SERIAL_BITS}"
  --serial_shuffle_within_patch "${SERIAL_SHUFFLE_WITHIN_PATCH}"
  --aug_preset "${AUG_PRESET}"
  --aug_eval "${AUG_EVAL}"
  --mc_eval_k_test "${MC_EVAL_K_TEST}"
  --val_ratio "${VAL_RATIO}"
  --val_seed "${VAL_SEED}"
  --val_split_mode "${VAL_SPLIT_MODE}"
  --allow_scan_uniscale_v2 "${ALLOW_SCAN_UNISCALE_V2}"
  --seed "${SEED}"
  --num_groups "${NUM_GROUPS}"
  --group_size "${GROUP_SIZE}"
  --backbone_mode "${BACKBONE_MODE}"
  --qk_norm "${QK_NORM}"
  --qk_norm_affine "${QK_NORM_AFFINE}"
  --qk_norm_bias "${QK_NORM_BIAS}"
  --layerscale_value "${LAYERSCALE_VALUE}"
  --rope_theta "${ROPE_THETA}"
  --rope_prefix_tokens "${ROPE_PREFIX_TOKENS}"
  --use_gated_mlp "${USE_GATED_MLP}"
  --hidden_act "${HIDDEN_ACT}"
  --pooling "${POOLING}"
  --pos_mode "${POS_MODE}"
  --head_mode "${HEAD_MODE}"
  --head_hidden_dim "${HEAD_HIDDEN_DIM}"
  --head_dropout "${HEAD_DROPOUT}"
  --init_mode "${INIT_MODE}"
  --is_causal 0
  --num_workers "${NUM_WORKERS}"
)

echo "[patchcls] data_format=${DATA_FORMAT} variant=${SCAN_VARIANT} run_name=${RUN_NAME}"
echo "[patchcls] nproc_per_node=${NPROC_PER_NODE} batch=${BATCH} batch_mode=${BATCH_MODE} aug_eval=${AUG_EVAL} mc_test=${MC_EVAL_K_TEST}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${ARGS[@]}"
fi
