#!/usr/bin/env bash
set -euo pipefail

# Objective-preserving scale retry (no B-2/C-2/D/E auxiliaries).
# Continuation from QA+dualmask ep049 with n_point schedule:
#   256 -> 512 -> 1024
# and dual-mask-window scaling enabled.
#
# Environment overrides:
#   RUN_TAG, BASE_CKPT, GPU_ID, LR, BATCH, EPOCHS,
#   DUAL_MASK_WINDOW, DUAL_MASK_WINDOW_SCALE,
#   MIX_NUM_SAMPLES, MIX_SEED,
#   MAX_SHAPES, HEAD_TRAIN_MAX_SHAPES, EVAL_SEED

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

RUN_TAG="${RUN_TAG:-eccv_upmix_nepa_qa_dualmask_scale_objpres_lin_s0}"
BASE_CKPT="${BASE_CKPT:-runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt}"
GPU_ID="${GPU_ID:-0}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-24}"
EPOCHS="${EPOCHS:-55}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-32}"
DUAL_MASK_WINDOW_SCALE="${DUAL_MASK_WINDOW_SCALE:-linear}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-12000}"
MIX_SEED="${MIX_SEED:-0}"
MAX_SHAPES="${MAX_SHAPES:-800}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-4000}"
EVAL_SEED="${EVAL_SEED:-0}"

SAVE_DIR="runs/${RUN_TAG}"
LOG_DIR="logs/analysis/${RUN_TAG}"
mkdir -p "${LOG_DIR}"

echo "[$(date +"%F %T")] [${RUN_TAG}] pretrain start (gpu=${GPU_ID}, lr=${LR}, scale=${DUAL_MASK_WINDOW_SCALE})"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_BIN" -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples "${MIX_NUM_SAMPLES}" --mix_seed "${MIX_SEED}" \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 \
  --dual_mask_window "${DUAL_MASK_WINDOW}" \
  --dual_mask_window_scale "${DUAL_MASK_WINDOW_SCALE}" \
  --dual_mask_warmup_frac 0.05 \
  --epochs "${EPOCHS}" --batch "${BATCH}" --lr "${LR}" \
  --n_point 256 --n_ray 256 \
  --n_point_schedule "0:256,50:512,53:1024" \
  --n_ray_schedule "0:256" \
  --max_len -1 \
  --num_workers 6 \
  --resume_optimizer 1 --resume_optimizer_partial 1 \
  --save_every 1 --save_last 1 \
  --resume "${BASE_CKPT}" \
  --save_dir "${SAVE_DIR}" --seed 0 \
  > "${LOG_DIR}/pretrain.log" 2>&1
echo "[$(date +"%F %T")] [${RUN_TAG}] pretrain done"

# Pick latest checkpoint in the save dir.
CKPT="$(ls -1 "${SAVE_DIR}"/ckpt_ep*.pt | sort | tail -n 1)"
if [ ! -f "${CKPT}" ]; then
  echo "[error] checkpoint not found in ${SAVE_DIR}" >&2
  exit 1
fi
echo "[$(date +"%F %T")] [${RUN_TAG}] eval ckpt=${CKPT}"

# CPAC: n_context=512
for QS in pool grid; do
  EXTRA_ARGS=()
  SUF="${QS}"
  if [ "${QS}" = "grid" ]; then
    EXTRA_ARGS=(--grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8)
    SUF="grid_near08"
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_BIN" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT}" --max_len 2562 \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context 512 --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source "${QS}" \
    "${EXTRA_ARGS[@]}" \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${RUN_TAG}_pc512q256_${SUF}_h_htrain4k.json" \
    > "${LOG_DIR}/cpac_pc512_${SUF}.log" 2>&1
done

# CPAC: n_context=1024
for QS in pool grid; do
  EXTRA_ARGS=()
  SUF="${QS}"
  if [ "${QS}" = "grid" ]; then
    EXTRA_ARGS=(--grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8)
    SUF="grid_near08"
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_BIN" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT}" --max_len 2562 \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context 1024 --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source "${QS}" \
    "${EXTRA_ARGS[@]}" \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${RUN_TAG}_pc1024q256_${SUF}_h_htrain4k.json" \
    > "${LOG_DIR}/cpac_pc1024_${SUF}.log" 2>&1
done

echo "[$(date +"%F %T")] [${RUN_TAG}] all done"
