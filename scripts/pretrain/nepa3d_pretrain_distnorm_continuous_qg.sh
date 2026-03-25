#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N cqa_dnc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_continuous.yaml}"
RUN_TAG="${RUN_TAG:-cqa_distnorm_continuous_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_pretrain}"

EPOCHS="${EPOCHS:-20}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_STEPS="${MAX_STEPS:-10000}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:--1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR="${MIN_LR:-1e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
QUERY_ORDER="${QUERY_ORDER:-shuffled}"

D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-6}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROPOUT="${DROPOUT:-0.0}"
DROP_PATH="${DROP_PATH:-0.0}"
BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PATCH_CENTER_MODE="${PATCH_CENTER_MODE:-fps}"
PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}"
LOCAL_ENCODER="${LOCAL_ENCODER:-pointmae_conv}"
QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-6}"
GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}"
DISTANCE_FLOOR="${DISTANCE_FLOOR:-0.0}"
TASK_LOSS_BALANCE="${TASK_LOSS_BALANCE:-mean}"
LOSS_EMA_MOMENTUM="${LOSS_EMA_MOMENTUM:-0.99}"
LOSS_EMA_EPS="${LOSS_EMA_EPS:-1e-6}"
LOSS_WEIGHT_DISTANCE="${LOSS_WEIGHT_DISTANCE:-1.0}"
LOSS_WEIGHT_AO="${LOSS_WEIGHT_AO:-1.0}"
LOSS_WEIGHT_NORMAL="${LOSS_WEIGHT_NORMAL:-1.0}"

LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"

mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${SAVE_DIR}")"
cd "${WORKDIR}"

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== CQA DIST+NORMAL CONTINUOUS PRETRAIN ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "workdir=${WORKDIR}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "epochs=${EPOCHS} batch=${BATCH} n_ctx=${N_CTX} n_qry=${N_QRY} query_order=${QUERY_ORDER}" | tee -a "${LOG_PATH}"
echo "loss_balance=${TASK_LOSS_BALANCE} ema_m=${LOSS_EMA_MOMENTUM} w=(dist=${LOSS_WEIGHT_DISTANCE},ao=${LOSS_WEIGHT_AO},norm=${LOSS_WEIGHT_NORMAL})" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.train.pretrain_distnorm_continuous \
  --mix_config_path "${MIX_CONFIG}" \
  --save_dir "$(dirname "${SAVE_DIR}")" \
  --run_name "$(basename "${SAVE_DIR}")" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --max_steps "${MAX_STEPS}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr "${MIN_LR}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --save_every "${SAVE_EVERY}" \
  --save_every_steps "${SAVE_EVERY_STEPS}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --query_order "${QUERY_ORDER}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --n_heads "${N_HEADS}" \
  --mlp_ratio "${MLP_RATIO}" \
  --dropout "${DROPOUT}" \
  --drop_path "${DROP_PATH}" \
  --backbone_impl "${BACKBONE_IMPL}" \
  --num_groups "${NUM_GROUPS}" \
  --group_size "${GROUP_SIZE}" \
  --patch_center_mode "${PATCH_CENTER_MODE}" \
  --patch_fps_random_start "${PATCH_FPS_RANDOM_START}" \
  --local_encoder "${LOCAL_ENCODER}" \
  --query_type_vocab "${QUERY_TYPE_VOCAB}" \
  --generator_depth "${GENERATOR_DEPTH}" \
  --distance_floor "${DISTANCE_FLOOR}" \
  --task_loss_balance "${TASK_LOSS_BALANCE}" \
  --loss_ema_momentum "${LOSS_EMA_MOMENTUM}" \
  --loss_ema_eps "${LOSS_EMA_EPS}" \
  --loss_weight_distance "${LOSS_WEIGHT_DISTANCE}" \
  --loss_weight_ao "${LOSS_WEIGHT_AO}" \
  --loss_weight_normal "${LOSS_WEIGHT_NORMAL}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
