#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N cqa_pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml}"
RUN_TAG="${RUN_TAG:-cqa_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_pretrain}"

EPOCHS="${EPOCHS:-20}"
SAVE_EVERY="${SAVE_EVERY:-5}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_STEPS="${MAX_STEPS:--1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:--1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR="${MIN_LR:-1e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"

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
ANSWER_VOCAB="${ANSWER_VOCAB:-640}"
GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-cqa,primitive-answering}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb_cqa}"

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

echo "=== CQA PRETRAIN ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "workdir=${WORKDIR}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "epochs=${EPOCHS} batch=${BATCH} n_ctx=${N_CTX} n_qry=${N_QRY}" | tee -a "${LOG_PATH}"
echo "max_steps=${MAX_STEPS} scheduler=${LR_SCHEDULER} warmup_steps=${WARMUP_STEPS} warmup_ratio=${WARMUP_RATIO} min_lr=${MIN_LR} clip=${MAX_GRAD_NORM}" | tee -a "${LOG_PATH}"
echo "model=d${D_MODEL}/L${N_LAYERS}/H${N_HEADS} groups=${NUM_GROUPS} group_size=${GROUP_SIZE} gdepth=${GENERATOR_DEPTH}" | tee -a "${LOG_PATH}"
echo "wandb: use=${USE_WANDB} project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} group=${WANDB_GROUP} mode=${WANDB_MODE} dir=${WANDB_DIR}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -m nepa3d.train.pretrain_primitive_answering \
  --mix_config_path "${MIX_CONFIG}" \
  --save_dir "$(dirname "${SAVE_DIR}")" \
  --run_name "$(basename "${SAVE_DIR}")" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
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
  --answer_vocab "${ANSWER_VOCAB}" \
  --generator_depth "${GENERATOR_DEPTH}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_tags "${WANDB_TAGS}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_every "${WANDB_LOG_EVERY}" \
  --wandb_dir "${WANDB_DIR}" \
  2>&1 | tee -a "${LOG_PATH}"
