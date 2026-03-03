#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N pntok_pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_tokens.yaml}"
RUN_TAG="${RUN_TAG:-patchnepa_tokens_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-runs/patchnepa_tokens/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/patch_nepa_pretrain_tokens}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"

MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"   # none|cosine
WARMUP_STEPS="${WARMUP_STEPS:--1}"       # -1 => derive from warmup_ratio*max_steps
WARMUP_RATIO="${WARMUP_RATIO:-0.025}"
MIN_LR="${MIN_LR:-1e-6}"

N_SURF="${N_SURF:-2048}"
N_QRY="${N_QRY:-1024}"
N_RAY="${N_RAY:-0}"
TOKEN_QA_LAYOUT="${TOKEN_QA_LAYOUT:-split}"  # interleave|split|split_sep

PM_PC_NORM="${PM_PC_NORM:-1}"
PM_SCALE_TRANSLATE="${PM_SCALE_TRANSLATE:-1}"
PM_SCALE_LOW="${PM_SCALE_LOW:-0.6666667}"
PM_SCALE_HIGH="${PM_SCALE_HIGH:-1.5}"
PM_TRANSLATE="${PM_TRANSLATE:-0.2}"
PM_TRANSFORM_ANSWERS="${PM_TRANSFORM_ANSWERS:-1}"

PATCH_EMBED="${PATCH_EMBED:-fps_knn}"
PATCH_LOCAL_ENCODER="${PATCH_LOCAL_ENCODER:-pointmae_conv}"
PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}"
GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_GROUPS="${NUM_GROUPS:-64}"
PATCH_ORDER_MODE="${PATCH_ORDER_MODE:-morton}"

D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-6}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.0}"
QK_NORM="${QK_NORM:-1}"
QK_NORM_AFFINE="${QK_NORM_AFFINE:-0}"
QK_NORM_BIAS="${QK_NORM_BIAS:-0}"
LAYERSCALE_VALUE="${LAYERSCALE_VALUE:-1e-5}"
ROPE_THETA="${ROPE_THETA:-100.0}"
BACKBONE_MODE="${BACKBONE_MODE:-nepa2d}"

ANSWER_IN_DIM="${ANSWER_IN_DIM:-0}"
ANSWER_MLP_LAYERS="${ANSWER_MLP_LAYERS:-2}"
ANSWER_POOL="${ANSWER_POOL:-max}"
LOSS_TARGET_MODE="${LOSS_TARGET_MODE:-content_tokens}"   # full_z|content_tokens
LOSS_MASK_MODE="${LOSS_MASK_MODE:-answer_and_point_context}"  # answer_only_if_present|answer_and_point_context|non_special
SKIP_K="${SKIP_K:-1}"

Q_MASK_PROB="${Q_MASK_PROB:-0.0}"
Q_MASK_MODE="${Q_MASK_MODE:-mask_token}"  # mask_token|zero
DUAL_MASK_NEAR="${DUAL_MASK_NEAR:-0.0}"
DUAL_MASK_FAR="${DUAL_MASK_FAR:-0.0}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-0}"
DUAL_MASK_TYPE_AWARE="${DUAL_MASK_TYPE_AWARE:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-tokens,v2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-1}"
DIAG_EVERY="${DIAG_EVERY:-1}"

mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${SAVE_DIR}")"
cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== PATCH-NEPA TOKEN PRETRAIN ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "workdir=${WORKDIR}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "token_qa_layout=${TOKEN_QA_LAYOUT}" | tee -a "${LOG_PATH}"
echo "n_surf=${N_SURF} n_qry=${N_QRY} n_ray=${N_RAY}" | tee -a "${LOG_PATH}"
echo "batch=${BATCH} max_steps=${MAX_STEPS} lr=${LR}" | tee -a "${LOG_PATH}"
echo "optimizer: wd=${WEIGHT_DECAY} lr_scheduler=${LR_SCHEDULER} warmup_steps=${WARMUP_STEPS} warmup_ratio=${WARMUP_RATIO} min_lr=${MIN_LR}" | tee -a "${LOG_PATH}"
echo "patch: embed=${PATCH_EMBED} local_encoder=${PATCH_LOCAL_ENCODER} fps_random_start=${PATCH_FPS_RANDOM_START} group_size=${GROUP_SIZE} num_groups=${NUM_GROUPS}" | tee -a "${LOG_PATH}"
echo "pm_compat: pc_norm=${PM_PC_NORM} scale_translate=${PM_SCALE_TRANSLATE} scale=[${PM_SCALE_LOW},${PM_SCALE_HIGH}] translate=${PM_TRANSLATE} transform_answers=${PM_TRANSFORM_ANSWERS}" | tee -a "${LOG_PATH}"
echo "diag: enabled=1 diag_every=${DIAG_EVERY}" | tee -a "${LOG_PATH}"
echo "loss: target_mode=${LOSS_TARGET_MODE} mask_mode=${LOSS_MASK_MODE}" | tee -a "${LOG_PATH}"
echo "wandb: use=${USE_WANDB} project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} group=${WANDB_GROUP} mode=${WANDB_MODE}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.train.pretrain_patch_nepa_tokens \
  --mix_config_path "${MIX_CONFIG}" \
  --save_dir "$(dirname "${SAVE_DIR}")" \
  --run_name "$(basename "${SAVE_DIR}")" \
  --save_every "${SAVE_EVERY}" \
  --n_surf "${N_SURF}" \
  --n_qry "${N_QRY}" \
  --n_ray "${N_RAY}" \
  --token_qa_layout "${TOKEN_QA_LAYOUT}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --pm_pc_norm "${PM_PC_NORM}" \
  --pm_scale_translate "${PM_SCALE_TRANSLATE}" \
  --pm_scale_low "${PM_SCALE_LOW}" \
  --pm_scale_high "${PM_SCALE_HIGH}" \
  --pm_translate "${PM_TRANSLATE}" \
  --pm_transform_answers "${PM_TRANSFORM_ANSWERS}" \
  --patch_embed "${PATCH_EMBED}" \
  --patch_local_encoder "${PATCH_LOCAL_ENCODER}" \
  --patch_fps_random_start "${PATCH_FPS_RANDOM_START}" \
  --group_size "${GROUP_SIZE}" \
  --num_groups "${NUM_GROUPS}" \
  --patch_order_mode "${PATCH_ORDER_MODE}" \
  --d_model "${D_MODEL}" \
  --n_layers "${N_LAYERS}" \
  --n_heads "${N_HEADS}" \
  --mlp_ratio "${MLP_RATIO}" \
  --drop_path_rate "${DROP_PATH_RATE}" \
  --qk_norm "${QK_NORM}" \
  --qk_norm_affine "${QK_NORM_AFFINE}" \
  --qk_norm_bias "${QK_NORM_BIAS}" \
  --layerscale_value "${LAYERSCALE_VALUE}" \
  --rope_theta "${ROPE_THETA}" \
  --backbone_mode "${BACKBONE_MODE}" \
  --answer_in_dim "${ANSWER_IN_DIM}" \
  --answer_mlp_layers "${ANSWER_MLP_LAYERS}" \
  --answer_pool "${ANSWER_POOL}" \
  --loss_target_mode "${LOSS_TARGET_MODE}" \
  --loss_mask_mode "${LOSS_MASK_MODE}" \
  --skip_k "${SKIP_K}" \
  --max_steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --min_lr "${MIN_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --grad_accum "${GRAD_ACCUM}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --q_mask_prob "${Q_MASK_PROB}" \
  --q_mask_mode "${Q_MASK_MODE}" \
  --dual_mask_near "${DUAL_MASK_NEAR}" \
  --dual_mask_far "${DUAL_MASK_FAR}" \
  --dual_mask_window "${DUAL_MASK_WINDOW}" \
  --dual_mask_type_aware "${DUAL_MASK_TYPE_AWARE}" \
  --use_wandb "${USE_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  --wandb_group "${WANDB_GROUP}" \
  --wandb_tags "${WANDB_TAGS}" \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_log_every "${WANDB_LOG_EVERY}" \
  --diag_every "${DIAG_EVERY}" \
  --seed "${SEED}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
