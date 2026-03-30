#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

RUN_SET="${RUN_SET:-patchnepa_cqa_external_pointmae_udfdist_pcbank_$(date +%Y%m%d_%H%M%S)}"
TRAIN_TAG="${TRAIN_TAG:-cqa_external_pointmae_udfdist_pcbank_g2_s10000}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${TRAIN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"

EXTERNAL_BACKBONE_CKPT="${EXTERNAL_BACKBONE_CKPT:-Point-MAE/pretrained/pretrain.pth}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-10:00:00}"

MAX_STEPS="${MAX_STEPS:-10000}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MIN_LR="${MIN_LR:-1e-6}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
QUERY_ORDER="${QUERY_ORDER:-shuffled}"
EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"

D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-6}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROPOUT="${DROPOUT:-0.0}"
DROP_PATH="${DROP_PATH:-0.0}"
BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PATCH_CENTER_MODE="${PATCH_CENTER_MODE:-fps}"
PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}"
QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-0}"
ANSWER_VOCAB="${ANSWER_VOCAB:-0}"
GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}"
MODEL_ARCH="${MODEL_ARCH:-external_pointmae}"
CODEC_VERSION="${CODEC_VERSION:-cqa_v1}"
FREEZE_EXTERNAL_ENCODER="${FREEZE_EXTERNAL_ENCODER:-1}"
EXTERNAL_BACKBONE_DEPTH="${EXTERNAL_BACKBONE_DEPTH:-12}"
EXTERNAL_BACKBONE_HEADS="${EXTERNAL_BACKBONE_HEADS:-6}"
EXTERNAL_BACKBONE_DROP_PATH="${EXTERNAL_BACKBONE_DROP_PATH:-0.1}"
ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-external}"
WANDB_GROUP="${WANDB_GROUP:-${RUN_SET}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${TRAIN_TAG}}"
WANDB_TAGS="${WANDB_TAGS:-abci,cqa,external,pointmae,udf_distance,pcbank}"
WANDB_MODE="${WANDB_MODE:-online}"

TRAIN_JOB_OUTPUT="$(
  env \
    RUN_SET="${RUN_SET}" \
    RUN_TAG="${TRAIN_TAG}" \
    SAVE_DIR="${SAVE_DIR}" \
    LOG_ROOT="${LOG_ROOT}" \
    MIX_CONFIG="${MIX_CONFIG}" \
    GROUP_LIST="${GROUP_LIST}" \
    RT_QG="${RT_QG}" \
    WALLTIME="${WALLTIME}" \
    MAX_STEPS="${MAX_STEPS}" \
    EPOCHS="${EPOCHS}" \
    BATCH="${BATCH}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SEED="${SEED}" \
    LR="${LR}" \
    WEIGHT_DECAY="${WEIGHT_DECAY}" \
    SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS}" \
    LR_SCHEDULER="${LR_SCHEDULER}" \
    WARMUP_RATIO="${WARMUP_RATIO}" \
    MIN_LR="${MIN_LR}" \
    MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
    N_CTX="${N_CTX}" \
    N_QRY="${N_QRY}" \
    QUERY_ORDER="${QUERY_ORDER}" \
    EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER}" \
    D_MODEL="${D_MODEL}" \
    N_LAYERS="${N_LAYERS}" \
    N_HEADS="${N_HEADS}" \
    MLP_RATIO="${MLP_RATIO}" \
    DROPOUT="${DROPOUT}" \
    DROP_PATH="${DROP_PATH}" \
    BACKBONE_IMPL="${BACKBONE_IMPL}" \
    NUM_GROUPS="${NUM_GROUPS}" \
    GROUP_SIZE="${GROUP_SIZE}" \
    PATCH_CENTER_MODE="${PATCH_CENTER_MODE}" \
    PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START}" \
    QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB}" \
    ANSWER_VOCAB="${ANSWER_VOCAB}" \
    GENERATOR_DEPTH="${GENERATOR_DEPTH}" \
    MODEL_ARCH="${MODEL_ARCH}" \
    CODEC_VERSION="${CODEC_VERSION}" \
    EXTERNAL_BACKBONE_CKPT="${EXTERNAL_BACKBONE_CKPT}" \
    FREEZE_EXTERNAL_ENCODER="${FREEZE_EXTERNAL_ENCODER}" \
    EXTERNAL_BACKBONE_DEPTH="${EXTERNAL_BACKBONE_DEPTH}" \
    EXTERNAL_BACKBONE_HEADS="${EXTERNAL_BACKBONE_HEADS}" \
    EXTERNAL_BACKBONE_DROP_PATH="${EXTERNAL_BACKBONE_DROP_PATH}" \
    ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION}" \
    QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE}" \
    RUN_EVAL_CONTROLS="0" \
    RUN_EVAL_CURVE="0" \
    USE_WANDB="${USE_WANDB}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_GROUP="${WANDB_GROUP}" \
    WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
    WANDB_TAGS="${WANDB_TAGS}" \
    WANDB_MODE="${WANDB_MODE}" \
    bash "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh"
)"

echo "${TRAIN_JOB_OUTPUT}"
TRAIN_JOB_ID="$(printf '%s\n' "${TRAIN_JOB_OUTPUT}" | awk '/^\[submitted\]/{print $2}' | tail -n1)"
if [[ -z "${TRAIN_JOB_ID}" ]]; then
  echo "[error] failed to parse train job id"
  exit 2
fi

CKPT_PATH="${SAVE_DIR}/ckpt_final.pt"
EVAL_ROOT="${ROOT_DIR}/results/cqa_eval/${RUN_SET}"
COMP_ROOT="${ROOT_DIR}/results/cqa_completion/${RUN_SET}"

env \
  CKPT="${CKPT_PATH}" \
  RUN_SET="${RUN_SET}_same_eval" \
  RUN_TAG="cqa_external_pointmae_udfdist_same_eval" \
  MIX_CONFIG="nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml" \
  OUT_JSON="${EVAL_ROOT}/cqa_external_pointmae_udfdist_same_eval.json" \
  QSUB_DEPEND="afterok:${TRAIN_JOB_ID}" \
  bash "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_offdiag.sh"

env \
  CKPT="${CKPT_PATH}" \
  RUN_SET="${RUN_SET}_offdiag_eval" \
  RUN_TAG="cqa_external_pointmae_udfdist_offdiag_eval" \
  MIX_CONFIG="nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml" \
  OUT_JSON="${EVAL_ROOT}/cqa_external_pointmae_udfdist_offdiag_eval.json" \
  QSUB_DEPEND="afterok:${TRAIN_JOB_ID}" \
  bash "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_offdiag.sh"

env \
  CKPT="${CKPT_PATH}" \
  MODE="same" \
  RUN_SET="${RUN_SET}_same_completion" \
  RUN_TAG="cqa_external_pointmae_udfdist_same_translation_g16_s64" \
  OUT_JSON="${COMP_ROOT}/cqa_external_pointmae_udfdist_same_translation_g16_s64.json" \
  QSUB_DEPEND="afterok:${TRAIN_JOB_ID}" \
  bash "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh"

env \
  CKPT="${CKPT_PATH}" \
  MODE="offdiag" \
  RUN_SET="${RUN_SET}_offdiag_completion" \
  RUN_TAG="cqa_external_pointmae_udfdist_offdiag_translation_g16_s64" \
  OUT_JSON="${COMP_ROOT}/cqa_external_pointmae_udfdist_offdiag_translation_g16_s64.json" \
  QSUB_DEPEND="afterok:${TRAIN_JOB_ID}" \
  bash "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh"

echo "[train_job] ${TRAIN_JOB_ID}"
echo "[ckpt] ${CKPT_PATH}"
