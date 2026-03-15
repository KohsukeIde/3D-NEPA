#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

RUN_SET="${RUN_SET:-patchnepa_cqa_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_visthick_g2}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"

export WORKDIR="${ROOT_DIR}"
export GROUP_LIST="${GROUP_LIST:-qgah50055}"
export RT_QG="${RT_QG:-1}"
export WALLTIME="${WALLTIME:-24:00:00}"

export RUN_SET
export RUN_TAG
export SAVE_DIR
export LOG_ROOT
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-pretrain}"
export WANDB_GROUP="${WANDB_GROUP:-${RUN_SET}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_SET}_${RUN_TAG}}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,current}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/wandb_cqa}"

export EPOCHS="${EPOCHS:-20}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
export BATCH="${BATCH:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export LR="${LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export MAX_STEPS="${MAX_STEPS:--1}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export WARMUP_STEPS="${WARMUP_STEPS:--1}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export MIN_LR="${MIN_LR:-1e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export N_CTX="${N_CTX:-2048}"
export N_QRY="${N_QRY:-64}"

export D_MODEL="${D_MODEL:-384}"
export N_LAYERS="${N_LAYERS:-12}"
export N_HEADS="${N_HEADS:-6}"
export MLP_RATIO="${MLP_RATIO:-4.0}"
export DROPOUT="${DROPOUT:-0.0}"
export DROP_PATH="${DROP_PATH:-0.0}"
export BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}"
export NUM_GROUPS="${NUM_GROUPS:-64}"
export GROUP_SIZE="${GROUP_SIZE:-32}"
export PATCH_CENTER_MODE="${PATCH_CENTER_MODE:-fps}"
export PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}"
export LOCAL_ENCODER="${LOCAL_ENCODER:-pointmae_conv}"
export QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-6}"
export ANSWER_VOCAB="${ANSWER_VOCAB:-640}"
export GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}"

echo "[abci-cqa-pretrain] mix_config=${MIX_CONFIG}"
echo "[abci-cqa-pretrain] save_dir=${SAVE_DIR}"
echo "[abci-cqa-pretrain] log_root=${LOG_ROOT}"

exec "${ROOT_DIR}/scripts/pretrain/submit_pretrain_primitive_answering_qf.sh"
