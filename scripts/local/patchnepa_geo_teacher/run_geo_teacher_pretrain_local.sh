#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="${RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
DEFAULT_SESSION="${RUN_TAG//[^[:alnum:]_]/_}"
TMUX_SESSION="${TMUX_SESSION:-${DEFAULT_SESSION}}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

mkdir -p "${LOG_ROOT}" "${SAVE_ROOT}"

export ROOT_DIR
export RUN_TAG SAVE_DIR LOG_ROOT LOG_FILE PID_FILE
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29624}"

export EPOCHS="${EPOCHS:-100}"
export SAVE_EVERY="${SAVE_EVERY:-10}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
export BATCH="${BATCH:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
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
export QUERY_ORDER="${QUERY_ORDER:-shuffled}"
export CODEC_VERSION="${CODEC_VERSION:-cqa_v2}"

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
export QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-0}"
export ANSWER_VOCAB="${ANSWER_VOCAB:-0}"
export GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}"
export MODEL_ARCH="${MODEL_ARCH:-prefixlm}"
export DECODER_LAYERS="${DECODER_LAYERS:-4}"
export EXTERNAL_BACKBONE_CKPT="${EXTERNAL_BACKBONE_CKPT:-}"
export FREEZE_EXTERNAL_ENCODER="${FREEZE_EXTERNAL_ENCODER:-1}"
export EXTERNAL_BACKBONE_DEPTH="${EXTERNAL_BACKBONE_DEPTH:-12}"
export EXTERNAL_BACKBONE_HEADS="${EXTERNAL_BACKBONE_HEADS:-6}"
export EXTERNAL_BACKBONE_DROP_PATH="${EXTERNAL_BACKBONE_DROP_PATH:-0.1}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-no_q}"
export HEAD_MODE="${HEAD_MODE:-multihead}"
export SAMPLING_PROTOCOL="${SAMPLING_PROTOCOL:-packed}"
export LOSS_BALANCE="${LOSS_BALANCE:-per_task}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-pretrain}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_TAG}}"
export WANDB_GROUP="${WANDB_GROUP:-itachi_geo_teacher}"
export WANDB_TAGS="${WANDB_TAGS:-local,itachi,cqa,geo_teacher,distnorm_unsigned}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
export WANDB_DIR="${WANDB_DIR:-${ROOT_DIR}/wandb_cqa}"
export RUN_EVAL_CONTROLS="${RUN_EVAL_CONTROLS:-0}"
export RUN_EVAL_CURVE="${RUN_EVAL_CURVE:-0}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] geo-teacher pretrain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting in foreground"
  echo "[info] log=${LOG_FILE}"
  exec bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_pretrain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] geo-teacher pretrain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    for name in \
      ROOT_DIR RUN_TAG SAVE_DIR LOG_ROOT LOG_FILE PID_FILE PYTHON_BIN MIX_CONFIG \
      CUDA_VISIBLE_DEVICES NPROC_PER_NODE MASTER_PORT EPOCHS SAVE_EVERY SAVE_EVERY_STEPS \
      BATCH NUM_WORKERS SEED LR WEIGHT_DECAY MAX_STEPS LR_SCHEDULER WARMUP_STEPS \
      WARMUP_RATIO MIN_LR MAX_GRAD_NORM N_CTX N_QRY QUERY_ORDER CODEC_VERSION \
      D_MODEL N_LAYERS N_HEADS MLP_RATIO DROPOUT DROP_PATH BACKBONE_IMPL \
      NUM_GROUPS GROUP_SIZE PATCH_CENTER_MODE PATCH_FPS_RANDOM_START LOCAL_ENCODER \
      QUERY_TYPE_VOCAB ANSWER_VOCAB GENERATOR_DEPTH MODEL_ARCH DECODER_LAYERS \
      EXTERNAL_BACKBONE_CKPT FREEZE_EXTERNAL_ENCODER EXTERNAL_BACKBONE_DEPTH \
      EXTERNAL_BACKBONE_HEADS EXTERNAL_BACKBONE_DROP_PATH ANSWER_FACTORIZATION \
      QUERY_INTERFACE_MODE HEAD_MODE SAMPLING_PROTOCOL LOSS_BALANCE USE_WANDB \
      WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_GROUP WANDB_TAGS WANDB_MODE \
      WANDB_LOG_EVERY WANDB_DIR RUN_EVAL_CONTROLS RUN_EVAL_CURVE OMP_NUM_THREADS \
      MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
    do
      tmux set-environment -g "${name}" "${!name:-}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_pretrain_inner.sh'"
    sleep 1
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached geo-teacher pretrain in tmux"
      echo "[info] session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    echo "[error] failed to start tmux session=${TMUX_SESSION}"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_pretrain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached geo-teacher pretrain"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached geo-teacher pretrain"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
