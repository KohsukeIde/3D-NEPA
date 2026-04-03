#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${ROOT_DIR}"

CHAIN_RUN_TAG="${CHAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_300ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.pid}"
PREV_POSTTRAIN_RUN_TAG="${PREV_POSTTRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"

mkdir -p "${LOG_ROOT}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT
echo "$$" > "${PID_FILE}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

wait_for_prev_posttrain() {
  if [[ "${WAIT_FOR_PREV_POSTTRAIN:-1}" != "1" ]]; then
    return 0
  fi
  while true; do
    local status_output status
    status_output="$(
      env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
      PRETRAIN_RUN_TAG="${PREV_POSTTRAIN_RUN_TAG}" \
      bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/status_geo_teacher_posttrain_local.sh" || true
    )"
    status="$(awk -F= '/^status=/{print $2; exit}' <<< "${status_output}")"
    if [[ "${status}" == "running" ]]; then
      log "waiting for previous posttrain run=${PREV_POSTTRAIN_RUN_TAG} to finish"
      sleep "${PREV_POSTTRAIN_POLL_SEC:-120}"
      continue
    fi
    break
  done
}

log "start full300 chain run=${CHAIN_RUN_TAG}"
wait_for_prev_posttrain

log "launch pretrain run=${CHAIN_RUN_TAG}"
(
  set -euo pipefail
  cd "${ROOT_DIR}"
  export RUN_TAG="${CHAIN_RUN_TAG}"
  export SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
  export SAVE_DIR="${SAVE_ROOT}/${RUN_TAG}"
  export LOG_ROOT="${PRETRAIN_LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher}"
  export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
  export MIX_CONFIG="${MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml}"
  export CUDA_VISIBLE_DEVICES="${PRETRAIN_VISIBLE_GPUS:-0,1,2,3}"
  export NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-4}"
  export MASTER_PORT="${PRETRAIN_MASTER_PORT:-29634}"
  export EPOCHS="${PRETRAIN_EPOCHS:-300}"
  export BATCH="${PRETRAIN_BATCH:-32}"
  export NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-4}"
  export SEED="${PRETRAIN_SEED:-0}"
  export LR="${PRETRAIN_LR:-3e-4}"
  export WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.05}"
  export MAX_STEPS="${PRETRAIN_MAX_STEPS:--1}"
  export SAVE_EVERY="${PRETRAIN_SAVE_EVERY:-10}"
  export SAVE_EVERY_STEPS="${PRETRAIN_SAVE_EVERY_STEPS:-0}"
  export LR_SCHEDULER="${PRETRAIN_LR_SCHEDULER:-cosine}"
  export WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:--1}"
  export WARMUP_RATIO="${PRETRAIN_WARMUP_RATIO:-0.05}"
  export MIN_LR="${PRETRAIN_MIN_LR:-1e-6}"
  export MAX_GRAD_NORM="${PRETRAIN_MAX_GRAD_NORM:-1.0}"
  export N_CTX="${PRETRAIN_N_CTX:-2048}"
  export N_QRY="${PRETRAIN_N_QRY:-64}"
  export QUERY_ORDER="${PRETRAIN_QUERY_ORDER:-shuffled}"
  export CODEC_VERSION="${PRETRAIN_CODEC_VERSION:-cqa_v2}"
  export D_MODEL="${PRETRAIN_D_MODEL:-384}"
  export N_LAYERS="${PRETRAIN_N_LAYERS:-12}"
  export N_HEADS="${PRETRAIN_N_HEADS:-6}"
  export MLP_RATIO="${PRETRAIN_MLP_RATIO:-4.0}"
  export DROPOUT="${PRETRAIN_DROPOUT:-0.0}"
  export DROP_PATH="${PRETRAIN_DROP_PATH:-0.0}"
  export BACKBONE_IMPL="${PRETRAIN_BACKBONE_IMPL:-nepa2d}"
  export NUM_GROUPS="${PRETRAIN_NUM_GROUPS:-64}"
  export GROUP_SIZE="${PRETRAIN_GROUP_SIZE:-32}"
  export PATCH_CENTER_MODE="${PRETRAIN_PATCH_CENTER_MODE:-fps}"
  export PATCH_FPS_RANDOM_START="${PRETRAIN_PATCH_FPS_RANDOM_START:-1}"
  export LOCAL_ENCODER="${PRETRAIN_LOCAL_ENCODER:-pointmae_conv}"
  export QUERY_TYPE_VOCAB="${PRETRAIN_QUERY_TYPE_VOCAB:-0}"
  export ANSWER_VOCAB="${PRETRAIN_ANSWER_VOCAB:-0}"
  export GENERATOR_DEPTH="${PRETRAIN_GENERATOR_DEPTH:-2}"
  export MODEL_ARCH="${PRETRAIN_MODEL_ARCH:-prefixlm}"
  export DECODER_LAYERS="${PRETRAIN_DECODER_LAYERS:-4}"
  export EXTERNAL_BACKBONE_CKPT="${PRETRAIN_EXTERNAL_BACKBONE_CKPT:-}"
  export FREEZE_EXTERNAL_ENCODER="${PRETRAIN_FREEZE_EXTERNAL_ENCODER:-1}"
  export EXTERNAL_BACKBONE_DEPTH="${PRETRAIN_EXTERNAL_BACKBONE_DEPTH:-12}"
  export EXTERNAL_BACKBONE_HEADS="${PRETRAIN_EXTERNAL_BACKBONE_HEADS:-6}"
  export EXTERNAL_BACKBONE_DROP_PATH="${PRETRAIN_EXTERNAL_BACKBONE_DROP_PATH:-0.1}"
  export ANSWER_FACTORIZATION="${PRETRAIN_ANSWER_FACTORIZATION:-independent}"
  export QUERY_INTERFACE_MODE="${PRETRAIN_QUERY_INTERFACE_MODE:-no_q}"
  export HEAD_MODE="${PRETRAIN_HEAD_MODE:-multihead}"
  export SAMPLING_PROTOCOL="${PRETRAIN_SAMPLING_PROTOCOL:-packed}"
  export LOSS_BALANCE="${PRETRAIN_LOSS_BALANCE:-per_task}"
  export USE_WANDB="${PRETRAIN_USE_WANDB:-1}"
  export WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-patchnepa-cqa-pretrain}"
  export WANDB_ENTITY="${PRETRAIN_WANDB_ENTITY:-}"
  export WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-itachi_geo_teacher_full300}"
  export WANDB_RUN_NAME="${CHAIN_RUN_TAG}"
  export WANDB_TAGS="${PRETRAIN_WANDB_TAGS:-local,itachi,cqa,geo_teacher,distnorm_unsigned,full300}"
  export WANDB_MODE="${PRETRAIN_WANDB_MODE:-online}"
  export WANDB_LOG_EVERY="${PRETRAIN_WANDB_LOG_EVERY:-10}"
  export WANDB_DIR="${PRETRAIN_WANDB_DIR:-${ROOT_DIR}/wandb_cqa}"
  export OMP_NUM_THREADS="${PRETRAIN_OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${PRETRAIN_MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${PRETRAIN_OPENBLAS_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${PRETRAIN_NUMEXPR_NUM_THREADS:-1}"
  export FOREGROUND=1
  env -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
    bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/run_geo_teacher_pretrain_local.sh"
) >> "${LOG_FILE}" 2>&1

log "launch downstream chain pretrain_run=${CHAIN_RUN_TAG}"
(
  set -euo pipefail
  cd "${ROOT_DIR}"
  export PRETRAIN_RUN_TAG="${CHAIN_RUN_TAG}"
  export PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}/${PRETRAIN_RUN_TAG}"
  export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
  export RUN_ROUTE_A="${RUN_ROUTE_A:-1}"
  export RUN_SHAPENETPART_FT="${RUN_SHAPENETPART_FT:-1}"
  export RUN_ROUTE_B="${RUN_ROUTE_B:-1}"
  export FT_VISIBLE_GPUS="${FT_VISIBLE_GPUS:-0,1,2,3}"
  export FT_NPROC_PER_NODE="${FT_NPROC_PER_NODE:-4}"
  export PARTSEG_VISIBLE_GPUS="${PARTSEG_VISIBLE_GPUS:-0,1,2,3}"
  export PARTSEG_NPROC_PER_NODE="${PARTSEG_NPROC_PER_NODE:-4}"
  export FT_DATA_FORMAT="${FT_DATA_FORMAT:-scan_h5}"
  export FOREGROUND=1
  env -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
    bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/run_geo_teacher_posttrain_local.sh"
) >> "${LOG_FILE}" 2>&1

log "full300 chain completed run=${CHAIN_RUN_TAG}"
