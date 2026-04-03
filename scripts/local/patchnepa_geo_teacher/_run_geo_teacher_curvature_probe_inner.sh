#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post/cqa_probe}"
RUN_TAG="${RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main__curvature_probe}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.launch.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${PRETRAIN_SAVE_ROOT}/${PRETRAIN_RUN_TAG}}"
CKPT_NAME="${CKPT_NAME:-ckpt_final.pt}"
CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/${CKPT_NAME}}"
GPU_ID="${GPU_ID:-3}"

mkdir -p "${LOG_ROOT}" "${SAVE_DIR:-${ROOT_DIR}/runs/cqa_probe_itachi}" "$(dirname "${OUT_JSON:-${ROOT_DIR}/results/cqa_probe_itachi/${RUN_TAG}.json}")"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}

trap cleanup EXIT

echo "$$" > "${PID_FILE}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

log "start curvature probe run=${RUN_TAG} gpu=${GPU_ID}"
if [[ ! -f "${CKPT_PATH}" ]]; then
  log "missing checkpoint: ${CKPT_PATH}"
  exit 1
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export WORKDIR="${ROOT_DIR}"
export VENV_ACTIVATE=""
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CKPT="${CKPT_PATH}"
export CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260401_worldvis}"
export PROBE_TARGET="${PROBE_TARGET:-curvature}"
export MANIFEST_JSON="${MANIFEST_JSON:-}"
export SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/cqa_probe_itachi}"
export OUT_JSON="${OUT_JSON:-${ROOT_DIR}/results/cqa_probe_itachi/${RUN_TAG}.json}"
export LOG_ROOT="${LOG_ROOT}"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export EVAL_SPLIT="${EVAL_SPLIT:-test}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export EVAL_EVERY="${EVAL_EVERY:-500}"
export BATCH="${BATCH:-8}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SEED="${SEED:-0}"
export LR="${LR:-1e-3}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
export N_CTX="${N_CTX:-2048}"
export N_QRY="${N_QRY:-64}"
export TRAIN_QUERY_ORDER="${TRAIN_QUERY_ORDER:-shuffled}"
export EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
export EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
export CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
export RUN_TAG="${RUN_TAG}"

bash "${ROOT_DIR}/scripts/eval/nepa3d_cqa_geo_probe_qg.sh" 2>&1 | tee -a "${LOG_FILE}"
