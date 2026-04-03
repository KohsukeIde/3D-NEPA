#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
RUN_TAG="${RUN_TAG:-${PRETRAIN_RUN_TAG}__curvature_probe}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post/cqa_probe}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.launch.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
TMUX_SESSION="${TMUX_SESSION:-${RUN_TAG//[^[:alnum:]_]/_}}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PRETRAIN_RUN_TAG RUN_TAG LOG_ROOT LOG_FILE PID_FILE
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
export PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${PRETRAIN_SAVE_ROOT}/${PRETRAIN_RUN_TAG}}"
export CKPT_NAME="${CKPT_NAME:-ckpt_final.pt}"
export CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/${CKPT_NAME}}"
export GPU_ID="${GPU_ID:-3}"
export CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260401_worldvis}"
export PROBE_TARGET="${PROBE_TARGET:-curvature}"
export MANIFEST_JSON="${MANIFEST_JSON:-}"
export SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/cqa_probe_itachi}"
export RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/cqa_probe_itachi}"
export OUT_JSON="${OUT_JSON:-${RESULT_ROOT}/${RUN_TAG}.json}"
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

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] curvature probe already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting curvature probe in foreground"
  echo "[info] log=${LOG_FILE}"
  bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_curvature_probe_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] curvature probe already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    for name in \
      ROOT_DIR PRETRAIN_RUN_TAG RUN_TAG LOG_ROOT LOG_FILE PID_FILE PYTHON_BIN \
      PRETRAIN_SAVE_ROOT PRETRAIN_SAVE_DIR CKPT_NAME CKPT_PATH GPU_ID CACHE_ROOT \
      PROBE_TARGET MANIFEST_JSON SAVE_DIR RESULT_ROOT OUT_JSON TRAIN_SPLIT \
      EVAL_SPLIT MAX_STEPS EVAL_EVERY BATCH NUM_WORKERS SEED LR WEIGHT_DECAY \
      N_CTX N_QRY TRAIN_QUERY_ORDER EVAL_QUERY_ORDER MAX_TRAIN_SAMPLES \
      MAX_EVAL_SAMPLES EVAL_SAMPLE_MODE CONTROLS
    do
      tmux set-environment -g "${name}" "${!name:-}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_curvature_probe_inner.sh'"
    sleep 1
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached curvature probe in tmux"
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
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_curvature_probe_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached curvature probe"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached curvature probe"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
