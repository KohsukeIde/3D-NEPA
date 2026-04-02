#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.pid}"
TMUX_SESSION="${TMUX_SESSION:-patchnepa_scanobjectnn_variants}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR LOG_ROOT LOG_FILE PID_FILE
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export MAIN_SPLIT_DIR="${MAIN_SPLIT_DIR:-data/ScanObjectNN/h5_files/main_split}"
export MAIN_SPLIT_NO_BG_DIR="${MAIN_SPLIT_NO_BG_DIR:-data/ScanObjectNN/h5_files/main_split_nobg}"
export VARIANT_H5_ROOT="${VARIANT_H5_ROOT:-data/ScanObjectNN/h5_files_protocol_variants}"
export OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
export OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
export PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
export WORKERS="${WORKERS:-8}"
export PT_FPS_WORKERS="${PT_FPS_WORKERS:-8}"
export NORMALIZE_PC="${NORMALIZE_PC:-0}"
export QUERY_BBOX_MODE="${QUERY_BBOX_MODE:-auto}"
export ENABLE_PT_FPS_ORDER="${ENABLE_PT_FPS_ORDER:-1}"
export NICE_LEVEL="${NICE_LEVEL:-10}"
export IONICE_CLASS="${IONICE_CLASS:-2}"
export IONICE_LEVEL="${IONICE_LEVEL:-7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] ScanObjectNN variant build already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting in foreground"
  echo "[info] log=${LOG_FILE}"
  exec bash "${ROOT_DIR}/scripts/local/patchnepa_data/_run_scanobjectnn_variants_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] ScanObjectNN variant build already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    for name in \
      ROOT_DIR LOG_ROOT LOG_FILE PID_FILE PYTHON_BIN MAIN_SPLIT_DIR MAIN_SPLIT_NO_BG_DIR \
      VARIANT_H5_ROOT OBJ_BG_CACHE OBJ_ONLY_CACHE PB_T50_RS_CACHE WORKERS PT_FPS_WORKERS \
      NORMALIZE_PC QUERY_BBOX_MODE ENABLE_PT_FPS_ORDER NICE_LEVEL IONICE_CLASS IONICE_LEVEL \
      OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS
    do
      tmux set-environment -g "${name}" "${!name:-}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_data/_run_scanobjectnn_variants_inner.sh'"
    sleep 1
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached ScanObjectNN variant build in tmux"
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
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_data/_run_scanobjectnn_variants_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached ScanObjectNN variant build"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached ScanObjectNN variant build"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
