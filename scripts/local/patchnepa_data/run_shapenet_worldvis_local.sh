#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/shapenet_worldvis_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/shapenet_worldvis_local.pid}"
TMUX_SESSION="${TMUX_SESSION:-patchnepa_worldvis}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

mkdir -p "${LOG_ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export SHAPENET_ROOT="${SHAPENET_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/ShapeNetCore.v2}"
export OUT_ROOT="${OUT_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/shapenet_cache_v2_20260401_worldvis}"

# Local workstation policy: keep canonical worldvis settings, but allow resume
# and defer long-tail meshes instead of stalling the whole run.
export WORKERS="${WORKERS:-16}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export MISSING_ONLY="${MISSING_ONLY:-1}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-900}"
export TASK_TIMEOUT_GRACE_SEC="${TASK_TIMEOUT_GRACE_SEC:-5}"
export NICE_LEVEL="${NICE_LEVEL:-10}"
export IONICE_CLASS="${IONICE_CLASS:-2}"
export IONICE_LEVEL="${IONICE_LEVEL:-7}"
export LOG_ROOT LOG_FILE PID_FILE

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] ShapeNet worldvis build already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting in foreground"
  echo "[info] log=${LOG_FILE}"
  exec bash "${ROOT_DIR}/scripts/preprocess/preprocess_shapenet_v2.sh" 2>&1 | tee -a "${LOG_FILE}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] ShapeNet worldvis build already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    for name in \
      PYTHON_BIN SHAPENET_ROOT OUT_ROOT WORKERS SKIP_EXISTING MISSING_ONLY \
      TASK_TIMEOUT_SEC TASK_TIMEOUT_GRACE_SEC LOG_ROOT LOG_FILE PID_FILE \
      OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS \
      NICE_LEVEL IONICE_CLASS IONICE_LEVEL
    do
      tmux set-environment -g "${name}" "${!name:-}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_data/_run_shapenet_worldvis_inner.sh'"
    sleep 1
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached ShapeNet worldvis build in tmux"
      echo "[info] session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    echo "[error] failed to start tmux session=${TMUX_SESSION}"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  nohup)
    {
      echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S %Z') start detached worldvis build"
      echo "[launcher] out_root=${OUT_ROOT}"
      echo "[launcher] workers=${WORKERS} timeout_sec=${TASK_TIMEOUT_SEC}"
    } >> "${LOG_FILE}"
    nohup bash -lc '
      bash "'"${ROOT_DIR}"'/scripts/preprocess/preprocess_shapenet_v2.sh"
      rc=$?
      printf "[launcher] %s exit_code=%s\n" "$(date "+%Y-%m-%d %H:%M:%S %Z")" "${rc}" >> "'"${LOG_FILE}"'"
      if [[ -f "'"${PID_FILE}"'" && "$(cat "'"${PID_FILE}"'" 2>/dev/null)" == "$$" ]]; then
        rm -f "'"${PID_FILE}"'"
      fi
      exit "${rc}"
    ' >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached ShapeNet worldvis build"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached ShapeNet worldvis build"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
