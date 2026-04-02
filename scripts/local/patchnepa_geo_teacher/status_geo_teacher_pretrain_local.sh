#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
SAVE_ROOT="${SAVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/runs/cqa_itachi}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_TAG}}"
DEFAULT_SESSION="${RUN_TAG//[^[:alnum:]_]/_}"
TMUX_SESSION="${TMUX_SESSION:-${DEFAULT_SESSION}}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

running=0
pid=""
tmux_running=0
if [[ "${LAUNCH_MODE}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux_running=1
  fi
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    running=1
  fi
fi

echo "run_tag=${RUN_TAG}"
echo "save_dir=${SAVE_DIR}"
echo "launch_mode=${LAUNCH_MODE}"
if [[ "${tmux_running}" == "1" ]]; then
  echo "status=running"
  echo "session=${TMUX_SESSION}"
  [[ "${running}" == "1" ]] && echo "pid=${pid}"
elif [[ "${running}" == "1" ]]; then
  echo "status=running"
  echo "pid=${pid}"
else
  echo "status=stopped"
  [[ -n "${pid}" ]] && echo "last_pid=${pid}"
fi
echo "log=${LOG_FILE}"

if [[ -d "${SAVE_DIR}" ]]; then
  echo "--- save dir ---"
  find "${SAVE_DIR}" -maxdepth 1 -type f | sort
fi

if [[ -f "${LOG_FILE}" ]]; then
  echo "--- log tail ---"
  tail -n 40 "${LOG_FILE}"
fi
