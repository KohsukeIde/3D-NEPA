#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CHAIN_RUN_TAG="${CHAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_300ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_RUN_TAG//[^[:alnum:]_]/_}_chain}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"

running=0
pid=""
tmux_running=0
if [[ "${LAUNCH_MODE}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    tmux_running=1
  fi
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    running=1
  fi
fi

echo "chain_run_tag=${CHAIN_RUN_TAG}"
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
if [[ -f "${LOG_FILE}" ]]; then
  echo "--- log tail ---"
  tail -n 20 "${LOG_FILE}"
fi
