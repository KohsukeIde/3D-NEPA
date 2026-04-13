#!/usr/bin/env bash
set -euo pipefail

QUEUE_NAME="${QUEUE_NAME:-geopcp_routea_compare_v1}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/compare_queue}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${QUEUE_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${QUEUE_NAME}.pid}"
TMUX_SESSION="${TMUX_SESSION:-${QUEUE_NAME//[^[:alnum:]_]/_}}"

status="stopped"
pid=""
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    status="running"
  fi
fi
if [[ "${status}" != "running" ]] && command -v tmux >/dev/null 2>&1 && env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
  status="running"
fi

echo "status=${status}"
echo "pid=${pid}"
echo "queue=${QUEUE_NAME}"
echo "log=${LOG_FILE}"
