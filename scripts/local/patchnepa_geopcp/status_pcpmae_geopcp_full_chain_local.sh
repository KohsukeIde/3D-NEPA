#!/usr/bin/env bash
set -euo pipefail

ARM_NAME="${ARM_NAME:-pcp_worldvis_base_100ep}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ARM_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${ARM_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${ARM_NAME//[^[:alnum:]_]/_}_chain}"

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
echo "arm=${ARM_NAME}"
echo "log=${LOG_FILE}"
