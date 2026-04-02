#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/shapenet_worldvis_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/shapenet_worldvis_local.pid}"
OUT_ROOT="${OUT_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/shapenet_cache_v2_20260401_worldvis}"
TMUX_SESSION="${TMUX_SESSION:-patchnepa_worldvis}"
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

count=0
if [[ -d "${OUT_ROOT}" ]]; then
  count="$(find "${OUT_ROOT}" -type f -name '*.npz' | wc -l)"
fi

echo "out_root=${OUT_ROOT}"
echo "npz_count=${count}"
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
