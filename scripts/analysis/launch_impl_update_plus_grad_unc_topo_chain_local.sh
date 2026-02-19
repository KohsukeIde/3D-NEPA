#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

mkdir -p logs/analysis
RUN_ID="${RUN_ID:-}"
if [ -n "${RUN_ID}" ]; then
  LOG_PATH_DEFAULT="logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline_${RUN_ID}.log"
  PID_PATH_DEFAULT="logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline_${RUN_ID}.pid"
else
  LOG_PATH_DEFAULT="logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline.log"
  PID_PATH_DEFAULT="logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline.pid"
fi
LOG_PATH="${LOG_PATH:-${LOG_PATH_DEFAULT}}"
PID_PATH="${PID_PATH:-${PID_PATH_DEFAULT}}"
mkdir -p "$(dirname "${LOG_PATH}")"

if [ -f "${PID_PATH}" ]; then
  old_pid="$(cat "${PID_PATH}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[abort] existing process is still alive: pid=${old_pid} (${PID_PATH})"
    exit 1
  fi
fi

nohup bash scripts/analysis/run_impl_update_plus_grad_unc_topo_chain_local.sh > "${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_PATH}"
echo "[launched] pid=${pid} log=${LOG_PATH}"
