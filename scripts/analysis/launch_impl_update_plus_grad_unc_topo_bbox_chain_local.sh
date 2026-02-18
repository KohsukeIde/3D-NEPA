#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

mkdir -p logs/analysis
LOG_PATH="${LOG_PATH:-logs/analysis/impl_update_plus_grad_unc_topo_bbox_chain/pipeline.log}"
PID_PATH="${PID_PATH:-logs/analysis/impl_update_plus_grad_unc_topo_bbox_chain/pipeline.pid}"
ACTIVE_PATH="${ACTIVE_PATH:-logs/analysis/impl_update_plus_grad_unc_topo_bbox_chain/active.pid}"
mkdir -p "$(dirname "${LOG_PATH}")"

setsid bash scripts/analysis/run_impl_update_plus_grad_unc_topo_bbox_chain_local.sh > "${LOG_PATH}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PID_PATH}"
cp "${PID_PATH}" "${ACTIVE_PATH}"
echo "[launched] pid=${pid} log=${LOG_PATH}"
