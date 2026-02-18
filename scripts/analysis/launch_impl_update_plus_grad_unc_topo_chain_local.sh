#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

mkdir -p logs/analysis
LOG_PATH="${LOG_PATH:-logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline.log}"
PID_PATH="${PID_PATH:-logs/analysis/impl_update_plus_grad_unc_topo_chain/pipeline.pid}"
mkdir -p "$(dirname "${LOG_PATH}")"

nohup bash scripts/analysis/run_impl_update_plus_grad_unc_topo_chain_local.sh > "${LOG_PATH}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_PATH}"
echo "[launched] pid=${pid} log=${LOG_PATH}"
