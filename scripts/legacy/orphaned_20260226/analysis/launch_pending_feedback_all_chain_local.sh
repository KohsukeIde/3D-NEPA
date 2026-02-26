#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/analysis/pending_feedback_all_chain_${RUN_ID}}"
mkdir -p "${LOG_DIR}"

PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/pipeline.log}"
PIPELINE_PID="${PIPELINE_PID:-${LOG_DIR}/pipeline.pid}"
ACTIVE_PID="${ACTIVE_PID:-${LOG_DIR}/active.pid}"

setsid bash scripts/analysis/run_pending_feedback_all_chain_local.sh > "${PIPELINE_LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${PIPELINE_PID}"
echo "${pid}" > "${ACTIVE_PID}"

echo "[launched] run_id=${RUN_ID} pid=${pid}"
echo "  pipeline_log=${PIPELINE_LOG}"
echo "  pipeline_pid=${PIPELINE_PID}"
echo "  active_pid=${ACTIVE_PID}"

