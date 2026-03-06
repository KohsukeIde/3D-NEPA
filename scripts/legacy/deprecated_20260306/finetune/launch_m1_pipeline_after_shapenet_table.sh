#!/usr/bin/env bash
set -eu

# Deprecated helper:
# waits for scan_shapenet_table completion, then launches archived M1 local chains.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

LOG_DIR="${LOG_DIR:-logs/finetune/m1_pipeline}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/pipeline.log}"
PIPELINE_PID="${PIPELINE_PID:-${LOG_DIR}/pipeline.pid}"

mkdir -p "${LOG_DIR}"

echo "[warn] deprecated launcher: kept only for archived local-chain flow"

nohup bash "${SCRIPT_DIR}/run_m1_pipeline_after_shapenet_table.sh" > "${PIPELINE_LOG}" 2>&1 &
echo $! > "${PIPELINE_PID}"

echo "started pid=$(cat "${PIPELINE_PID}")"
echo "pipeline_log=${PIPELINE_LOG}"
