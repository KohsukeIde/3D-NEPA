#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

LOG_DIR="${LOG_DIR:-logs/finetune/m1_after_pretrain}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/pipeline.log}"
PIPELINE_PID="${PIPELINE_PID:-${LOG_DIR}/pipeline.pid}"

mkdir -p "${LOG_DIR}"

nohup bash scripts/finetune/run_scanobjectnn_m1_after_pretrain.sh > "${PIPELINE_LOG}" 2>&1 &
echo $! > "${PIPELINE_PID}"

echo "started pid=$(cat "${PIPELINE_PID}")"
echo "pipeline_log=${PIPELINE_LOG}"
