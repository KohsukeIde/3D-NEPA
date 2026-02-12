#!/usr/bin/env bash
set -eu

# Legacy helper:
# waits for scan_shapenet_table completion, then launches M1 pretrains + M1 fine-tune.
# Current preferred path is:
#   1) scripts/pretrain/launch_shapenet_m1_pretrains_local.sh
#   2) scripts/finetune/launch_scanobjectnn_m1_after_pretrain.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

LOG_DIR="${LOG_DIR:-logs/finetune/m1_pipeline}"
PIPELINE_LOG="${PIPELINE_LOG:-${LOG_DIR}/pipeline.log}"
PIPELINE_PID="${PIPELINE_PID:-${LOG_DIR}/pipeline.pid}"

mkdir -p "${LOG_DIR}"

echo "[warn] legacy launcher: use launch_scanobjectnn_m1_after_pretrain.sh for current M1 flow"

nohup bash scripts/finetune/run_m1_pipeline_after_shapenet_table.sh > "${PIPELINE_LOG}" 2>&1 &
echo $! > "${PIPELINE_PID}"

echo "started pid=$(cat "${PIPELINE_PID}")"
echo "pipeline_log=${PIPELINE_LOG}"
