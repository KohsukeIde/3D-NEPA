#!/usr/bin/env bash
set -eu

# Wait current review chain, then launch ModelNet40 protocol pipeline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

wait_for_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [ ! -f "${pid_file}" ]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [ -z "${pid}" ]; then
    return 0
  fi
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "[wait] ${label} pid=${pid}"
    while ps -p "${pid}" >/dev/null 2>&1; do
      sleep 30
    done
    echo "[wait] ${label} finished"
  fi
}

REVIEW_PID_FILE="${REVIEW_PID_FILE:-logs/finetune/scan_variants_review_chain/pipeline.pid}"
MODELNET_LOG_DIR="${MODELNET_LOG_DIR:-logs/finetune/modelnet40_pointgpt_protocol}"
MODELNET_PID_FILE="${MODELNET_LOG_DIR}/pipeline.pid"

echo "[stage1] wait current review chain"
wait_for_pid_file "${REVIEW_PID_FILE}" "scan_variants_review_chain"

echo "[stage2] launch modelnet40 protocol"
if [ -f "${MODELNET_PID_FILE}" ] && ps -p "$(cat "${MODELNET_PID_FILE}")" >/dev/null 2>&1; then
  echo "[stage2] attach existing modelnet pipeline pid=$(cat "${MODELNET_PID_FILE}")"
else
  LOG_DIR="${MODELNET_LOG_DIR}" \
    bash scripts/finetune/launch_modelnet40_pointgpt_protocol_local.sh
fi
wait_for_pid_file "${MODELNET_PID_FILE}" "modelnet40_pointgpt_protocol"

echo "[done] post-review modelnet chain finished"
