#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PRE_PID_FILE="${PRE_PID_FILE:-logs/finetune/scan_shapenet_table/pipeline.pid}"
PRE_LOG_FILE="${PRE_LOG_FILE:-logs/finetune/scan_shapenet_table/pipeline.log}"
POLL_SEC="${POLL_SEC:-60}"
REQUIRE_SUCCESS="${REQUIRE_SUCCESS:-1}"

MAIN_PID_FILE="${MAIN_PID_FILE:-logs/finetune/scan_main_table/pipeline.pid}"
# Safety default: no automatic follow-up launch.
# Explicitly set MAIN_LAUNCH to run a specific next stage.
MAIN_LAUNCH="${MAIN_LAUNCH:-}"

# Backward compatibility for older flat log paths.
if [ ! -f "${PRE_PID_FILE}" ] && [ -f "logs/scan_shapenet_table_pipeline.pid" ]; then
  PRE_PID_FILE="logs/scan_shapenet_table_pipeline.pid"
fi
if [ ! -f "${PRE_LOG_FILE}" ] && [ -f "logs/scan_shapenet_table_pipeline.log" ]; then
  PRE_LOG_FILE="logs/scan_shapenet_table_pipeline.log"
fi

if [ ! -f "${PRE_PID_FILE}" ]; then
  echo "[error] shapenet-table pid file not found: ${PRE_PID_FILE}"
  exit 1
fi

pid="$(cat "${PRE_PID_FILE}")"
echo "[wait] scan_shapenet_table pid=${pid}"
while kill -0 "${pid}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done
echo "[done] scan_shapenet_table process exited"

if [ "${REQUIRE_SUCCESS}" = "1" ]; then
  if ! rg -q "\\[info\\] completed rc_gpu0=0 rc_gpu1=0" "${PRE_LOG_FILE}"; then
    echo "[abort] shapenet-table did not report successful completion"
    exit 1
  fi
fi

if [ -z "${MAIN_LAUNCH}" ]; then
  echo "[done] shapenet-table completed; no follow-up launch configured (MAIN_LAUNCH is empty)"
  exit 0
fi

if [ -f "${MAIN_PID_FILE}" ]; then
  main_pid="$(cat "${MAIN_PID_FILE}" || true)"
  if [ -n "${main_pid}" ] && kill -0 "${main_pid}" 2>/dev/null; then
    echo "[skip] mixed/main-table already running pid=${main_pid}"
    exit 0
  fi
fi

echo "[start] launching mixed/main-table fine-tune"
bash "${MAIN_LAUNCH}"
