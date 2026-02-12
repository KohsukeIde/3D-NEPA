#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PRE_PID_FILE="${PRE_PID_FILE:-logs/finetune/scan_shapenet_table/pipeline.pid}"
PRE_LOG_FILE="${PRE_LOG_FILE:-logs/finetune/scan_shapenet_table/pipeline.log}"
POLL_SEC="${POLL_SEC:-60}"
REQUIRE_SUCCESS="${REQUIRE_SUCCESS:-1}"

PRETRAIN_LAUNCH="${PRETRAIN_LAUNCH:-scripts/pretrain/launch_shapenet_m1_pretrains_local.sh}"
PRETRAIN_PID_FILE="${PRETRAIN_PID_FILE:-logs/pretrain/m1/pipeline.pid}"
PRETRAIN_LOG_FILE="${PRETRAIN_LOG_FILE:-logs/pretrain/m1/pipeline.log}"

FINETUNE_LAUNCH="${FINETUNE_LAUNCH:-scripts/finetune/launch_scanobjectnn_m1_table_local.sh}"

if [ ! -f "${PRE_PID_FILE}" ]; then
  echo "[error] shapenet-table pid file not found: ${PRE_PID_FILE}"
  exit 1
fi

pre_pid="$(cat "${PRE_PID_FILE}")"
echo "[wait] scan_shapenet_table pid=${pre_pid}"
while kill -0 "${pre_pid}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done
echo "[done] scan_shapenet_table process exited"

if [ "${REQUIRE_SUCCESS}" = "1" ]; then
  if ! rg -q "\\[info\\] completed rc_gpu0=0 rc_gpu1=0" "${PRE_LOG_FILE}"; then
    echo "[abort] scan_shapenet_table did not report successful completion"
    exit 1
  fi
fi

echo "[start] launching M1 pretrains"
bash "${PRETRAIN_LAUNCH}"

if [ ! -f "${PRETRAIN_PID_FILE}" ]; then
  echo "[error] M1 pretrain pid file not found after launch: ${PRETRAIN_PID_FILE}"
  exit 1
fi

pt_pid="$(cat "${PRETRAIN_PID_FILE}")"
echo "[wait] M1 pretrains pid=${pt_pid}"
while kill -0 "${pt_pid}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done
echo "[done] M1 pretrains process exited"

if ! rg -q "\\[info\\] completed rc_gpu0=0 rc_gpu1=0" "${PRETRAIN_LOG_FILE}"; then
  echo "[abort] M1 pretrains did not report successful completion"
  exit 1
fi

echo "[start] launching M1 ScanObjectNN fine-tune table"
bash "${FINETUNE_LAUNCH}"
