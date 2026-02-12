#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PRE_PID_FILE="${PRE_PID_FILE:-logs/pretrain/m1/pipeline.pid}"
PRE_LOG_FILE="${PRE_LOG_FILE:-logs/pretrain/m1/pipeline.log}"
POLL_SEC="${POLL_SEC:-60}"
REQUIRE_SUCCESS="${REQUIRE_SUCCESS:-1}"

FINETUNE_LAUNCH="${FINETUNE_LAUNCH:-scripts/finetune/launch_scanobjectnn_m1_table_local.sh}"

if [ ! -f "${PRE_PID_FILE}" ]; then
  echo "[error] M1 pretrain pid file not found: ${PRE_PID_FILE}"
  exit 1
fi

pid="$(cat "${PRE_PID_FILE}")"
echo "[wait] M1 pretrain pid=${pid}"
while kill -0 "${pid}" 2>/dev/null; do
  sleep "${POLL_SEC}"
done
echo "[done] M1 pretrain process exited"

if [ "${REQUIRE_SUCCESS}" = "1" ]; then
  if ! rg -q "\\[info\\] completed rc_gpu0=0 rc_gpu1=0" "${PRE_LOG_FILE}"; then
    echo "[abort] M1 pretrain did not report successful completion"
    exit 1
  fi
fi

echo "[start] launching ScanObjectNN M1 fine-tune table"
bash "${FINETUNE_LAUNCH}"
