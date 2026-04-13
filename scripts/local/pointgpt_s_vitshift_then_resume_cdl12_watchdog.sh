#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
CHAIN_SCRIPT="${CHAIN_SCRIPT:-${SCRIPT_DIR}/pointgpt_s_vitshift_then_resume_cdl12.sh}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_s_watchdog}"
POLL_SEC="${POLL_SEC:-60}"
RETRY_SEC="${RETRY_SEC:-30}"

MASKOFF_RUN_TAG="${MASKOFF_RUN_TAG:-pointnepa_s_maskoff_20260403_212525}"
VITSHIFT_RUN_TAG="${VITSHIFT_RUN_TAG:-pointnepa_s_vitshift_maskoff_20260403_221453}"
S_MATRIX_RUN_TAG="${S_MATRIX_RUN_TAG:-pointgpt_s_ft_recipe_matrix_2x2_20260318}"

FINAL_DONE_PATH="${FINAL_DONE_PATH:-${WORKDIR}/logs/local/pointgpt_s_objective_matrix/${S_MATRIX_RUN_TAG}_summary.md.done}"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${CHAIN_SCRIPT}" ]]; then
  echo "[error] missing chain script: ${CHAIN_SCRIPT}"
  exit 2
fi

attempt=1
while true; do
  if [[ -f "${FINAL_DONE_PATH}" ]]; then
    echo "[done] watchdog observed final completion: ${FINAL_DONE_PATH}"
    exit 0
  fi

  log_path="${LOG_ROOT}/pointgpt_s_vitshift_then_resume_cdl12_watchdog_attempt$(printf '%03d' "${attempt}")_$(date +%Y%m%d_%H%M%S).log"
  echo "[watchdog] attempt=${attempt} log=${log_path}"

  set +e
  WORKDIR="${WORKDIR}" \
  POLL_SEC="${POLL_SEC}" \
  MASKOFF_RUN_TAG="${MASKOFF_RUN_TAG}" \
  VITSHIFT_RUN_TAG="${VITSHIFT_RUN_TAG}" \
  S_MATRIX_RUN_TAG="${S_MATRIX_RUN_TAG}" \
  bash "${CHAIN_SCRIPT}" |& tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ -f "${FINAL_DONE_PATH}" ]]; then
    echo "[done] watchdog observed final completion after attempt=${attempt}: ${FINAL_DONE_PATH}"
    exit 0
  fi

  echo "[watchdog] chain exited rc=${rc} without final done. retry in ${RETRY_SEC}s"
  sleep "${RETRY_SEC}"
  attempt=$((attempt + 1))
done
