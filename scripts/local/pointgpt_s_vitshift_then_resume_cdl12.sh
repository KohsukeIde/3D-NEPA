#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
VITSHIFT_SCRIPT="${SCRIPT_DIR}/pointgpt_s_pointnepa_vitshift_ablation.sh"
S_MATRIX_SCRIPT="${SCRIPT_DIR}/pointgpt_s_objective_matrix.sh"
POLL_SEC="${POLL_SEC:-60}"

MASKOFF_RUN_TAG="${MASKOFF_RUN_TAG:-pointnepa_s_maskoff_20260403_212525}"
MASKOFF_LOG_ROOT="${MASKOFF_LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_s_pointnepa_mask_ablation}"
VITSHIFT_RUN_TAG="${VITSHIFT_RUN_TAG:-pointnepa_s_vitshift_maskoff_20260403_$(date +%H%M%S)}"
S_MATRIX_RUN_TAG="${S_MATRIX_RUN_TAG:-pointgpt_s_ft_recipe_matrix_2x2_20260318}"

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "[wait] missing file: ${path}"
    sleep "${POLL_SEC}"
  done
}

if [[ ! -x "${VITSHIFT_SCRIPT}" ]]; then
  echo "[error] missing vitshift script: ${VITSHIFT_SCRIPT}"
  exit 2
fi
if [[ ! -x "${S_MATRIX_SCRIPT}" ]]; then
  echo "[error] missing S matrix script: ${S_MATRIX_SCRIPT}"
  exit 2
fi

MASKOFF_SUMMARY_PATH="${MASKOFF_LOG_ROOT}/${MASKOFF_RUN_TAG}_summary.md"
wait_for_file "${MASKOFF_SUMMARY_PATH}.done"

RUN_TAG="${VITSHIFT_RUN_TAG}" \
BASELINE_SUMMARY_PATH="${MASKOFF_SUMMARY_PATH}" \
bash "${VITSHIFT_SCRIPT}"

RUN_TAG="${S_MATRIX_RUN_TAG}" \
bash "${S_MATRIX_SCRIPT}"
