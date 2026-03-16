#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

POLL_SEC="${POLL_SEC:-60}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TEST_CUDA_VISIBLE_DEVICES="${TEST_CUDA_VISIBLE_DEVICES:-0}"
WANDB_MODE="${WANDB_MODE:-online}"

PROTOCOL_RUN_TAG="${PROTOCOL_RUN_TAG:-pointgpt_protocol_compare_official_b_20260312}"
PROTOCOL_SOURCE_LABEL="${PROTOCOL_SOURCE_LABEL:-pointgpt_b_postpretrain_official}"
PROTOCOL_CKPT_PATH="${PROTOCOL_CKPT_PATH:-${WORKDIR}/PointGPT/checkpoints/official/pointgpt_b_post_pretrain_official.pth}"

MATRIX_RUN_TAG="${MATRIX_RUN_TAG:-pointgpt_ft_recipe_matrix_2x2_20260311_153835}"
MATRIX_NEPA_CLS_ONLY_RUN_TAG="${MATRIX_NEPA_CLS_ONLY_RUN_TAG:-pointgpt_nepa_cosine_ft_clsonly_sparseckpt_20260311_132956}"

wait_for_no_protocol_compare_jobs() {
  while pgrep -af -- "${PROTOCOL_RUN_TAG}" >/dev/null; do
    echo "[wait] protocol_compare still active for ${PROTOCOL_RUN_TAG}"
    sleep "${POLL_SEC}"
  done
}

echo "=== POINTGPT RESUME REMAINING ALL ==="
echo "date=$(date -Is)"
echo "protocol_run_tag=${PROTOCOL_RUN_TAG}"
echo "matrix_run_tag=${MATRIX_RUN_TAG}"
echo

wait_for_no_protocol_compare_jobs

RUN_TAG="${PROTOCOL_RUN_TAG}" \
SOURCE_LABEL="${PROTOCOL_SOURCE_LABEL}" \
CKPT_PATH="${PROTOCOL_CKPT_PATH}" \
WANDB_MODE="${WANDB_MODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
TEST_CUDA_VISIBLE_DEVICES="${TEST_CUDA_VISIBLE_DEVICES}" \
bash "${SCRIPT_DIR}/pointgpt_protocol_compare.sh"

RUN_TAG="${MATRIX_RUN_TAG}" \
NEPA_CLS_ONLY_RUN_TAG="${MATRIX_NEPA_CLS_ONLY_RUN_TAG}" \
WANDB_MODE="${WANDB_MODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
bash "${SCRIPT_DIR}/pointgpt_ft_recipe_matrix_2x2.sh"

