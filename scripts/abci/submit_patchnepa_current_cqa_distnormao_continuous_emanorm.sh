#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_distnormao_continuous_emanorm_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_distnormao_continuous_emanorm_g2_s10000}"

env RUN_SET="${RUN_SET}" TRAIN_RUN_TAG="${TRAIN_RUN_TAG}" \
TASK_LOSS_BALANCE="${TASK_LOSS_BALANCE:-ema_norm}" \
LOSS_EMA_MOMENTUM="${LOSS_EMA_MOMENTUM:-0.99}" \
LOSS_EMA_EPS="${LOSS_EMA_EPS:-1e-6}" \
SUITE_RUN_TAG="${SUITE_RUN_TAG:-cqa_distnormao_continuous_emanorm_suite}" \
bash scripts/abci/submit_patchnepa_current_cqa_distnormao_continuous.sh
