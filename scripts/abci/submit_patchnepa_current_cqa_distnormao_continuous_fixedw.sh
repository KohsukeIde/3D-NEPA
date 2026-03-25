#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_distnormao_continuous_fixedw_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_distnormao_continuous_fixedw_g2_s10000}"

env RUN_SET="${RUN_SET}" TRAIN_RUN_TAG="${TRAIN_RUN_TAG}" \
TASK_LOSS_BALANCE="${TASK_LOSS_BALANCE:-fixed}" \
LOSS_WEIGHT_DISTANCE="${LOSS_WEIGHT_DISTANCE:-10.0}" \
LOSS_WEIGHT_AO="${LOSS_WEIGHT_AO:-1.0}" \
LOSS_WEIGHT_NORMAL="${LOSS_WEIGHT_NORMAL:-0.3}" \
SUITE_RUN_TAG="${SUITE_RUN_TAG:-cqa_distnormao_continuous_fixedw_suite}" \
bash scripts/abci/submit_patchnepa_current_cqa_distnormao_continuous.sh
