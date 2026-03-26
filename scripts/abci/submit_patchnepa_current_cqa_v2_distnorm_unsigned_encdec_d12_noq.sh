#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_SET="${RUN_SET:-patchnepa_cqa_v2_distnorm_unsigned_encdec_d12_noq_${STAMP}}"
export TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_v2_distnorm_unsigned_encdec_independent_d12_noq_g2_s10000}"
export MODEL_ARCH="${MODEL_ARCH:-encdec}"
export DECODER_LAYERS="${DECODER_LAYERS:-12}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-no_q}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,cqa_v2,distnorm_unsigned,encdec,d12,no_q,independent}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_v2_distnorm_unsigned_encdec.sh"
