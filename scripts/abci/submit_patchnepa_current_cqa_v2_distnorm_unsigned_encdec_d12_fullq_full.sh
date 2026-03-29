#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
export RUN_SET="${RUN_SET:-patchnepa_cqa_v2_distnorm_unsigned_encdec_d12_fullq_full_${STAMP}}"
export RUN_SLUG="${RUN_SLUG:-cqa_v2_distnorm_unsigned_encdec_d12_fullq_full_e100}"
export TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_v2_distnorm_unsigned_encdec_independent_d12_fullq_e100}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned.yaml}"
export SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned.yaml}"
export OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned_pcbank_eval.yaml}"
export TASKS="${TASKS:-udf_distance,mesh_normal_unsigned}"
export MODEL_ARCH="${MODEL_ARCH:-encdec}"
export DECODER_LAYERS="${DECODER_LAYERS:-12}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,cqa_v2,distnorm_unsigned,encdec,d12,full_q,independent,full100}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_full_chain.sh"
