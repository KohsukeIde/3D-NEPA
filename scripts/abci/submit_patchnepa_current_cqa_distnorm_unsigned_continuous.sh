#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_distnorm_unsigned_continuous_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_distnorm_unsigned_continuous_independent_g2_s10000}"

env RUN_SET="${RUN_SET}" TRAIN_RUN_TAG="${TRAIN_RUN_TAG}" \
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_continuous.yaml}" \
SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_continuous.yaml}" \
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_continuous_pcbank_eval.yaml}" \
TASKS="${TASKS:-udf_distance,mesh_normal_unsigned}" \
RUN_TAG="${RUN_TAG:-cqa_distnorm_unsigned_continuous_suite}" \
bash scripts/abci/submit_patchnepa_current_cqa_distnorm_continuous.sh
