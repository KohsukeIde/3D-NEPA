#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

export RUN_SET="${RUN_SET:-patchnepa_cqa_udfdist_curve_noq_$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG="${RUN_TAG:-cqa_udfdist_worldv3_independent_noq_g2_s10000}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
export ANSWER_FACTORIZATION="independent"
export QUERY_INTERFACE_MODE="no_q"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-curve}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,curve,udf_distance,world_v3,independent,no_q}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_curve.sh"
