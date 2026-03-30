#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

export RUN_SET="${RUN_SET:-patchnepa_cqa_v2_distnorm_unsigned_trainmesh_packedmultihead30ep_${STAMP}}"
export RUN_SLUG="${RUN_SLUG:-cqa_v2_distnorm_unsigned_trainmesh_packedmultihead30ep}"
export TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_v2_distnorm_unsigned_trainmesh_packedmultihead30ep_independent_g2_e30}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned_trainmesh_common.yaml}"
export TASKS="${TASKS:-udf_distance,mesh_normal_unsigned}"
export SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned.yaml}"
export OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned_pcbank_eval.yaml}"
export CODEC_VERSION="${CODEC_VERSION:-cqa_v2}"
export MODEL_ARCH="${MODEL_ARCH:-prefixlm}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"
export HEAD_MODE="${HEAD_MODE:-multihead}"
export SAMPLING_PROTOCOL="${SAMPLING_PROTOCOL:-packed}"
export QUERY_ORDER="${QUERY_ORDER:-shuffled}"
export EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
export MAX_STEPS="${MAX_STEPS:-0}"
export EPOCHS="${EPOCHS:-30}"
export BATCH="${BATCH:-4}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export WALLTIME="${WALLTIME:-02:55:00}"
export SUITE_WALLTIME="${SUITE_WALLTIME:-02:55:00}"
export COMP_WALLTIME="${COMP_WALLTIME:-02:55:00}"
export CLS_WALLTIME="${CLS_WALLTIME:-02:55:00}"
export COMP_EXPORT_ASSETS="${COMP_EXPORT_ASSETS:-0}"
export ENABLE_GEO_PROBES="${ENABLE_GEO_PROBES:-0}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,cqa_v2,distnorm_unsigned,trainmesh_common,packed,multihead,e30}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_full_chain.sh"
