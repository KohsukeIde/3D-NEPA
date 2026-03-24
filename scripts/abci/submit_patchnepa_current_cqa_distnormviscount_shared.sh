#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

RUN_SET="${RUN_SET:-patchnepa_cqa_distnormviscount_shared_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_distnormviscount_independent_g2_s10000}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"

export RUN_SET
export RUN_TAG
export SAVE_DIR
export LOG_ROOT
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_unsigned_viscount.yaml}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export EPOCHS="${EPOCHS:-20}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
export BATCH="${BATCH:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export QUERY_ORDER="${QUERY_ORDER:-shuffled}"
export EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
export EVAL_TASK_FILTER="${EVAL_TASK_FILTER:-udf_distance,mesh_normal_unsigned,mesh_viscount}"
export EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}"
export EVAL_CONTROLS="${EVAL_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,distnormviscount,independent}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh"
