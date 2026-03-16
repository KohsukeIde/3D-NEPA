#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

export RUN_SET="${RUN_SET:-patchnepa_cqa_udfdist_curve_$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG="${RUN_TAG:-cqa_udfdist_worldv3_g2_s10000}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export EPOCHS="${EPOCHS:-20}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export MIN_LR="${MIN_LR:-1e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-curve}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,curve,udf_distance,world_v3}"
export RUN_EVAL_CONTROLS="${RUN_EVAL_CONTROLS:-1}"
export RUN_EVAL_CURVE="${RUN_EVAL_CURVE:-1}"
export EVAL_BATCH="${EVAL_BATCH:-128}"
export EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
export EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}"
export EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
export EVAL_TASK_FILTER="${EVAL_TASK_FILTER:-udf_distance}"
export EVAL_CONTROLS="${EVAL_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh"
