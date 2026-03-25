#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

RUN_SET="${RUN_SET:-patchnepa_cqa_v3_thick128_single_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_v3_thick128_independent_g2_s10000}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"

export RUN_SET
export RUN_TAG
export SAVE_DIR
export LOG_ROOT
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v3_thick.yaml}"
export CODEC_VERSION="${CODEC_VERSION:-cqa_v3}"
export ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
export QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"
export QUERY_ORDER="${QUERY_ORDER:-shuffled}"
export EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
export MAX_STEPS="${MAX_STEPS:-10000}"
export EPOCHS="${EPOCHS:-20}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}"
export BATCH="${BATCH:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export SEED="${SEED:-0}"
export EVAL_TASK_FILTER="${EVAL_TASK_FILTER:-udf_thickness_valid_qbin}"
export EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}"
export EVAL_CONTROLS="${EVAL_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,cqa_v3,thick128,independent}"

train_out="$("${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh")"
printf '%s\n' "${train_out}"
train_job="$(printf '%s\n' "${train_out}" | awk '/^\[submitted\]/{print $2; exit}')"
if [[ -z "${train_job:-}" ]]; then
  echo "ERROR: failed to parse train job id" >&2
  exit 1
fi

SUITE_RUN_SET="${SUITE_RUN_SET:-${RUN_SET}_suite}"
SUITE_RUN_TAG="${SUITE_RUN_TAG:-cqa_v3_thick128_suite}"
SUITE_RESULTS_ROOT="${SUITE_RESULTS_ROOT:-${ROOT_DIR}/results/cqa_multitype/${SUITE_RUN_SET}}"
SUITE_LOG_ROOT="${SUITE_LOG_ROOT:-${ROOT_DIR}/logs/cqa_multitype/${SUITE_RUN_SET}}"
SUITE_OUT_JSON="${SUITE_OUT_JSON:-${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.json}"
SUITE_OUT_CSV="${SUITE_OUT_CSV:-${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.csv}"
SUITE_OUT_MD="${SUITE_OUT_MD:-${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.md}"

suite_out="$(
  QSUB_DEPEND="afterok:${train_job}" \
  CKPT="${SAVE_DIR}/ckpt_final.pt" \
  RUN_SET="${SUITE_RUN_SET}" \
  RUN_TAG="${SUITE_RUN_TAG}" \
  SAME_MIX_CONFIG="${MIX_CONFIG}" \
  OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v3_thick_pcbank_eval.yaml}" \
  TASKS="${TASKS:-udf_thickness_valid_qbin}" \
  CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}" \
  QUERY_ORDER="${EVAL_QUERY_ORDER}" \
  LOG_ROOT="${SUITE_LOG_ROOT}" \
  RESULTS_ROOT="${SUITE_RESULTS_ROOT}" \
  OUT_JSON="${SUITE_OUT_JSON}" \
  OUT_CSV="${SUITE_OUT_CSV}" \
  OUT_MD="${SUITE_OUT_MD}" \
  "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_multitype_suite.sh"
)"
printf '%s\n' "${suite_out}"
