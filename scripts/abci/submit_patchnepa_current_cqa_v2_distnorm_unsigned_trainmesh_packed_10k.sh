#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_v2_distnorm_unsigned_trainmesh_packed10k_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_v2_distnorm_unsigned_trainmesh_packed10k_independent_g2_s10000}"
WALLTIME="${WALLTIME:-10:00:00}"

train_out="$(
  env \
    RUN_SET="${RUN_SET}" \
    RUN_TAG="${TRAIN_RUN_TAG}" \
    WALLTIME="${WALLTIME}" \
    SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${TRAIN_RUN_TAG}}" \
    LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}" \
    MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned_trainmesh_common.yaml}" \
    CODEC_VERSION="${CODEC_VERSION:-cqa_v2}" \
    MODEL_ARCH="${MODEL_ARCH:-prefixlm}" \
    ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}" \
    QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}" \
    SAMPLING_PROTOCOL="${SAMPLING_PROTOCOL:-packed}" \
    QUERY_ORDER="${QUERY_ORDER:-shuffled}" \
    EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}" \
    EVAL_TASK_FILTER="${EVAL_TASK_FILTER:-udf_distance,mesh_normal_unsigned}" \
    EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}" \
    EVAL_CONTROLS="${EVAL_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}" \
    MAX_STEPS="${MAX_STEPS:-10000}" \
    EPOCHS="${EPOCHS:-20}" \
    SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}" \
    BATCH="${BATCH:-4}" \
    NUM_WORKERS="${NUM_WORKERS:-8}" \
    SEED="${SEED:-0}" \
    RUN_EVAL_CONTROLS="${RUN_EVAL_CONTROLS:-0}" \
    WANDB_TAGS="${WANDB_TAGS:-abci,cqa,cqa_v2,distnorm_unsigned,trainmesh_common,packed,protocol_isolation}" \
    "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh"
)"
printf '%s\n' "${train_out}"
train_job="$(printf '%s\n' "${train_out}" | awk '/^\[submitted\]/{print $2; exit}')"
train_save_dir="$(printf '%s\n' "${train_out}" | awk '/^\[save_dir\]/{print $2; exit}')"
if [[ -z "${train_job:-}" ]] || [[ -z "${train_save_dir:-}" ]]; then
  echo "ERROR: failed to parse train submission output" >&2
  exit 1
fi
CKPT_PATH="${CKPT_PATH:-${train_save_dir}/ckpt_final.pt}"

SUITE_RUN_SET="${SUITE_RUN_SET:-${RUN_SET}_suite}"
SUITE_RUN_TAG="${SUITE_RUN_TAG:-cqa_v2_distnorm_unsigned_trainmesh_packed10k_suite}"
SUITE_RESULTS_ROOT="${SUITE_RESULTS_ROOT:-${ROOT_DIR}/results/cqa_multitype/${SUITE_RUN_SET}}"
SUITE_LOG_ROOT="${SUITE_LOG_ROOT:-${ROOT_DIR}/logs/cqa_multitype/${SUITE_RUN_SET}}"
suite_out="$(
  QSUB_DEPEND="afterok:${train_job}" \
  CKPT="${CKPT_PATH}" \
  RUN_SET="${SUITE_RUN_SET}" \
  RUN_TAG="${SUITE_RUN_TAG}" \
  SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned.yaml}" \
  OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_v2_dist_norm_unsigned_pcbank_eval.yaml}" \
  TASKS="${TASKS:-udf_distance,mesh_normal_unsigned}" \
  CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}" \
  QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}" \
  LOG_ROOT="${SUITE_LOG_ROOT}" \
  RESULTS_ROOT="${SUITE_RESULTS_ROOT}" \
  OUT_JSON="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.json" \
  OUT_CSV="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.csv" \
  OUT_MD="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.md" \
  "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_multitype_suite.sh"
)"
printf '%s\n' "${suite_out}"

for mode in same offdiag; do
  case "${mode}" in
    same) mix_cfg="${SAME_COMPLETION_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}" ;;
    offdiag) mix_cfg="${OFFDIAG_COMPLETION_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}" ;;
  esac
  comp_run_set="${RUN_SET}_${mode}_completion"
  comp_run_tag="cqa_v2_distnorm_unsigned_trainmesh_packed10k_${mode}_translation_g16_s64"
  comp_out="$(
    QSUB_DEPEND="afterok:${train_job}" \
    CKPT="${CKPT_PATH}" \
    MODE="${mode}" \
    MIX_CONFIG="${mix_cfg}" \
    RUN_SET="${comp_run_set}" \
    RUN_TAG="${comp_run_tag}" \
    LOG_ROOT="${ROOT_DIR}/logs/cqa_completion/${comp_run_set}" \
    RESULTS_ROOT="${ROOT_DIR}/results/cqa_completion/${comp_run_set}" \
    OUT_JSON="${ROOT_DIR}/results/cqa_completion/${comp_run_set}/${comp_run_tag}.json" \
    ASSETS_ROOT="${ROOT_DIR}/results/cqa_completion/${comp_run_set}/${comp_run_tag}_assets" \
    GRID_RES="${GRID_RES:-16}" \
    MAX_SHAPES="${MAX_SHAPES:-64}" \
    CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}" \
    MESH_EVAL="${MESH_EVAL:-1}" \
    EXPORT_ASSETS="${EXPORT_ASSETS:-0}" \
    "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh"
  )"
  printf '%s\n' "${comp_out}"
done
