#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:?set RUN_SET=...}"
RUN_SLUG="${RUN_SLUG:?set RUN_SLUG=...}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:?set TRAIN_RUN_TAG=...}"
MIX_CONFIG="${MIX_CONFIG:?set MIX_CONFIG=...}"
TASKS="${TASKS:?set TASKS=task_a,task_b,...}"

CODEC_VERSION="${CODEC_VERSION:-cqa_v2}"
MODEL_ARCH="${MODEL_ARCH:-prefixlm}"
DECODER_LAYERS="${DECODER_LAYERS:-4}"
ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION:-independent}"
QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE:-full_q}"
HEAD_MODE="${HEAD_MODE:-shared}"
QUERY_ORDER="${QUERY_ORDER:-shuffled}"
EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"

SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-${MIX_CONFIG}}"
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-${MIX_CONFIG}}"
SAME_COMPLETION_MIX_CONFIG="${SAME_COMPLETION_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
OFFDIAG_COMPLETION_MIX_CONFIG="${OFFDIAG_COMPLETION_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"

SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}/${TRAIN_RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"
WANDB_TAGS="${WANDB_TAGS:-abci,cqa,current,full100}"

MAX_STEPS="${MAX_STEPS:-0}"
EPOCHS="${EPOCHS:-100}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-0}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
WALLTIME="${WALLTIME:-48:00:00}"
RT_QG="${RT_QG:-1}"
SUITE_WALLTIME="${SUITE_WALLTIME:-${WALLTIME}}"
COMP_WALLTIME="${COMP_WALLTIME:-${WALLTIME}}"
CLS_WALLTIME="${CLS_WALLTIME:-${WALLTIME}}"
PROBE_WALLTIME="${PROBE_WALLTIME:-${WALLTIME}}"

EVAL_TASK_FILTER="${EVAL_TASK_FILTER:-${TASKS}}"
EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}"
EVAL_CONTROLS="${EVAL_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"
COMP_MAX_SHAPES="${COMP_MAX_SHAPES:-64}"
COMP_GRID_RES="${COMP_GRID_RES:-16}"
COMP_CHUNK_N_QUERY="${COMP_CHUNK_N_QUERY:-64}"
COMP_EXPORT_ASSETS="${COMP_EXPORT_ASSETS:-1}"
COMP_MESH_EVAL="${COMP_MESH_EVAL:-1}"
CLS_POOL="${CLS_POOL:-mean}"
CLS_EPOCHS="${CLS_EPOCHS:-300}"
ENABLE_GEO_PROBES="${ENABLE_GEO_PROBES:-1}"
GEO_PROBE_TARGETS="${GEO_PROBE_TARGETS:-curvature,signed_normal}"
SIGNED_MANIFEST_JSON_DEFAULT="${ROOT_DIR}/results/cqa_probe/patchnepa_cqa_probe_signed_subset_20260326/winding_consistent_subset.json"
SIGNED_MANIFEST_JSON="${SIGNED_MANIFEST_JSON:-${SIGNED_MANIFEST_JSON_DEFAULT}}"

train_out="$(
  env \
    RUN_SET="${RUN_SET}" \
    RUN_TAG="${TRAIN_RUN_TAG}" \
    SAVE_DIR="${SAVE_DIR}" \
    LOG_ROOT="${LOG_ROOT}" \
    MIX_CONFIG="${MIX_CONFIG}" \
    CODEC_VERSION="${CODEC_VERSION}" \
    MODEL_ARCH="${MODEL_ARCH}" \
    DECODER_LAYERS="${DECODER_LAYERS}" \
    ANSWER_FACTORIZATION="${ANSWER_FACTORIZATION}" \
    QUERY_INTERFACE_MODE="${QUERY_INTERFACE_MODE}" \
    HEAD_MODE="${HEAD_MODE}" \
    QUERY_ORDER="${QUERY_ORDER}" \
    EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER}" \
    EVAL_TASK_FILTER="${EVAL_TASK_FILTER}" \
    EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK}" \
    EVAL_CONTROLS="${EVAL_CONTROLS}" \
    MAX_STEPS="${MAX_STEPS}" \
    EPOCHS="${EPOCHS}" \
    SAVE_EVERY="${SAVE_EVERY}" \
    SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS}" \
    BATCH="${BATCH}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SEED="${SEED}" \
    WALLTIME="${WALLTIME}" \
    RT_QG="${RT_QG}" \
    WANDB_TAGS="${WANDB_TAGS}" \
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
SUITE_RUN_TAG="${SUITE_RUN_TAG:-${RUN_SLUG}_suite}"
SUITE_RESULTS_ROOT="${SUITE_RESULTS_ROOT:-${ROOT_DIR}/results/cqa_multitype/${SUITE_RUN_SET}}"
SUITE_LOG_ROOT="${SUITE_LOG_ROOT:-${ROOT_DIR}/logs/cqa_multitype/${SUITE_RUN_SET}}"
suite_out="$(
  QSUB_DEPEND="afterok:${train_job}" \
  CKPT="${CKPT_PATH}" \
  RUN_SET="${SUITE_RUN_SET}" \
  RUN_TAG="${SUITE_RUN_TAG}" \
  SAME_MIX_CONFIG="${SAME_MIX_CONFIG}" \
  OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG}" \
  TASKS="${TASKS}" \
  CONTROLS="${EVAL_CONTROLS}" \
  QUERY_ORDER="${EVAL_QUERY_ORDER}" \
  LOG_ROOT="${SUITE_LOG_ROOT}" \
  RESULTS_ROOT="${SUITE_RESULTS_ROOT}" \
  OUT_JSON="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.json" \
  OUT_CSV="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.csv" \
  OUT_MD="${SUITE_RESULTS_ROOT}/${SUITE_RUN_TAG}.md" \
  WALLTIME="${SUITE_WALLTIME}" \
  "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_multitype_suite.sh"
)"
printf '%s\n' "${suite_out}"

for mode in same offdiag; do
  case "${mode}" in
    same) mix_cfg="${SAME_COMPLETION_MIX_CONFIG}" ;;
    offdiag) mix_cfg="${OFFDIAG_COMPLETION_MIX_CONFIG}" ;;
  esac
  comp_run_set="${RUN_SET}_${mode}_completion"
  comp_run_tag="${RUN_SLUG}_${mode}_translation_g16_s64"
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
    GRID_RES="${COMP_GRID_RES}" \
    MAX_SHAPES="${COMP_MAX_SHAPES}" \
    CHUNK_N_QUERY="${COMP_CHUNK_N_QUERY}" \
    MESH_EVAL="${COMP_MESH_EVAL}" \
    EXPORT_ASSETS="${COMP_EXPORT_ASSETS}" \
    WALLTIME="${COMP_WALLTIME}" \
    "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh"
  )"
  printf '%s\n' "${comp_out}"
done

for variant in obj_bg obj_only pb_t50_rs; do
  cls_run_set="${RUN_SET}_${variant}_cls"
  cls_run_tag="${RUN_SLUG}_${variant}_seed${SEED}"
  cls_out="$(
    QSUB_DEPEND="afterok:${train_job}" \
    CKPT="${CKPT_PATH}" \
    VARIANT="${variant}" \
    RUN_SET="${cls_run_set}" \
    RUN_TAG="${cls_run_tag}" \
    SAVE_DIR="${ROOT_DIR}/runs/cqa_cls/${cls_run_set}" \
    LOG_ROOT="${ROOT_DIR}/logs/cqa_cls/${cls_run_set}" \
    POOL="${CLS_POOL}" \
    EPOCHS="${CLS_EPOCHS}" \
    WALLTIME="${CLS_WALLTIME}" \
    "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_cls.sh"
  )"
  printf '%s\n' "${cls_out}"
done

if [[ "${ENABLE_GEO_PROBES}" == "1" ]]; then
  IFS=',' read -r -a probe_targets <<< "${GEO_PROBE_TARGETS}"
  for probe_target in "${probe_targets[@]}"; do
    probe_run_set="${RUN_SET}_${probe_target}_probe"
    probe_run_tag="${RUN_SLUG}_${probe_target}_seed${SEED}"
    probe_manifest_json=""
    if [[ "${probe_target}" == "signed_normal" ]]; then
      probe_manifest_json="${SIGNED_MANIFEST_JSON}"
    fi
    probe_out="$(
      QSUB_DEPEND="afterok:${train_job}" \
      CKPT="${CKPT_PATH}" \
      PROBE_TARGET="${probe_target}" \
      MANIFEST_JSON="${probe_manifest_json}" \
      RUN_SET="${probe_run_set}" \
      RUN_TAG="${probe_run_tag}" \
      SAVE_DIR="${ROOT_DIR}/runs/cqa_probe/${probe_run_set}" \
      RESULTS_ROOT="${ROOT_DIR}/results/cqa_probe/${probe_run_set}" \
      LOG_ROOT="${ROOT_DIR}/logs/cqa_probe/${probe_run_set}" \
      WALLTIME="${PROBE_WALLTIME}" \
      "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_geo_probe.sh"
    )"
    printf '%s\n' "${probe_out}"
  done
fi
