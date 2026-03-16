#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/analysis/nepa3d_cqa_offdiag_udfdist_qg.sh"
DEFAULT_CKPT="runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/ckpt_final.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
RUN_SET="${RUN_SET:-patchnepa_cqa_udfdist_offdiag_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_udfdist_offdiag_eval}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cqa_eval/${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/cqa_eval/${RUN_SET}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-06:00:00}"
JOB_NAME="${JOB_NAME:-cqa_odg}"

DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
TASK_FILTER="${TASK_FILTER:-udf_distance}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"
OUT_JSON="${OUT_JSON:-${RESULTS_ROOT}/${RUN_TAG}.json}"

mkdir -p "${PBS_LOG_DIR}" "${RESULTS_ROOT}" "${ENV_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found in PATH"
  exit 1
fi

write_env_file() {
  local path="$1"
  shift
  : > "${path}"
  for kv in "$@"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    printf '%s=%q\n' "${key}" "${val}" >> "${path}"
  done
}

qvars=(
  "WORKDIR=${ROOT_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "CKPT=${CKPT}"
  "MIX_CONFIG=${MIX_CONFIG}"
  "LOG_ROOT=${LOG_ROOT}"
  "OUT_JSON=${OUT_JSON}"
  "DEVICE=${DEVICE}"
  "BATCH=${BATCH}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "SEED=${SEED}"
  "N_CTX=${N_CTX}"
  "N_QRY=${N_QRY}"
  "MAX_SAMPLES=${MAX_SAMPLES}"
  "SPLIT_OVERRIDE=${SPLIT_OVERRIDE}"
  "TASK_FILTER=${TASK_FILTER}"
  "EVAL_SAMPLE_MODE=${EVAL_SAMPLE_MODE}"
  "CONTROLS=${CONTROLS}"
)
ENV_FILE="${ENV_FILE:-${ENV_DIR}/${RUN_TAG}.env}"
write_env_file "${ENV_FILE}" "${qvars[@]}"

cmd=(
  qsub
  -l "rt_QG=${RT_QG}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${PBS_LOG_PATH}"
  -v "WORKDIR=${ROOT_DIR},ENV_FILE=${ENV_FILE}"
)
if [[ -n "${QSUB_DEPEND:-}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[pbs_log] ${PBS_LOG_PATH}"
echo "[env_file] ${ENV_FILE}"
echo "[out_json] ${OUT_JSON}"
