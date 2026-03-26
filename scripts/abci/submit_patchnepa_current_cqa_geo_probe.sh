#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/eval/nepa3d_cqa_geo_probe_qg.sh"
DEFAULT_CACHE_ROOT="${ROOT_DIR}/data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1"
PYTHON_BIN_DEFAULT="${ROOT_DIR}/.venv/bin/python"

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${ROOT_DIR}" "$1" ;;
  esac
}

CKPT="${CKPT:?set CKPT=...}"
PROBE_TARGET="${PROBE_TARGET:?set PROBE_TARGET=curvature|signed_normal}"
CACHE_ROOT="${CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_probe_${PROBE_TARGET}_${STAMP}}"
RUN_TAG="${RUN_TAG:-cqa_probe_${PROBE_TARGET}_seed${SEED:-0}}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/cqa_probe/${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/cqa_probe/${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cqa_probe/${RUN_SET}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"
MANIFEST_JSON="${MANIFEST_JSON:-}"

CKPT="$(abspath "${CKPT}")"
CACHE_ROOT="$(abspath "${CACHE_ROOT}")"
SAVE_DIR="$(abspath "${SAVE_DIR}")"
RESULTS_ROOT="$(abspath "${RESULTS_ROOT}")"
LOG_ROOT="$(abspath "${LOG_ROOT}")"
PBS_LOG_DIR="$(abspath "${PBS_LOG_DIR}")"
PBS_LOG_PATH="$(abspath "${PBS_LOG_PATH}")"
ENV_DIR="$(abspath "${ENV_DIR}")"
if [[ -n "${MANIFEST_JSON}" ]]; then
  MANIFEST_JSON="$(abspath "${MANIFEST_JSON}")"
fi

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-cqa_probe}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_mesh}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
TRAIN_QUERY_ORDER="${TRAIN_QUERY_ORDER:-shuffled}"
EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"

if [[ "${PROBE_TARGET}" == "signed_normal" && -z "${MANIFEST_JSON}" ]]; then
  MANIFEST_JSON="${RESULTS_ROOT}/winding_consistent_subset.json"
fi
if [[ -n "${MANIFEST_JSON}" ]]; then
  MANIFEST_JSON="$(abspath "${MANIFEST_JSON}")"
fi
if [[ "${PROBE_TARGET}" == "signed_normal" && ! -f "${MANIFEST_JSON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
  fi
  mkdir -p "${RESULTS_ROOT}"
  "${PYTHON_BIN}" -m nepa3d.data.build_shapenet_subset_manifest \
    --cache_root "${CACHE_ROOT}" \
    --splits "${TRAIN_SPLIT}:${EVAL_SPLIT}" \
    --out_json "${MANIFEST_JSON}" \
    --require_watertight 0 \
    --require_winding_consistent 1 >/dev/null
fi

mkdir -p "${PBS_LOG_DIR}" "${SAVE_DIR}" "${RESULTS_ROOT}" "${ENV_DIR}"

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
  "CACHE_ROOT=${CACHE_ROOT}"
  "PROBE_TARGET=${PROBE_TARGET}"
  "MANIFEST_JSON=${MANIFEST_JSON}"
  "SAVE_DIR=${SAVE_DIR}"
  "OUT_JSON=${RESULTS_ROOT}/${RUN_TAG}.json"
  "LOG_ROOT=${LOG_ROOT}"
  "TRAIN_SPLIT=${TRAIN_SPLIT}"
  "EVAL_SPLIT=${EVAL_SPLIT}"
  "SEED=${SEED}"
  "MAX_STEPS=${MAX_STEPS}"
  "EVAL_EVERY=${EVAL_EVERY}"
  "BATCH=${BATCH}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "LR=${LR}"
  "WEIGHT_DECAY=${WEIGHT_DECAY}"
  "N_CTX=${N_CTX}"
  "N_QRY=${N_QRY}"
  "TRAIN_QUERY_ORDER=${TRAIN_QUERY_ORDER}"
  "EVAL_QUERY_ORDER=${EVAL_QUERY_ORDER}"
  "MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
  "MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES}"
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
echo "[probe_target] ${PROBE_TARGET}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[pbs_log] ${PBS_LOG_PATH}"
echo "[env_file] ${ENV_FILE}"
echo "[out_json] ${RESULTS_ROOT}/${RUN_TAG}.json"
if [[ -n "${MANIFEST_JSON}" ]]; then
  echo "[manifest_json] ${MANIFEST_JSON}"
fi
