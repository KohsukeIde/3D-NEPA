#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/eval/nepa3d_cqa_cls_qg.sh"
DEFAULT_CKPT="runs/cqa/patchnepa_cqa_udfdist_worldv3_curve_20260316/cqa_udfdist_worldv3_g2_s10000/ckpt_final.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
VARIANT="${VARIANT:-obj_only}"
case "${VARIANT}" in
  obj_bg) DEFAULT_CACHE_ROOT="data/scanobjectnn_obj_bg_v3_nonorm" ;;
  obj_only) DEFAULT_CACHE_ROOT="data/scanobjectnn_obj_only_v3_nonorm" ;;
  pb_t50_rs) DEFAULT_CACHE_ROOT="data/scanobjectnn_pb_t50_rs_v3_nonorm" ;;
  *)
    echo "[error] unsupported VARIANT=${VARIANT} (use obj_bg|obj_only|pb_t50_rs)"
    exit 2
    ;;
esac

CACHE_ROOT="${CACHE_ROOT:-${DEFAULT_CACHE_ROOT}}"
RUN_SET="${RUN_SET:-patchnepa_cqa_cls_${VARIANT}_pointmae_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_udfdist_${VARIANT}_pointmae_seed${SEED:-0}}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/cqa_cls/${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cqa_cls/${RUN_SET}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-cqa_cls}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-pointmae}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
N_POINT="${N_POINT:-2048}"
SAMPLE_MODE_TRAIN="${SAMPLE_MODE_TRAIN:-random}"
SAMPLE_MODE_EVAL="${SAMPLE_MODE_EVAL:-random}"
POOL="${POOL:-mean}"
POINTMAE_AUG="${POINTMAE_AUG:-1}"

mkdir -p "${PBS_LOG_DIR}" "${SAVE_DIR}" "${ENV_DIR}"

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
  "CACHE_ROOT=${CACHE_ROOT}"
  "TRAIN_SPLIT=${TRAIN_SPLIT}"
  "VAL_SPLIT=${VAL_SPLIT}"
  "VAL_SPLIT_MODE=${VAL_SPLIT_MODE}"
  "CKPT=${CKPT}"
  "SAVE_DIR=${SAVE_DIR}"
  "LOG_ROOT=${LOG_ROOT}"
  "SEED=${SEED}"
  "EPOCHS=${EPOCHS}"
  "BATCH=${BATCH}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "LR=${LR}"
  "WEIGHT_DECAY=${WEIGHT_DECAY}"
  "N_POINT=${N_POINT}"
  "SAMPLE_MODE_TRAIN=${SAMPLE_MODE_TRAIN}"
  "SAMPLE_MODE_EVAL=${SAMPLE_MODE_EVAL}"
  "POOL=${POOL}"
  "POINTMAE_AUG=${POINTMAE_AUG}"
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
echo "[variant] ${VARIANT}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[pbs_log] ${PBS_LOG_PATH}"
echo "[env_file] ${ENV_FILE}"
