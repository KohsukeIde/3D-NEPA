#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/analysis/nepa3d_cqa_type_switch_qg.sh"
DEFAULT_CKPT="runs/cqa/patchnepa_cqa_distnorm_shared_20260323/cqa_distnorm_independent_g2_s10000/ckpt_final.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
RUN_SET="${RUN_SET:-patchnepa_cqa_type_switch_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_distnorm_type_switch_pc}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cqa_multitype/${RUN_SET}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/cqa_multitype/${RUN_SET}/${RUN_TAG}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-08:00:00}"
JOB_NAME="${JOB_NAME:-cqa_tsw}"

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-eval}"
SEED="${SEED:-0}"
MAX_SHAPES="${MAX_SHAPES:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-random}"
CONTEXT_SOURCE="${CONTEXT_SOURCE:-pc_bank}"
N_CTX="${N_CTX:-2048}"
N_QRY_SURFACE="${N_QRY_SURFACE:-1000000}"
TASKS="${TASKS:-udf_distance,mesh_normal}"
DISTANCE_GRID_RES="${DISTANCE_GRID_RES:-16}"
DISTANCE_CHUNK_N_QUERY="${DISTANCE_CHUNK_N_QUERY:-64}"
DISTANCE_MC_LEVEL="${DISTANCE_MC_LEVEL:-0.02}"

mkdir -p "${PBS_LOG_DIR}" "${OUTPUT_ROOT}" "${ENV_DIR}"

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
  "LOG_ROOT=${LOG_ROOT}"
  "OUTPUT_ROOT=${OUTPUT_ROOT}"
  "DEVICE=${DEVICE}"
  "SPLIT=${SPLIT}"
  "SEED=${SEED}"
  "MAX_SHAPES=${MAX_SHAPES}"
  "SAMPLE_MODE=${SAMPLE_MODE}"
  "CONTEXT_SOURCE=${CONTEXT_SOURCE}"
  "N_CTX=${N_CTX}"
  "N_QRY_SURFACE=${N_QRY_SURFACE}"
  "TASKS=${TASKS}"
  "DISTANCE_GRID_RES=${DISTANCE_GRID_RES}"
  "DISTANCE_CHUNK_N_QUERY=${DISTANCE_CHUNK_N_QUERY}"
  "DISTANCE_MC_LEVEL=${DISTANCE_MC_LEVEL}"
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
echo "[output_root] ${OUTPUT_ROOT}"
