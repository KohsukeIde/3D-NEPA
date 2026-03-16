#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/freeze_shapenet_world_v3_qc.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
WALLTIME="${WALLTIME:-24:00:00}"
RUN_TAG="${RUN_TAG:-world_v3_freeze_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/world_v3_freeze/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260311_worldvis}"
SPLITS="${SPLITS:-train:test}"
OUT_DIR="${OUT_DIR:-${WORKDIR}/results/data_freeze/${RUN_TAG}}"
NUM_WORKERS="${NUM_WORKERS:-12}"
REFRESH="${REFRESH:-0}"
UDF_SURF_MAX_T="${UDF_SURF_MAX_T:-2.0}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
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

ENV_FILE="${LOG_DIR}/${RUN_TAG}.env"
write_env_file "${ENV_FILE}" \
  "WORKDIR=${WORKDIR}" \
  "RUN_TAG=${RUN_TAG}" \
  "CACHE_ROOT=${CACHE_ROOT}" \
  "SPLITS=${SPLITS}" \
  "OUT_DIR=${OUT_DIR}" \
  "NUM_WORKERS=${NUM_WORKERS}" \
  "REFRESH=${REFRESH}" \
  "UDF_SURF_MAX_T=${UDF_SURF_MAX_T}"

cmd=(
  qsub
  -l "rt_QC=1"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "wv3_${RUN_TAG}"
  -o "${LOG_DIR}/${RUN_TAG}.out"
  -e "${LOG_DIR}/${RUN_TAG}.err"
  -v "WORKDIR=${WORKDIR},ENV_FILE=${ENV_FILE}"
  "${SCRIPT}"
)

jid="$("${cmd[@]}")"
echo "[submitted] ${jid}"
echo "[run_tag] ${RUN_TAG}"
echo "[log_dir] ${LOG_DIR}"
echo "[out_dir] ${OUT_DIR}"
echo "[env_file] ${ENV_FILE}"
