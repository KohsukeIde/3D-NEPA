#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/augment_world_v3_mesh_aux_qc.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
WALLTIME="${WALLTIME:-12:00:00}"
RUN_TAG="${RUN_TAG:-world_v3_mesh_aux_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/world_v3_mesh_aux/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
DST_CACHE_ROOT="${DST_CACHE_ROOT:?set DST_CACHE_ROOT}"
SPLIT="${SPLIT:-eval}"
LIMIT="${LIMIT:-64}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
REFRESH="${REFRESH:-0}"
COPY_META="${COPY_META:-1}"
COMPUTE_AO_HQ="${COMPUTE_AO_HQ:-1}"
AO_RAYS="${AO_RAYS:-128}"
AO_EPS="${AO_EPS:-1e-4}"
AO_MAX_T="${AO_MAX_T:-2.5}"
AO_BATCH_SIZE="${AO_BATCH_SIZE:-64}"
COMPUTE_HKS="${COMPUTE_HKS:-0}"
HKS_EIGS="${HKS_EIGS:-64}"
HKS_TIMES="${HKS_TIMES:-0.05,0.2,1.0}"
SUFFIX="${SUFFIX:-}"
OUT_DIR="${OUT_DIR:-${WORKDIR}/results/data_freeze/${RUN_TAG}}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUT_DIR}/mesh_aux_summary.json}"
mkdir -p "${OUT_DIR}"

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
  "SRC_CACHE_ROOT=${SRC_CACHE_ROOT}" \
  "DST_CACHE_ROOT=${DST_CACHE_ROOT}" \
  "SPLIT=${SPLIT}" \
  "LIMIT=${LIMIT}" \
  "SAMPLE_SEED=${SAMPLE_SEED}" \
  "NUM_SHARDS=${NUM_SHARDS}" \
  "SHARD_INDEX=${SHARD_INDEX}" \
  "REFRESH=${REFRESH}" \
  "COPY_META=${COPY_META}" \
  "COMPUTE_AO_HQ=${COMPUTE_AO_HQ}" \
  "AO_RAYS=${AO_RAYS}" \
  "AO_EPS=${AO_EPS}" \
  "AO_MAX_T=${AO_MAX_T}" \
  "AO_BATCH_SIZE=${AO_BATCH_SIZE}" \
  "COMPUTE_HKS=${COMPUTE_HKS}" \
  "HKS_EIGS=${HKS_EIGS}" \
  "HKS_TIMES=${HKS_TIMES}" \
  "SUFFIX=${SUFFIX}" \
  "OUTPUT_JSON=${OUTPUT_JSON}"

cmd=(
  qsub
  -l "rt_QC=1"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${RUN_TAG}"
  -o "${LOG_DIR}/${RUN_TAG}.out"
  -e "${LOG_DIR}/${RUN_TAG}.err"
  -v "WORKDIR=${WORKDIR},ENV_FILE=${ENV_FILE}"
  "${SCRIPT}"
)

jid="$("${cmd[@]}")"
echo "[submitted] ${jid}"
echo "[run_tag] ${RUN_TAG}"
echo "[log_dir] ${LOG_DIR}"
echo "[env_file] ${ENV_FILE}"
echo "[dst_cache_root] ${DST_CACHE_ROOT}"
echo "[output_json] ${OUTPUT_JSON}"
