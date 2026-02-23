#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/preprocess_modelnet40.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-72:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
JOB_NAME="${JOB_NAME:-preprocess_modelnet40_v2}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

MODELNET_ROOT="${MODELNET_ROOT:-data/ModelNet40}"
OUT_ROOT="${OUT_ROOT:-data/modelnet40_cache_v2}"
SPLIT="${SPLIT:-all}"
MESH_GLOB="${MESH_GLOB:-}"
OVERWRITE="${OVERWRITE:-0}"

N_WORKERS="${N_WORKERS:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-8}"

PC_POINTS="${PC_POINTS:-1024}"
PT_POOL="${PT_POOL:-2000}"
RAY_POOL="${RAY_POOL:-1000}"
N_VIEWS="${N_VIEWS:-10}"
RAYS_PER_VIEW="${RAYS_PER_VIEW:-200}"
SEED="${SEED:-0}"
PC_GRID="${PC_GRID:-64}"
PC_DILATE="${PC_DILATE:-1}"
PC_MAX_STEPS="${PC_MAX_STEPS:-0}"
NO_PC_RAYS="${NO_PC_RAYS:-0}"
DF_GRID="${DF_GRID:-64}"
DF_DILATE="${DF_DILATE:-1}"
NO_UDF="${NO_UDF:-0}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
PT_QUERY_CHUNK="${PT_QUERY_CHUNK:-2048}"
RAY_QUERY_CHUNK="${RAY_QUERY_CHUNK:-2048}"
PT_DIST_MODE="${PT_DIST_MODE:-mesh}"
DIST_REF_POINTS="${DIST_REF_POINTS:-8192}"

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/modelnet40_cache_v2}"
mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "MODELNET_ROOT=${MODELNET_ROOT}"
  "OUT_ROOT=${OUT_ROOT}"
  "SPLIT=${SPLIT}"
  "OVERWRITE=${OVERWRITE}"
  "N_WORKERS=${N_WORKERS}"
  "CHUNK_SIZE=${CHUNK_SIZE}"
  "MAX_TASKS_PER_CHILD=${MAX_TASKS_PER_CHILD}"
  "PC_POINTS=${PC_POINTS}"
  "PT_POOL=${PT_POOL}"
  "RAY_POOL=${RAY_POOL}"
  "N_VIEWS=${N_VIEWS}"
  "RAYS_PER_VIEW=${RAYS_PER_VIEW}"
  "SEED=${SEED}"
  "PC_GRID=${PC_GRID}"
  "PC_DILATE=${PC_DILATE}"
  "PC_MAX_STEPS=${PC_MAX_STEPS}"
  "NO_PC_RAYS=${NO_PC_RAYS}"
  "DF_GRID=${DF_GRID}"
  "DF_DILATE=${DF_DILATE}"
  "NO_UDF=${NO_UDF}"
  "PT_SURFACE_RATIO=${PT_SURFACE_RATIO}"
  "PT_SURFACE_SIGMA=${PT_SURFACE_SIGMA}"
  "PT_QUERY_CHUNK=${PT_QUERY_CHUNK}"
  "RAY_QUERY_CHUNK=${RAY_QUERY_CHUNK}"
  "PT_DIST_MODE=${PT_DIST_MODE}"
  "DIST_REF_POINTS=${DIST_REF_POINTS}"
)
if [[ -n "${MESH_GLOB}" ]]; then
  qvars+=("MESH_GLOB=${MESH_GLOB}")
fi
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${LOG_DIR}/${JOB_NAME}.out"
  -e "${LOG_DIR}/${JOB_NAME}.err"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

echo "[submit] ${JOB_NAME}"
echo "[submit] modelnet_root=${MODELNET_ROOT} out_root=${OUT_ROOT} split=${SPLIT}"
echo "[submit] workers=${N_WORKERS} rt_QF=${RT_QF} walltime=${WALLTIME}"
"${cmd[@]}"
