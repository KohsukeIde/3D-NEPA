#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${WORKDIR}"

PREPROC_SUBMIT="${SCRIPT_DIR}/submit_preprocess_shapenet_v2_qf.sh"
POST_SUBMIT="${SCRIPT_DIR}/submit_shapenet_unpaired_post_qf.sh"

RUN_TAG="${RUN_TAG:-shapenet_worldvis_$(date +%Y%m%d_%H%M%S)}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
SHAPENET_ROOT="${SHAPENET_ROOT:-data/ShapeNetCore.v2}"

SOURCE_CACHE="${SOURCE_CACHE:-data/shapenet_cache_v2_20260311_worldvis}"
SPLIT_JSON_BASE="${SPLIT_JSON_BASE:-data/shapenet_unpaired_splits_v2_20260311_worldvis.json}"
SPLIT_JSON_PC33="${SPLIT_JSON_PC33:-data/shapenet_unpaired_splits_v2_pc33_mesh33_udf33_worldvis.json}"
SPLIT_JSON_M50U50="${SPLIT_JSON_M50U50:-data/shapenet_unpaired_splits_v2_mesh50_udf50_worldvis.json}"
UNPAIRED_ROOT="${UNPAIRED_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis}"
UNPAIRED_DROP1_ROOT="${UNPAIRED_DROP1_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
UNPAIRED_PC33_ROOT="${UNPAIRED_PC33_ROOT:-data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33_worldvis}"
UNPAIRED_M50U50_ROOT="${UNPAIRED_M50U50_ROOT:-data/shapenet_unpaired_cache_v2_mesh50_udf50_worldvis}"

NUM_SHARDS="${NUM_SHARDS:-16}"
WORKERS="${WORKERS:-32}"
WALLTIME_PREPROC="${WALLTIME_PREPROC:-72:00:00}"
RT_QF_PREPROC="${RT_QF_PREPROC:-1}"
PC_CTX_BANK="${PC_CTX_BANK:-4}"
UDF_PROBE_DELTAS="${UDF_PROBE_DELTAS:-0.01,0.02,0.05}"
MESH_VIS_N_DIRS="${MESH_VIS_N_DIRS:-8}"
MESH_VIS_MAX_T="${MESH_VIS_MAX_T:-2.5}"
MESH_VIS_EPS="${MESH_VIS_EPS:-1e-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MISSING_ONLY="${MISSING_ONLY:-0}"
AUGMENT_EXISTING="${AUGMENT_EXISTING:-0}"

PREPROC_LOG_DIR="${WORKDIR}/logs/preprocess/shapenet_v2/${RUN_TAG}"
mkdir -p "${PREPROC_LOG_DIR}"

env \
  WORKDIR="${WORKDIR}" \
  GROUP_LIST="${GROUP_LIST}" \
  RUN_TAG="${RUN_TAG}" \
  LOG_DIR="${PREPROC_LOG_DIR}" \
  SHAPENET_ROOT="${SHAPENET_ROOT}" \
  OUT_ROOT="${SOURCE_CACHE}" \
  NUM_SHARDS="${NUM_SHARDS}" \
  WORKERS="${WORKERS}" \
  WALLTIME="${WALLTIME_PREPROC}" \
  RT_QF="${RT_QF_PREPROC}" \
  PC_CTX_BANK="${PC_CTX_BANK}" \
  UDF_PROBE_DELTAS="${UDF_PROBE_DELTAS}" \
  MESH_VIS_N_DIRS="${MESH_VIS_N_DIRS}" \
  MESH_VIS_MAX_T="${MESH_VIS_MAX_T}" \
  MESH_VIS_EPS="${MESH_VIS_EPS}" \
  SKIP_EXISTING="${SKIP_EXISTING}" \
  MISSING_ONLY="${MISSING_ONLY}" \
  AUGMENT_EXISTING="${AUGMENT_EXISTING}" \
  bash "${PREPROC_SUBMIT}"

SUBMIT_LOG="${PREPROC_LOG_DIR}/submit.log"
if [[ ! -f "${SUBMIT_LOG}" ]]; then
  echo "[error] missing preprocess submit log: ${SUBMIT_LOG}"
  exit 1
fi
PREPROC_JOB_IDS="$(sed -n 's/.*job=\([^ ]*\).*/\1/p' "${SUBMIT_LOG}" | paste -sd, -)"
if [[ -z "${PREPROC_JOB_IDS}" ]]; then
  echo "[error] failed to extract preprocess job ids from ${SUBMIT_LOG}"
  exit 1
fi

submit_post() {
  local suffix="$1"
  shift
  env \
    WORKDIR="${WORKDIR}" \
    GROUP_LIST="${GROUP_LIST}" \
    PREPROC_JOB_IDS="${PREPROC_JOB_IDS}" \
    RUN_TAG="${RUN_TAG}_${suffix}" \
    LOG_DIR="${WORKDIR}/logs/preprocess/shapenet_unpaired/${RUN_TAG}_${suffix}" \
    "$@" \
    bash "${POST_SUBMIT}"
}

submit_post "base" \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_BASE}" \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_ROOT="${UNPAIRED_ROOT}" \
  RATIOS="0.34:0.33:0.33" \
  ALLOW_EMPTY_SPLITS="0" \
  OVERWRITE="1"

submit_post "drop1" \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_BASE}" \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_ROOT="${UNPAIRED_DROP1_ROOT}" \
  RATIOS="0.34:0.33:0.33" \
  ALLOW_EMPTY_SPLITS="0" \
  OVERWRITE="1"

submit_post "pc33" \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_PC33}" \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_ROOT="${UNPAIRED_PC33_ROOT}" \
  RATIOS="0.33:0.33:0.34" \
  ALLOW_EMPTY_SPLITS="0" \
  OVERWRITE="1"

submit_post "mesh50udf50" \
  CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_JSON="${SPLIT_JSON_M50U50}" \
  SRC_CACHE_ROOT="${SOURCE_CACHE}" \
  OUT_ROOT="${UNPAIRED_M50U50_ROOT}" \
  RATIOS="0.5:0.0:0.5" \
  ALLOW_EMPTY_SPLITS="1" \
  OVERWRITE="1"

echo "[submitted] preprocess job ids: ${PREPROC_JOB_IDS}"
echo "[source_cache] ${SOURCE_CACHE}"
echo "[unpaired_root] ${UNPAIRED_ROOT}"
