#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SINGLE_SUBMIT="${SCRIPT_DIR}/submit_world_v3_mesh_aux_qc.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -x "${SINGLE_SUBMIT}" ]]; then
  echo "[error] missing submit helper: ${SINGLE_SUBMIT}"
  exit 1
fi

RUN_TAG_BASE="${RUN_TAG_BASE:-world_v3_mesh_aux_sharded_$(date +%Y%m%d_%H%M%S)}"
NUM_SHARDS="${NUM_SHARDS:-8}"

SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
DST_CACHE_ROOT="${DST_CACHE_ROOT:?set DST_CACHE_ROOT}"
SPLIT="${SPLIT:-eval}"
LIMIT="${LIMIT:-64}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
REFRESH="${REFRESH:-0}"
COMPUTE_AO_HQ="${COMPUTE_AO_HQ:-1}"
COMPUTE_HKS="${COMPUTE_HKS:-0}"
AO_RAYS="${AO_RAYS:-128}"
AO_EPS="${AO_EPS:-1e-4}"
AO_MAX_T="${AO_MAX_T:-2.5}"
HKS_EIGS="${HKS_EIGS:-64}"
HKS_TIMES="${HKS_TIMES:-0.05,0.2,1.0}"
SUFFIX="${SUFFIX:-}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
WALLTIME="${WALLTIME:-12:00:00}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${WORKDIR}/results/data_freeze/${RUN_TAG_BASE}}"

mkdir -p "${OUT_DIR_BASE}"

echo "[sharded-submit] run_tag_base=${RUN_TAG_BASE}"
echo "[sharded-submit] num_shards=${NUM_SHARDS}"
echo "[sharded-submit] dst_cache_root=${DST_CACHE_ROOT}"

for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  run_tag="${RUN_TAG_BASE}_s$(printf '%02d' "${shard}")of$(printf '%02d' "${NUM_SHARDS}")"
  copy_meta="0"
  if [[ "${shard}" == "0" ]]; then
    copy_meta="1"
  fi
  out_dir="${OUT_DIR_BASE}/shard_${shard}"
  mkdir -p "${out_dir}"
  echo "[submit] shard=${shard}/${NUM_SHARDS}"
  env \
    WORKDIR="${WORKDIR}" \
    GROUP_LIST="${GROUP_LIST}" \
    WALLTIME="${WALLTIME}" \
    RUN_TAG="${run_tag}" \
    SRC_CACHE_ROOT="${SRC_CACHE_ROOT}" \
    DST_CACHE_ROOT="${DST_CACHE_ROOT}" \
    SPLIT="${SPLIT}" \
    LIMIT="${LIMIT}" \
    SAMPLE_SEED="${SAMPLE_SEED}" \
    NUM_SHARDS="${NUM_SHARDS}" \
    SHARD_INDEX="${shard}" \
    REFRESH="${REFRESH}" \
    COPY_META="${copy_meta}" \
    COMPUTE_AO_HQ="${COMPUTE_AO_HQ}" \
    AO_RAYS="${AO_RAYS}" \
    AO_EPS="${AO_EPS}" \
    AO_MAX_T="${AO_MAX_T}" \
    COMPUTE_HKS="${COMPUTE_HKS}" \
    HKS_EIGS="${HKS_EIGS}" \
    HKS_TIMES="${HKS_TIMES}" \
    SUFFIX="${SUFFIX}" \
    OUT_DIR="${out_dir}" \
    OUTPUT_JSON="${out_dir}/mesh_aux_summary.json" \
    "${SINGLE_SUBMIT}"
done
