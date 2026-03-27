#!/usr/bin/env bash
#PBS -l rt_QC=1
#PBS -l walltime=12:00:00
#PBS -j oe
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SRC_CACHE_ROOT="${SRC_CACHE_ROOT:?set SRC_CACHE_ROOT}"
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
OUTPUT_JSON="${OUTPUT_JSON:?set OUTPUT_JSON}"

python -m nepa3d.data.augment_world_v3_mesh_aux \
  --src_cache_root "${SRC_CACHE_ROOT}" \
  --dst_cache_root "${DST_CACHE_ROOT}" \
  --split "${SPLIT}" \
  --limit "${LIMIT}" \
  --sample_seed "${SAMPLE_SEED}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${SHARD_INDEX}" \
  --refresh "${REFRESH}" \
  --copy_meta "${COPY_META}" \
  --compute_ao_hq "${COMPUTE_AO_HQ}" \
  --ao_rays "${AO_RAYS}" \
  --ao_eps "${AO_EPS}" \
  --ao_max_t "${AO_MAX_T}" \
  --ao_batch_size "${AO_BATCH_SIZE}" \
  --compute_hks "${COMPUTE_HKS}" \
  --hks_eigs "${HKS_EIGS}" \
  --hks_times "${HKS_TIMES}" \
  --suffix "${SUFFIX}" \
  --output_json "${OUTPUT_JSON}"
