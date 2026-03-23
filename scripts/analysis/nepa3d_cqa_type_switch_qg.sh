#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -N cqa_tsw

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

RUN_TAG="${RUN_TAG:-cqa_type_switch_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_multitype}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/results/cqa_multitype/${RUN_TAG}}"

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

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== CQA TYPE-SWITCH ASSETS ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "cache_root=${CACHE_ROOT}" | tee -a "${LOG_PATH}"
echo "context_source=${CONTEXT_SOURCE}" | tee -a "${LOG_PATH}"
echo "tasks=${TASKS}" | tee -a "${LOG_PATH}"
echo "output_root=${OUTPUT_ROOT}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.export_promptable_type_switch_assets \
  --ckpt "${CKPT}" \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --max_shapes "${MAX_SHAPES}" \
  --sample_mode "${SAMPLE_MODE}" \
  --context_source "${CONTEXT_SOURCE}" \
  --n_ctx "${N_CTX}" \
  --n_qry_surface "${N_QRY_SURFACE}" \
  --tasks "${TASKS}" \
  --distance_grid_res "${DISTANCE_GRID_RES}" \
  --distance_chunk_n_query "${DISTANCE_CHUNK_N_QUERY}" \
  --distance_mc_level "${DISTANCE_MC_LEVEL}" \
  --output_root "${OUTPUT_ROOT}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
