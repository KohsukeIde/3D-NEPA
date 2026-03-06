#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N pntok_cpac

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

RUN_TAG="${RUN_TAG:-patchnepa_cpac_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT=...}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/patch_nepa_cpac}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKDIR}/results}"

HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
REP_SOURCE="${REP_SOURCE:-h}"   # h | zhat
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
TAU="${TAU:-0.01}"
SEED="${SEED:-0}"
ANSWER_IN_DIM="${ANSWER_IN_DIM:-0}"
SURF_XYZ_KEY="${SURF_XYZ_KEY:-}"
QRY_XYZ_KEY="${QRY_XYZ_KEY:-}"
QRY_DIST_KEY="${QRY_DIST_KEY:-}"
CONTEXT_PRIMITIVE="${CONTEXT_PRIMITIVE:-generic}"
QUERY_PRIMITIVE="${QUERY_PRIMITIVE:-udf}"

# mini-CPAC defaults for screening
MINI_CPAC="${MINI_CPAC:-1}"
N_CTX_POINTS="${N_CTX_POINTS:-1024}"
N_QUERY="${N_QUERY:-1024}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-1024}"
if [[ "${MINI_CPAC}" == "1" ]]; then
  MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES:-64}"
  MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES:-64}"
else
  MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES:-0}"
  MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES:-0}"
fi

MESH_EVAL="${MESH_EVAL:-0}"
MESH_GRID_RES="${MESH_GRID_RES:-24}"
MESH_MC_LEVEL="${MESH_MC_LEVEL:-0.03}"
MESH_FSCORE_TAU="${MESH_FSCORE_TAU:-0.01}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"

mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"
OUT_JSON="${OUT_JSON:-${RESULTS_ROOT}/cpac_patchnepa_${RUN_TAG}.json}"

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== PATCHNEPA CPAC(UDF) ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "data_root=${DATA_ROOT}" | tee -a "${LOG_PATH}"
echo "head_train_split=${HEAD_TRAIN_SPLIT} eval_split=${EVAL_SPLIT}" | tee -a "${LOG_PATH}"
echo "mini_cpac=${MINI_CPAC} train_shapes=${MAX_TRAIN_SHAPES} eval_shapes=${MAX_EVAL_SHAPES}" | tee -a "${LOG_PATH}"
echo "n_ctx=${N_CTX_POINTS} n_query=${N_QUERY} chunk_n_query=${CHUNK_N_QUERY}" | tee -a "${LOG_PATH}"
echo "rep_source=${REP_SOURCE} ridge_alpha=${RIDGE_ALPHA} tau=${TAU} seed=${SEED}" | tee -a "${LOG_PATH}"
echo "context_primitive=${CONTEXT_PRIMITIVE} query_primitive=${QUERY_PRIMITIVE}" | tee -a "${LOG_PATH}"
echo "surf_xyz_key=${SURF_XYZ_KEY:-<auto>} qry_xyz_key=${QRY_XYZ_KEY:-<auto>} qry_dist_key=${QRY_DIST_KEY:-<auto>}" | tee -a "${LOG_PATH}"
echo "mesh_eval=${MESH_EVAL}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] ckpt not found: ${CKPT}" | tee -a "${LOG_PATH}"
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[error] data_root not found: ${DATA_ROOT}" | tee -a "${LOG_PATH}"
  exit 2
fi

ARGS=(
  --ckpt "${CKPT}"
  --data_root "${DATA_ROOT}"
  --head_train_split "${HEAD_TRAIN_SPLIT}"
  --eval_split "${EVAL_SPLIT}"
  --max_train_shapes "${MAX_TRAIN_SHAPES}"
  --max_eval_shapes "${MAX_EVAL_SHAPES}"
  --n_ctx_points "${N_CTX_POINTS}"
  --n_query "${N_QUERY}"
  --chunk_n_query "${CHUNK_N_QUERY}"
  --rep_source "${REP_SOURCE}"
  --answer_in_dim "${ANSWER_IN_DIM}"
  --surf_xyz_key "${SURF_XYZ_KEY}"
  --qry_xyz_key "${QRY_XYZ_KEY}"
  --qry_dist_key "${QRY_DIST_KEY}"
  --context_primitive "${CONTEXT_PRIMITIVE}"
  --query_primitive "${QUERY_PRIMITIVE}"
  --ridge_alpha "${RIDGE_ALPHA}"
  --tau "${TAU}"
  --seed "${SEED}"
  --out_json "${OUT_JSON}"
)

if [[ "${MESH_EVAL}" == "1" ]]; then
  ARGS+=(
    --mesh_eval
    --mesh_grid_res "${MESH_GRID_RES}"
    --mesh_mc_level "${MESH_MC_LEVEL}"
    --mesh_fscore_tau "${MESH_FSCORE_TAU}"
    --mesh_num_samples "${MESH_NUM_SAMPLES}"
  )
fi

python -u -m nepa3d.analysis.completion_cpac_udf_patchnepa "${ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"
echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
