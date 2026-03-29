#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N cqa_deg

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

RUN_TAG="${RUN_TAG:-cqa_degraded_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
SAME_MIX_CONFIG="${SAME_MIX_CONFIG:?set SAME_MIX_CONFIG=...}"
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:?set OFFDIAG_MIX_CONFIG=...}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_degraded}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/cqa_degraded/${RUN_TAG}.json}"
OUT_CSV="${OUT_CSV:-${WORKDIR}/results/cqa_degraded/${RUN_TAG}.csv}"
OUT_MD="${OUT_MD:-${WORKDIR}/results/cqa_degraded/${RUN_TAG}.md}"
ASSETS_ROOT="${ASSETS_ROOT:-${WORKDIR}/results/cqa_degraded/${RUN_TAG}_assets}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
BATCH="${BATCH:-4}"
MAX_SHAPES="${MAX_SHAPES:-64}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
QUERY_ORDER="${QUERY_ORDER:-sampled}"
GRID_RES="${GRID_RES:-16}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}"
TAU_LIST="${TAU_LIST:-0.01,0.02,0.05}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"
MC_LEVEL="${MC_LEVEL:-0.05}"
DROPOUT_KEEP_LIST="${DROPOUT_KEEP_LIST:-0.50,0.25,0.10,0.05}"
GAUSSIAN_SIGMA_LIST="${GAUSSIAN_SIGMA_LIST:-0.01,0.02,0.05}"
EXPORT_ASSETS="${EXPORT_ASSETS:-1}"
MAX_ASSET_SHAPES="${MAX_ASSET_SHAPES:-8}"

mkdir -p "${LOG_ROOT}" "$(dirname "${OUT_JSON}")" "$(dirname "${OUT_CSV}")" "$(dirname "${OUT_MD}")" "${ASSETS_ROOT}"
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

echo "=== CQA UDFDIST DEGRADED ROBUSTNESS ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "same_mix_config=${SAME_MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "offdiag_mix_config=${OFFDIAG_MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "grid_res=${GRID_RES} max_shapes=${MAX_SHAPES}" | tee -a "${LOG_PATH}"
echo "dropout_keep_list=${DROPOUT_KEEP_LIST}" | tee -a "${LOG_PATH}"
echo "gaussian_sigma_list=${GAUSSIAN_SIGMA_LIST}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_degraded_cqa \
  --ckpt "${CKPT}" \
  --same_mix_config_path "${SAME_MIX_CONFIG}" \
  --offdiag_mix_config_path "${OFFDIAG_MIX_CONFIG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --batch_size "${BATCH}" \
  --max_shapes "${MAX_SHAPES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --query_order "${QUERY_ORDER}" \
  --grid_res "${GRID_RES}" \
  --chunk_n_query "${CHUNK_N_QUERY}" \
  --tau_list "${TAU_LIST}" \
  --mesh_num_samples "${MESH_NUM_SAMPLES}" \
  --mc_level "${MC_LEVEL}" \
  --dropout_keep_list "${DROPOUT_KEEP_LIST}" \
  --gaussian_sigma_list "${GAUSSIAN_SIGMA_LIST}" \
  --export_assets "${EXPORT_ASSETS}" \
  --max_asset_shapes "${MAX_ASSET_SHAPES}" \
  --assets_root "${ASSETS_ROOT}" \
  --output_json "${OUT_JSON}" \
  --output_csv "${OUT_CSV}" \
  --output_md "${OUT_MD}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
