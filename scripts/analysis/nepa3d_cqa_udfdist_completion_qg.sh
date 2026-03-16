#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N cqa_cmp

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

RUN_TAG="${RUN_TAG:-cqa_completion_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_completion}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/cqa_completion/${RUN_TAG}.json}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
BATCH="${BATCH:-4}"
MAX_SHAPES="${MAX_SHAPES:-16}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
TASK_FILTER="${TASK_FILTER:-udf_distance}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
GRID_RES="${GRID_RES:-12}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}"
TAU_LIST="${TAU_LIST:-0.01,0.02,0.05}"
MESH_EVAL="${MESH_EVAL:-0}"
MC_LEVEL="${MC_LEVEL:-0.02}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"

mkdir -p "${LOG_ROOT}" "$(dirname "${OUT_JSON}")"
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

echo "=== CQA UDFDIST COMPLETION ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "split_override=${SPLIT_OVERRIDE}" | tee -a "${LOG_PATH}"
echo "grid_res=${GRID_RES} max_shapes=${MAX_SHAPES} batch=${BATCH} chunk_n_query=${CHUNK_N_QUERY}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.completion_udfdist_cqa \
  --ckpt "${CKPT}" \
  --mix_config_path "${MIX_CONFIG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --batch_size "${BATCH}" \
  --max_shapes "${MAX_SHAPES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --task_filter "${TASK_FILTER}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --grid_res "${GRID_RES}" \
  --chunk_n_query "${CHUNK_N_QUERY}" \
  --tau_list "${TAU_LIST}" \
  --mesh_eval "${MESH_EVAL}" \
  --mc_level "${MC_LEVEL}" \
  --mesh_num_samples "${MESH_NUM_SAMPLES}" \
  --output_json "${OUT_JSON}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
