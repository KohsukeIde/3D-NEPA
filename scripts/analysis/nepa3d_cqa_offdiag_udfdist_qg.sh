#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N cqa_odg

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

RUN_TAG="${RUN_TAG:-cqa_offdiag_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_eval}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/cqa_eval/${RUN_TAG}.json}"

DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
TASK_FILTER="${TASK_FILTER:-udf_distance}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,wrong_type,shuffled_query}"

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

echo "=== CQA ZERO-SHOT OFFDIAG EVAL ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "split_override=${SPLIT_OVERRIDE}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.eval_primitive_answering_controls \
  --ckpt "${CKPT}" \
  --mix_config_path "${MIX_CONFIG}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --max_samples_per_task "${MAX_SAMPLES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --task_filter "${TASK_FILTER}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --controls "${CONTROLS}" \
  --output_json "${OUT_JSON}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
