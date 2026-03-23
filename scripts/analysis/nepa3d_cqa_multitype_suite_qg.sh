#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -N cqa_mts

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

RUN_TAG="${RUN_TAG:-cqa_multitype_suite_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm.yaml}"
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_pcbank_eval.yaml}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_multitype}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/cqa_multitype/${RUN_TAG}.json}"
OUT_CSV="${OUT_CSV:-${WORKDIR}/results/cqa_multitype/${RUN_TAG}.csv}"
OUT_MD="${OUT_MD:-${WORKDIR}/results/cqa_multitype/${RUN_TAG}.md}"

DEVICE="${DEVICE:-cuda}"
BATCH="${BATCH:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
TASKS="${TASKS:-udf_distance,mesh_normal}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
QUERY_ORDER="${QUERY_ORDER:-sampled}"

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

echo "=== CQA MULTITYPE SUITE ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "same_mix_config=${SAME_MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "offdiag_mix_config=${OFFDIAG_MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "tasks=${TASKS}" | tee -a "${LOG_PATH}"
echo "query_order=${QUERY_ORDER}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.eval_multitype_cqa_suite \
  --ckpt "${CKPT}" \
  --same_mix_config "${SAME_MIX_CONFIG}" \
  --offdiag_mix_config "${OFFDIAG_MIX_CONFIG}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --max_samples_per_task "${MAX_SAMPLES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --tasks "${TASKS}" \
  --controls "${CONTROLS}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --query_order "${QUERY_ORDER}" \
  --output_json "${OUT_JSON}" \
  --output_csv "${OUT_CSV}" \
  --output_md "${OUT_MD}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
