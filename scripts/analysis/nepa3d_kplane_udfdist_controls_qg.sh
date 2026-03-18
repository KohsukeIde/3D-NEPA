#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N kp_eval

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
RUN_TAG="${RUN_TAG:-kplane_udfdist_eval_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/kplane_udfdist_eval}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/kplane_udfdist_eval/${RUN_TAG}.json}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-256}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
TAU="${TAU:-0.05}"
mkdir -p "${LOG_ROOT}" "$(dirname "${OUT_JSON}")"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"
cd "${WORKDIR}"
[[ -f "${VENV_ACTIVATE}" ]] && source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE}" 2>/dev/null || true
python -u -m nepa3d.tracks.kplane.analysis.eval_udfdist_worldv3_controls \
  --ckpt "${CKPT}" \
  --mix_config_path "${MIX_CONFIG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --max_samples_per_task "${MAX_SAMPLES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --controls "${CONTROLS}" \
  --tau "${TAU}" \
  --output_json "${OUT_JSON}" \
  2>&1 | tee "${LOG_PATH}"
