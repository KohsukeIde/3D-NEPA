#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N cqa_probe

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

cd "${WORKDIR}"
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

RUN_TAG="${RUN_TAG:-cqa_probe_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT=...}"
PROBE_TARGET="${PROBE_TARGET:?set PROBE_TARGET=curvature|signed_normal}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
SAVE_DIR="${SAVE_DIR:-${WORKDIR}/runs/cqa_probe}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/cqa_probe/${RUN_TAG}.json}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_probe}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train_mesh}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
MAX_STEPS="${MAX_STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-500}"
BATCH="${BATCH:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
TRAIN_QUERY_ORDER="${TRAIN_QUERY_ORDER:-shuffled}"
EVAL_QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-128}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"

mkdir -p "${LOG_ROOT}" "$(dirname "${OUT_JSON}")" "${SAVE_DIR}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== CQA FROZEN GEOMETRIC PROBE ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "cache_root=${CACHE_ROOT}" | tee -a "${LOG_PATH}"
echo "probe_target=${PROBE_TARGET}" | tee -a "${LOG_PATH}"
echo "manifest_json=${MANIFEST_JSON}" | tee -a "${LOG_PATH}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "out_json=${OUT_JSON}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.analysis.train_frozen_geometric_probe \
  --ckpt "${CKPT}" \
  --cache_root "${CACHE_ROOT}" \
  --probe_target "${PROBE_TARGET}" \
  --manifest_json "${MANIFEST_JSON}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_TAG}" \
  --out_json "${OUT_JSON}" \
  --train_split "${TRAIN_SPLIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --max_steps "${MAX_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --train_query_order "${TRAIN_QUERY_ORDER}" \
  --eval_query_order "${EVAL_QUERY_ORDER}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --controls "${CONTROLS}" \
  --device cuda \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
