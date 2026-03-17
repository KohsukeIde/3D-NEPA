#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N cqa_cls

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

RUN_TAG="${RUN_TAG:-cqa_cls_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT=...}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-test}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-pointmae}"
CKPT="${CKPT:?set CKPT=...}"
SAVE_DIR="${SAVE_DIR:-${WORKDIR}/runs/cqa_cls}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/cqa_cls}"

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
N_POINT="${N_POINT:-2048}"
SAMPLE_MODE_TRAIN="${SAMPLE_MODE_TRAIN:-random}"
SAMPLE_MODE_EVAL="${SAMPLE_MODE_EVAL:-random}"
POOL="${POOL:-mean}"
POINTMAE_AUG="${POINTMAE_AUG:-1}"

mkdir -p "${LOG_ROOT}" "${SAVE_DIR}"
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

echo "=== CQA CLASSIFICATION UTILITY ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "cache_root=${CACHE_ROOT}" | tee -a "${LOG_PATH}"
echo "train_split=${TRAIN_SPLIT} val_split=${VAL_SPLIT} val_split_mode=${VAL_SPLIT_MODE}" | tee -a "${LOG_PATH}"
echo "ckpt=${CKPT}" | tee -a "${LOG_PATH}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

python -u -m nepa3d.tracks.patch_nepa.cqa.train.finetune_primitive_answering_cls \
  --cache_root "${CACHE_ROOT}" \
  --train_split "${TRAIN_SPLIT}" \
  --val_split "${VAL_SPLIT}" \
  --val_split_mode "${VAL_SPLIT_MODE}" \
  --ckpt "${CKPT}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_TAG}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_point "${N_POINT}" \
  --sample_mode_train "${SAMPLE_MODE_TRAIN}" \
  --sample_mode_eval "${SAMPLE_MODE_EVAL}" \
  --pool "${POOL}" \
  --pointmae_aug "${POINTMAE_AUG}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
