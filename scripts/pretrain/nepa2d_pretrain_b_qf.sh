#!/usr/bin/env bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N nepa2d_b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

RUN_TAG="${RUN_TAG:-nepa2d_b_imagenet_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${RUN_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-Nepa-Pretrain}"

CONFIG_NAME="${CONFIG_NAME:-configs/pretrain/nepa-base-patch14-224}"
TRAIN_DIR="${TRAIN_DIR:-${WORKDIR}/data/ImageNet/train}"
VALIDATION_DIR="${VALIDATION_DIR:-${WORKDIR}/data/ImageNet/val}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXPERIMENT_NAME}}"
LOAD_FROM_DISK="${LOAD_FROM_DISK:-False}"
IMAGE_COLUMN_NAME="${IMAGE_COLUMN_NAME:-image}"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-4096}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-1600}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-3e-4}"
DIAG_COPY="${DIAG_COPY:-1}"
DIAG_EVERY="${DIAG_EVERY:-50}"
DIAG_K="${DIAG_K:-1}"

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/pretrain/nepa2d}"
mkdir -p "${LOG_ROOT}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"
: > "${LOG_PATH}"

if [[ ! -d "${TRAIN_DIR}" ]]; then
  echo "[error] train dir not found: ${TRAIN_DIR}"
  exit 2
fi
if [[ ! -d "${VALIDATION_DIR}" ]]; then
  echo "[error] validation dir not found: ${VALIDATION_DIR}"
  exit 2
fi

cd "${WORKDIR}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate not found: ${VENV_ACTIVATE}"
  exit 2
fi
# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# Dependency preflight for ImageNet AutoImageProcessor path.
# Known failure mode: transformers<->timm API mismatch
# (ImportError: cannot import name 'ImageNetInfo' from timm.data).
python - <<'PY' | tee -a "${LOG_PATH}"
import importlib
import sys

def _ver(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as e:
        return f"import_error:{e}"

tv = _ver("transformers")
tm = _ver("timm")
print(f"[deps] transformers={tv} timm={tm}")
try:
    from transformers import AutoImageProcessor  # noqa: F401
    from timm.data import ImageNetInfo  # noqa: F401
except Exception as e:
    print(f"[deps][error] incompatible transformers/timm stack: {e}")
    print("[deps][hint] ensure timm>=1.0.25 with transformers==4.56.2")
    sys.exit(3)
print("[deps] preflight OK")
PY

echo "=== 2D-NEPA PRETRAIN (B) ===" | tee -a "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "workdir=${WORKDIR}" | tee -a "${LOG_PATH}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_PATH}"
echo "experiment_name=${EXPERIMENT_NAME}" | tee -a "${LOG_PATH}"
echo "config_name=${CONFIG_NAME}" | tee -a "${LOG_PATH}"
echo "train_dir=${TRAIN_DIR}" | tee -a "${LOG_PATH}"
echo "validation_dir=${VALIDATION_DIR}" | tee -a "${LOG_PATH}"
echo "output_dir=${OUTPUT_DIR}" | tee -a "${LOG_PATH}"
echo "wandb_project=${WANDB_PROJECT}" | tee -a "${LOG_PATH}"
echo "batch: total=${TOTAL_BATCH_SIZE} per_device=${PER_DEVICE_BATCH_SIZE}" | tee -a "${LOG_PATH}"
echo "num_epochs=${NUM_EPOCHS} base_lr=${BASE_LEARNING_RATE}" | tee -a "${LOG_PATH}"
echo "diag_copy=${DIAG_COPY} diag_every=${DIAG_EVERY} diag_k=${DIAG_K}" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

export EXPERIMENT_NAME
export WANDB_PROJECT
export CONFIG_NAME
export TRAIN_DIR
export VALIDATION_DIR
export OUTPUT_DIR
export LOAD_FROM_DISK
export IMAGE_COLUMN_NAME
export TOTAL_BATCH_SIZE
export PER_DEVICE_BATCH_SIZE
export NUM_EPOCHS
export BASE_LEARNING_RATE
export DIAG_COPY
export DIAG_EVERY
export DIAG_K

bash "${WORKDIR}/scripts/pretrain/nepa_b.sh" 2>&1 | tee -a "${LOG_PATH}"
