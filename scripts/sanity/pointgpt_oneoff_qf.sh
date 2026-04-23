#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -N pointgpt_oneoff

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointgpt_oneoff}"
RUN_TAG="${RUN_TAG:-pointgpt_oneoff_$(date +%Y%m%d_%H%M%S)}"
POINTGPT_CMD="${POINTGPT_CMD:-}"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

if [[ -z "${POINTGPT_CMD}" ]]; then
  echo "[error] POINTGPT_CMD is required"
  exit 2
fi
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi
if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}"
  exit 2
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
  module load "${GCC_MODULE}" 2>/dev/null || true
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi

export TORCH_CUDA_ARCH_LIST
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

cd "${POINTGPT_DIR}"

echo "=== POINTGPT ONEOFF ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "pointgpt_dir=${POINTGPT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "command=${POINTGPT_CMD}"
echo

python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops import: OK", pointnet2_utils.__name__)
from knn_cuda import KNN
print("knn_cuda import: OK", KNN.__name__)
PY

bash -lc "${POINTGPT_CMD}"

echo "[done] PointGPT oneoff completed"
