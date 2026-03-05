#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N pointgpt_pretrain_shn

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointgpt_pretrain_shapenet}"
RUN_TAG="${RUN_TAG:-pointgpt_pretrain_shapenet_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-S/pretrain.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SHAPENET_CACHE_ROOT="${SHAPENET_CACHE_ROOT:-${WORKDIR}/data/shapenet_cache_v2_20260303}"
FORCE_REBUILD_SPLITS="${FORCE_REBUILD_SPLITS:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_shapenet_pretrain}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-pointgpt,pretrain,shapenet,scratch}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi
if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}"
  exit 2
fi
if [[ ! -d "${SHAPENET_CACHE_ROOT}/train" || ! -d "${SHAPENET_CACHE_ROOT}/test" ]]; then
  echo "[error] ShapeNet cache root missing train/test: ${SHAPENET_CACHE_ROOT}"
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

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torchstat") else 1)
PY
then
  python -m pip install -q torchstat
fi
if [[ "${USE_WANDB}" == "1" ]] && ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("wandb") else 1)
PY
then
  python -m pip install -q wandb
fi
if [[ "${USE_WANDB}" == "1" ]] && ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tensorboard") else 1)
PY
then
  python -m pip install -q tensorboard
fi

export TORCH_CUDA_ARCH_LIST
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_GROUP
export WANDB_RUN_NAME
export WANDB_TAGS
export WANDB_DIR

SHAPENET55_ROOT="${POINTGPT_DIR}/data/ShapeNet55-34/ShapeNet-55"
SHAPENET55_PC="${POINTGPT_DIR}/data/ShapeNet55-34/shapenet_pc"
mkdir -p "${SHAPENET55_ROOT}"
mkdir -p "$(dirname "${SHAPENET55_PC}")"
ln -sfn "${SHAPENET_CACHE_ROOT}" "${SHAPENET55_PC}"

if [[ "${FORCE_REBUILD_SPLITS}" == "1" || ! -s "${SHAPENET55_ROOT}/train.txt" || ! -s "${SHAPENET55_ROOT}/test.txt" ]]; then
  find "${SHAPENET_CACHE_ROOT}/train" -type f -name '*.npz' \
    | sed "s#^${SHAPENET_CACHE_ROOT}/##" \
    | LC_ALL=C sort > "${SHAPENET55_ROOT}/train.txt"
  find "${SHAPENET_CACHE_ROOT}/test" -type f -name '*.npz' \
    | sed "s#^${SHAPENET_CACHE_ROOT}/##" \
    | LC_ALL=C sort > "${SHAPENET55_ROOT}/test.txt"
fi

cd "${POINTGPT_DIR}"

echo "=== POINTGPT PRETRAIN SHAPENET ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "pointgpt_dir=${POINTGPT_DIR}"
echo "venv=${VENV_ACTIVATE}"
echo "config=${CONFIG_PATH}"
echo "exp_name=${EXP_NAME}"
echo "shapenet_cache_root=${SHAPENET_CACHE_ROOT}"
echo "shapenet55_root=${SHAPENET55_ROOT}"
echo "shapenet55_pc=${SHAPENET55_PC}"
echo "train_count=$(wc -l < "${SHAPENET55_ROOT}/train.txt")"
echo "test_count=$(wc -l < "${SHAPENET55_ROOT}/test.txt")"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
echo "use_wandb=${USE_WANDB} project=${WANDB_PROJECT} group=${WANDB_GROUP} run=${WANDB_RUN_NAME}"
echo

python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops import: OK", pointnet2_utils.__name__)
import extensions.chamfer_dist as chamfer_dist
print("chamfer_dist import: OK", chamfer_dist.__name__)
from knn_cuda import KNN
print("knn_cuda import: OK", KNN.__name__)
PY

python main.py \
  --config "${CONFIG_PATH}" \
  --exp_name "${EXP_NAME}" \
  --val_freq 10 \
  --num_workers "${NUM_WORKERS}"

echo "[done] PointGPT pretrain ShapeNet completed"
