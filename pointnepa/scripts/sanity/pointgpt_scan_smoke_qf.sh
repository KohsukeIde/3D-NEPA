#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -N pointgpt_scan_smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointgpt_scan_smoke}"
RUN_TAG="${RUN_TAG:-pointgpt_scan_smoke_$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-cfgs/local/finetune_scan_objbg_smoke.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OBJ_BG_ROOT="${OBJ_BG_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/obj_bg}"
OBJ_ONLY_ROOT="${OBJ_ONLY_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/obj_only}"
PB_T50_ROOT="${PB_T50_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/pb_t50_rs}"

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

for f in \
  "${OBJ_BG_ROOT}/training_objectdataset.h5" \
  "${OBJ_BG_ROOT}/test_objectdataset.h5" \
  "${OBJ_ONLY_ROOT}/training_objectdataset.h5" \
  "${OBJ_ONLY_ROOT}/test_objectdataset.h5" \
  "${PB_T50_ROOT}/training_objectdataset_augmentedrot_scale75.h5" \
  "${PB_T50_ROOT}/test_objectdataset_augmentedrot_scale75.h5"
do
  if [[ ! -f "${f}" ]]; then
    echo "[error] required file missing: ${f}"
    exit 2
  fi
done

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

mkdir -p "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split"
mkdir -p "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split_nobg"

ln -sfn "${OBJ_BG_ROOT}/training_objectdataset.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split/training_objectdataset.h5"
ln -sfn "${OBJ_BG_ROOT}/test_objectdataset.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split/test_objectdataset.h5"
ln -sfn "${PB_T50_ROOT}/training_objectdataset_augmentedrot_scale75.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5"
ln -sfn "${PB_T50_ROOT}/test_objectdataset_augmentedrot_scale75.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5"

ln -sfn "${OBJ_ONLY_ROOT}/training_objectdataset.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split_nobg/training_objectdataset.h5"
ln -sfn "${OBJ_ONLY_ROOT}/test_objectdataset.h5" \
  "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split_nobg/test_objectdataset.h5"

cd "${POINTGPT_DIR}"

echo "=== POINTGPT SCAN SMOKE ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "pointgpt_dir=${POINTGPT_DIR}"
echo "venv=${VENV_ACTIVATE}"
echo "config=${CONFIG_PATH}"
echo "exp_name=${EXP_NAME}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
echo

python -V
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torchstat") else 1)
PY
then
  python -m pip install -q torchstat
fi
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
  --scratch_model \
  --exp_name "${EXP_NAME}" \
  --val_freq 1 \
  --num_workers "${NUM_WORKERS}"

echo "[done] PointGPT scan smoke completed"
