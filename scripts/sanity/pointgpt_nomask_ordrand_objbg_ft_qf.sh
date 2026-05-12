#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -N ptgpt_orft

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"

mkdir -p "${WORKDIR}/logs/sanity/pointgpt_oneoff"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi

source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6/12.6.2 2>/dev/null || true
  module load gcc/11.4.1 2>/dev/null || true
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi

export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CONFIG_PATH="cfgs/PointGPT-S/finetune_scan_objbg.yaml"
export EXP_NAME="pgpt_s_nomask_ordrand_objbg_e300"
export CKPT_PATH="${WORKDIR}/PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300/ckpt-last.pth"
export NPROC_PER_NODE="4"
export NUM_WORKERS="8"
export USE_WANDB="0"

cd "${POINTGPT_DIR}"

echo "=== PointGPT no-mask order-randomized obj_bg fine-tune ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "pointgpt_dir=${POINTGPT_DIR}"
echo "config=${CONFIG_PATH}"
echo "exp_name=${EXP_NAME}"
echo "ckpt_path=${CKPT_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
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

bash ../scripts/local/pointgpt_s_nomask_ordrand_objbg_ft.sh

echo "[done] PointGPT no-mask order-randomized obj_bg fine-tune"
