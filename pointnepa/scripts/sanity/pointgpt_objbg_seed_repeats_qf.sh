#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N ptgpt_obj_seed
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"

idx="${PBS_ARRAY_INDEX:-0}"
variants=(official official nomask nomask nomask_orderrandom nomask_orderrandom)
seeds=(1 2 1 2 1 2)
variant="${variants[$idx]}"
seed="${seeds[$idx]}"

case "${variant}" in
  official)
    ckpt="${POINTGPT_DIR}/checkpoints/official/pointgpt_s_pretrain_official.pth"
    exp="pgpt_s_official_objbg_e300_seed${seed}"
    ;;
  nomask)
    ckpt="${POINTGPT_DIR}/experiments/pretrain_nomask/PointGPT-S/pgpt_s_nomask_e300/ckpt-last.pth"
    exp="pgpt_s_nomask_objbg_e300_seed${seed}"
    ;;
  nomask_orderrandom)
    ckpt="${POINTGPT_DIR}/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300/ckpt-last.pth"
    exp="pgpt_s_nomask_ordrand_objbg_e300_seed${seed}"
    ;;
  *)
    echo "[error] unknown variant ${variant}"
    exit 2
    ;;
esac

mkdir -p "${WORKDIR}/logs/sanity"

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
export EXP_NAME="${exp}"
export CKPT_PATH="${ckpt}"
export NPROC_PER_NODE="4"
export NUM_WORKERS="8"
export USE_WANDB="0"
export EXTRA_ARGS="--seed ${seed}"
export MASTER_PORT="$((29600 + idx))"

cd "${POINTGPT_DIR}"

echo "=== PointGPT-S ScanObjectNN obj_bg seed repeat ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "array_index=${idx}"
echo "variant=${variant}"
echo "seed=${seed}"
echo "exp=${EXP_NAME}"
echo "ckpt=${CKPT_PATH}"
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

bash ../scripts/local/pointgpt_finetune_local_ddp.sh

echo "[done] PointGPT-S obj_bg seed repeat"
