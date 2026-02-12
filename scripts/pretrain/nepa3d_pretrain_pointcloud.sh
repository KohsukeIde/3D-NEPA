#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P gag51403
#PBS -N nepa3d_pretrain_pc
#PBS -o nepa3d_pretrain_pc.out
#PBS -e nepa3d_pretrain_pc.err

set -eu

. /etc/profile.d/modules.sh
module load cuda/12.6

cd /groups/gag51403/ide/3D-NEPA

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v0}"
BACKEND="${BACKEND:-pointcloud}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
SAVE_DIR="${SAVE_DIR:-runs/querynepa3d_pcpre_v0}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${PYTHON_BIN}" -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --backend "${BACKEND}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --num_workers "${NUM_WORKERS}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --save_dir "${SAVE_DIR}"
