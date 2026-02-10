#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_ft_scanobj
#PBS -o nepa3d_ft_scanobj.out
#PBS -e nepa3d_ft_scanobj.err

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
CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_cache_v1}"
BACKEND="${BACKEND:-pointcloud_noray}"
# Use n_types=5 ckpt by default (compatible with pointcloud_noray / TYPE_MISSING_RAY).
CKPT="${CKPT:-runs/querynepa3d_ab_meshpre_p0/ckpt_ep049.pt}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SEED="${EVAL_SEED:-0}"
MC_EVAL_K="${MC_EVAL_K:-4}"
SEED="${SEED:-0}"
ADD_EOS="${ADD_EOS:--1}"

"${PYTHON_BIN}" -m nepa3d.train.finetune_cls \
  --cache_root "${CACHE_ROOT}" \
  --backend "${BACKEND}" \
  --ckpt "${CKPT}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" \
  --n_ray "${N_RAY}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --add_eos "${ADD_EOS}" \
  --eval_seed "${EVAL_SEED}" \
  --mc_eval_k "${MC_EVAL_K}"
