#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_finetune
#PBS -o nepa3d_finetune.out
#PBS -e nepa3d_finetune.err

set -eu

# Environment setup
. /etc/profile.d/modules.sh
module load cuda/12.6

# Move to working directory
cd /groups/gag51403/ide/3D-NEPA

# Activate Python virtual environment
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v0}"
BACKEND="${BACKEND:-mesh}"
TRAIN_BACKEND="${TRAIN_BACKEND:-}"
EVAL_BACKEND="${EVAL_BACKEND:-}"
CKPT="${CKPT:-runs/querynepa3d_meshpre_v0/ckpt_ep049.pt}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SEED="${EVAL_SEED:-0}"
MC_EVAL_K="${MC_EVAL_K:-1}"
SEED="${SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
DROP_RAY_PROB_TRAIN="${DROP_RAY_PROB_TRAIN:-0.0}"
FORCE_MISSING_RAY="${FORCE_MISSING_RAY:-0}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
ADD_EOS="${ADD_EOS:--1}"
VOXEL_GRID="${VOXEL_GRID:-64}"
VOXEL_DILATE="${VOXEL_DILATE:-1}"
VOXEL_MAX_STEPS="${VOXEL_MAX_STEPS:-0}"
SAVE_DIR="${SAVE_DIR:-}"

EXTRA_FORCE=""
if [ "${FORCE_MISSING_RAY}" = "1" ]; then
  EXTRA_FORCE="--force_missing_ray"
fi

EXTRA_TRAIN_EVAL=""
if [ -n "${TRAIN_BACKEND}" ]; then
  EXTRA_TRAIN_EVAL="${EXTRA_TRAIN_EVAL} --train_backend ${TRAIN_BACKEND}"
fi
if [ -n "${EVAL_BACKEND}" ]; then
  EXTRA_TRAIN_EVAL="${EXTRA_TRAIN_EVAL} --eval_backend ${EVAL_BACKEND}"
fi

EXTRA_FREEZE=""
if [ "${FREEZE_BACKBONE}" = "1" ]; then
  EXTRA_FREEZE="--freeze_backbone"
fi

EXTRA_SAVE=""
if [ -n "${SAVE_DIR}" ]; then
  EXTRA_SAVE="--save_dir ${SAVE_DIR}"
fi

"${PYTHON_BIN}" -m nepa3d.train.finetune_cls \
  --cache_root "${CACHE_ROOT}" \
  --backend "${BACKEND}" \
  --ckpt "${CKPT}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --num_workers "${NUM_WORKERS}" \
  --drop_ray_prob_train "${DROP_RAY_PROB_TRAIN}" \
  --add_eos "${ADD_EOS}" \
  --val_ratio "${VAL_RATIO}" \
  --val_seed "${VAL_SEED}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --seed "${SEED}" \
  ${EXTRA_TRAIN_EVAL} \
  ${EXTRA_FREEZE} \
  ${EXTRA_SAVE} \
  ${EXTRA_FORCE} \
  --eval_seed "${EVAL_SEED}" --mc_eval_k "${MC_EVAL_K}"
