#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P gag51403
#PBS -N nepa3d_pretrain
#PBS -o nepa3d_pretrain.out
#PBS -e nepa3d_pretrain.err

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
MIX_CONFIG="${MIX_CONFIG:-}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-0}"
MIX_SEED="${MIX_SEED:-0}"
BACKEND="${BACKEND:-mesh}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
SAVE_DIR="${SAVE_DIR:-runs/querynepa3d_meshpre_v0}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
DROP_RAY_PROB="${DROP_RAY_PROB:-0.0}"
FORCE_MISSING_RAY="${FORCE_MISSING_RAY:-0}"
ADD_EOS="${ADD_EOS:-1}"
VOXEL_GRID="${VOXEL_GRID:-64}"
VOXEL_DILATE="${VOXEL_DILATE:-1}"
VOXEL_MAX_STEPS="${VOXEL_MAX_STEPS:-0}"
OBJECTIVE="${OBJECTIVE:-nepa}"
MASK_RATIO="${MASK_RATIO:-0.4}"

EXTRA_FORCE=""
if [ "${FORCE_MISSING_RAY}" = "1" ]; then
  EXTRA_FORCE="--force_missing_ray"
fi

"${PYTHON_BIN}" -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples "${MIX_NUM_SAMPLES}" \
  --mix_seed "${MIX_SEED}" \
  --backend "${BACKEND}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --num_workers "${NUM_WORKERS}" \
  --drop_ray_prob "${DROP_RAY_PROB}" \
  --add_eos "${ADD_EOS}" \
  --objective "${OBJECTIVE}" \
  --mask_ratio "${MASK_RATIO}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --seed "${SEED}" \
  ${EXTRA_FORCE} \
  --save_dir "${SAVE_DIR}"
