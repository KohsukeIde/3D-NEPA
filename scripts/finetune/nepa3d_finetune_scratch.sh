#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_finetune_scratch
#PBS -o nepa3d_finetune_scratch.out
#PBS -e nepa3d_finetune_scratch.err

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
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v1}"
BACKEND="${BACKEND:-mesh}"
TRAIN_BACKEND="${TRAIN_BACKEND:-}"
EVAL_BACKEND="${EVAL_BACKEND:-}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
MC_EVAL_K="${MC_EVAL_K:-1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
SAVE_DIR="${SAVE_DIR:-runs/querynepa3d_scratch_s${SEED}}"
SCRATCH_CKPT="${SCRATCH_CKPT:-${SAVE_DIR}/ckpt_ep000.pt}"
ADD_EOS="${ADD_EOS:-1}"
N_TYPES="${N_TYPES:-5}"
DROP_RAY_PROB_TRAIN="${DROP_RAY_PROB_TRAIN:-0.0}"
FORCE_MISSING_RAY="${FORCE_MISSING_RAY:-0}"
VOXEL_GRID="${VOXEL_GRID:-64}"
VOXEL_DILATE="${VOXEL_DILATE:-1}"
VOXEL_MAX_STEPS="${VOXEL_MAX_STEPS:-0}"

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

# Build scratch checkpoint (random init)
"${PYTHON_BIN}" - <<PY
import os
import torch
from nepa3d.models.query_nepa import QueryNepa
from nepa3d.utils.seed import set_seed

seed = int("${SEED}")
n_point = int("${N_POINT}")
n_ray = int("${N_RAY}")
d_model = int("${D_MODEL}")
layers = int("${LAYERS}")
heads = int("${HEADS}")
save_dir = "${SAVE_DIR}"
ckpt_path = "${SCRATCH_CKPT}"
add_eos = bool(int("${ADD_EOS}"))
n_types = int("${N_TYPES}")

set_seed(seed)
t = 1 + n_point + n_ray + (1 if add_eos else 0)
model = QueryNepa(
    feat_dim=15,
    d_model=d_model,
    n_types=n_types,
    nhead=heads,
    num_layers=layers,
    max_len=t,
)

os.makedirs(save_dir, exist_ok=True)
ckpt = {
    "model": model.state_dict(),
    "args": {
        "n_point": n_point,
        "n_ray": n_ray,
        "d_model": d_model,
        "layers": layers,
        "heads": heads,
        "add_eos": int(add_eos),
        "n_types": n_types,
        "backend": "scratch",
        "seed": seed,
    },
    "epoch": 0,
}
torch.save(ckpt, ckpt_path)
print(f"scratch ckpt saved: {ckpt_path}")
PY

# Finetune from scratch ckpt
"${PYTHON_BIN}" -m nepa3d.train.finetune_cls \
  --cache_root "${CACHE_ROOT}" \
  --backend "${BACKEND}" \
  --ckpt "${SCRATCH_CKPT}" \
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
  --save_dir "${SAVE_DIR}" \
  ${EXTRA_FORCE} \
  --eval_seed "${EVAL_SEED}" --mc_eval_k "${MC_EVAL_K}"
