#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=64
#PBS -l walltime=24:00:00
#PBS -P gag51403
#PBS -N nepa3d_preprocess
#PBS -o nepa3d_preprocess.out
#PBS -e nepa3d_preprocess.err

set -euo pipefail

# Environment setup
source /etc/profile.d/modules.sh

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
MODELNET_ROOT="${MODELNET_ROOT:-data/ModelNet40}"
OUT_ROOT="${OUT_ROOT:-data/modelnet40_cache_v0}"
SPLIT="${SPLIT:-all}"  # train, test, or all

PC_POINTS="${PC_POINTS:-1024}"
PT_POOL="${PT_POOL:-2000}"
RAY_POOL="${RAY_POOL:-1000}"
N_VIEWS="${N_VIEWS:-10}"
RAYS_PER_VIEW="${RAYS_PER_VIEW:-200}"
SEED="${SEED:-0}"
PC_GRID="${PC_GRID:-64}"
PC_DILATE="${PC_DILATE:-1}"
PC_MAX_STEPS="${PC_MAX_STEPS:-0}"
NO_PC_RAYS="${NO_PC_RAYS:-0}"

WORKERS="${N_WORKERS:-${PBS_NP:-64}}"
CHUNK_SIZE="${CHUNK_SIZE:-2}"

run_split() {
  split="$1"
  "${PYTHON_BIN}" nepa3d/data/preprocess_modelnet40.py \
    --modelnet_root "${MODELNET_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --split "${split}" \
    --pc_points "${PC_POINTS}" \
    --pt_pool "${PT_POOL}" \
    --ray_pool "${RAY_POOL}" \
    --n_views "${N_VIEWS}" \
    --rays_per_view "${RAYS_PER_VIEW}" \
    --pc_grid "${PC_GRID}" \
    --pc_dilate "${PC_DILATE}" \
    --pc_max_steps "${PC_MAX_STEPS}" \
    $( [ "${NO_PC_RAYS}" = "1" ] && echo "--no_pc_rays" ) \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --chunk_size "${CHUNK_SIZE}"
}

if [ "${SPLIT}" = "all" ]; then
  run_split train
  run_split test
else
  run_split "${SPLIT}"
fi
