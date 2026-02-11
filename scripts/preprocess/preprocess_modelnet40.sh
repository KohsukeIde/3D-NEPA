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
DF_GRID="${DF_GRID:-64}"
DF_DILATE="${DF_DILATE:-1}"
NO_UDF="${NO_UDF:-0}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
PT_QUERY_CHUNK="${PT_QUERY_CHUNK:-2048}"
RAY_QUERY_CHUNK="${RAY_QUERY_CHUNK:-2048}"
PT_DIST_MODE="${PT_DIST_MODE:-mesh}"
DIST_REF_POINTS="${DIST_REF_POINTS:-8192}"

WORKERS="${N_WORKERS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-2}"

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
    --df_grid "${DF_GRID}" \
    --df_dilate "${DF_DILATE}" \
    --pt_surface_ratio "${PT_SURFACE_RATIO}" \
    --pt_surface_sigma "${PT_SURFACE_SIGMA}" \
    --pt_query_chunk "${PT_QUERY_CHUNK}" \
    --ray_query_chunk "${RAY_QUERY_CHUNK}" \
    --pt_dist_mode "${PT_DIST_MODE}" \
    --dist_ref_points "${DIST_REF_POINTS}" \
    $( [ "${NO_UDF}" = "1" ] && echo "--no_udf" ) \
    $( [ "${NO_PC_RAYS}" = "1" ] && echo "--no_pc_rays" ) \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --max_tasks_per_child "${MAX_TASKS_PER_CHILD}"
}

if [ "${SPLIT}" = "all" ]; then
  run_split train
  run_split test
else
  run_split "${SPLIT}"
fi
