#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=72:00:00
#PBS -W group_list=qgah50055
#PBS -N nepa3d_preprocess_modelnet40
#PBS -o logs/preprocess/preprocess_modelnet40.out
#PBS -e logs/preprocess/preprocess_modelnet40.err

set -euo pipefail

# Resolve repository root.
if [[ -n "${WORKDIR:-}" ]]; then
  cd "${WORKDIR}"
elif [[ -n "${REPO_ROOT:-}" ]]; then
  cd "${REPO_ROOT}"
elif [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}/3D-NEPA" ]]; then
  cd "${PBS_O_WORKDIR}/3D-NEPA"
elif [[ -n "${PBS_O_WORKDIR:-}" && -f "${PBS_O_WORKDIR}/nepa3d/data/preprocess_modelnet40.py" ]]; then
  cd "${PBS_O_WORKDIR}"
else
  cd /groups/qgah50055/ide/VGI/3D-NEPA
fi

mkdir -p logs/preprocess

source /etc/profile.d/modules.sh 2>/dev/null || true
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# Activate Python virtual environment
if [[ -f ".venv/bin/activate" ]]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELNET_ROOT="${MODELNET_ROOT:-data/ModelNet40}"
OUT_ROOT="${OUT_ROOT:-data/modelnet40_cache_v2}"
SPLIT="${SPLIT:-all}"  # train, test, or all
MESH_GLOB="${MESH_GLOB:-}"
OVERWRITE="${OVERWRITE:-0}"

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

WORKERS="${N_WORKERS:-32}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-8}"

run_split() {
  split="$1"
  extra_mesh_args=()
  if [[ -n "${MESH_GLOB}" ]]; then
    extra_mesh_args+=(--mesh_glob "${MESH_GLOB}")
  fi
  extra_overwrite=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    extra_overwrite+=(--overwrite)
  fi
  "${PYTHON_BIN}" -u -m nepa3d.data.preprocess_modelnet40 \
    --modelnet_root "${MODELNET_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --split "${split}" \
    "${extra_mesh_args[@]}" \
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
    $( [[ "${NO_UDF}" == "1" ]] && echo "--no_udf" ) \
    $( [[ "${NO_PC_RAYS}" == "1" ]] && echo "--no_pc_rays" ) \
    "${extra_overwrite[@]}" \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --chunk_size "${CHUNK_SIZE}" \
    --max_tasks_per_child "${MAX_TASKS_PER_CHILD}"
}

echo "[info] date=$(date -Is) host=$(hostname)"
echo "[info] modelnet_root=${MODELNET_ROOT} out_root=${OUT_ROOT} split=${SPLIT}"
echo "[info] workers=${WORKERS} chunk_size=${CHUNK_SIZE} max_tasks_per_child=${MAX_TASKS_PER_CHILD} overwrite=${OVERWRITE}"
echo "[info] pc_points=${PC_POINTS} pt_pool=${PT_POOL} ray_pool=${RAY_POOL}"

if [[ "${SPLIT}" == "all" ]]; then
  run_split train
  run_split test
else
  run_split "${SPLIT}"
fi
