#!/bin/bash
#PBS -q abciq
#PBS -l select=1:ncpus=32:mem=100gb:ngpus=0
#PBS -l walltime=24:00:00
#PBS -l rt_QC=1
#PBS -l spot_rt_QC=1
#PBS -W group_list=qgah50055
#PBS -N nepa3d_preprocess_scan
#PBS -o logs/preprocess/preprocess_scanobjectnn.out
#PBS -e logs/preprocess/preprocess_scanobjectnn.err

set -euo pipefail

# Resolve repository root. Priority:
#  1) explicit REPO_ROOT
#  2) PBS submit dir if it contains 3D-NEPA
#  3) PBS submit dir itself if it is repo root
#  4) this environment's default checkout path
if [ -n "${REPO_ROOT:-}" ]; then
  cd "${REPO_ROOT}"
elif [ -n "${PBS_O_WORKDIR:-}" ] && [ -d "${PBS_O_WORKDIR}/3D-NEPA" ]; then
  cd "${PBS_O_WORKDIR}/3D-NEPA"
elif [ -n "${PBS_O_WORKDIR:-}" ] && [ -f "${PBS_O_WORKDIR}/nepa3d/data/preprocess_scanobjectnn.py" ]; then
  cd "${PBS_O_WORKDIR}"
else
  cd /groups/qgah50055/ide/VGI/3D-NEPA
fi

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
SCAN_ROOT="${SCAN_ROOT:-data/ScanObjectNN/h5_files/main_split}"
OUT_ROOT="${OUT_ROOT:-data/scanobjectnn_main_split_v2}"
SPLIT="${SPLIT:-all}"  # train|test|all
PT_POOL="${PT_POOL:-4000}"
RAY_POOL="${RAY_POOL:-256}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
ALLOW_DUPLICATE_STEMS="${ALLOW_DUPLICATE_STEMS:-0}"

extra_overwrite=()
if [ "${OVERWRITE}" = "1" ]; then
  extra_overwrite+=(--overwrite)
fi

extra_dup=()
if [ "${ALLOW_DUPLICATE_STEMS}" = "1" ]; then
  extra_dup+=(--allow_duplicate_stems)
fi

"${PYTHON_BIN}" -u -m nepa3d.data.preprocess_scanobjectnn \
  --scan_root "${SCAN_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --split "${SPLIT}" \
  --pt_pool "${PT_POOL}" \
  --ray_pool "${RAY_POOL}" \
  --pt_surface_ratio "${PT_SURFACE_RATIO}" \
  --pt_surface_sigma "${PT_SURFACE_SIGMA}" \
  --seed "${SEED}" \
  --workers "${WORKERS}" \
  "${extra_overwrite[@]}" \
  "${extra_dup[@]}"
