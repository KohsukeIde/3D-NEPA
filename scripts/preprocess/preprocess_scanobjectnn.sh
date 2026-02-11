#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_preprocess_scan
#PBS -o nepa3d_preprocess_scan.out
#PBS -e nepa3d_preprocess_scan.err

set -euo pipefail

. /etc/profile.d/modules.sh

cd /groups/gag51403/ide/3D-NEPA
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
SCAN_ROOT="${SCAN_ROOT:-data/ScanObjectNN/h5_files}"
OUT_ROOT="${OUT_ROOT:-data/scanobjectnn_cache_v1}"
SPLIT="${SPLIT:-all}"  # train|test|all
PT_POOL="${PT_POOL:-2000}"
RAY_POOL="${RAY_POOL:-1000}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
SEED="${SEED:-0}"
OVERWRITE="${OVERWRITE:-0}"

EXTRA_OVERWRITE=""
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA_OVERWRITE="--overwrite"
fi

"${PYTHON_BIN}" nepa3d/data/preprocess_scanobjectnn.py \
  --scan_root "${SCAN_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --split "${SPLIT}" \
  --pt_pool "${PT_POOL}" \
  --ray_pool "${RAY_POOL}" \
  --pt_surface_ratio "${PT_SURFACE_RATIO}" \
  --pt_surface_sigma "${PT_SURFACE_SIGMA}" \
  --seed "${SEED}" \
  ${EXTRA_OVERWRITE}
