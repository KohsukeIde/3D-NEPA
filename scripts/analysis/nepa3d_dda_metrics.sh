#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=16:mem=64gb
#PBS -l walltime=04:00:00
#PBS -P gag51403
#PBS -N nepa3d_dda_metrics
#PBS -o nepa3d_dda_metrics.out
#PBS -e nepa3d_dda_metrics.err

set -eu

. /etc/profile.d/modules.sh

cd /groups/gag51403/ide/3D-NEPA
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v1}"
SPLIT="${SPLIT:-test}"
RAY_SUBSAMPLE="${RAY_SUBSAMPLE:-0}"
MAX_FILES="${MAX_FILES:-0}"
RESERVOIR="${RESERVOIR:-200000}"
SEED="${SEED:-0}"
OUT_CSV="${OUT_CSV:-results/dda_metrics_${SPLIT}.csv}"
PLOT_DIR="${PLOT_DIR:-results/dda_figs_${SPLIT}}"

"${PYTHON_BIN}" -m nepa3d.analysis.dda_metrics \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --ray_subsample "${RAY_SUBSAMPLE}" \
  --max_files "${MAX_FILES}" \
  --reservoir "${RESERVOIR}" \
  --seed "${SEED}" \
  --out_csv "${OUT_CSV}" \
  --plot_dir "${PLOT_DIR}"
