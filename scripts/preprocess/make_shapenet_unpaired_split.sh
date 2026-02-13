#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=4
#PBS -l walltime=01:00:00
#PBS -P gag51403
#PBS -N shapenet_unpaired_split
#PBS -o shapenet_unpaired_split.out
#PBS -e shapenet_unpaired_split.err

set -eu

. /etc/profile.d/modules.sh
cd /groups/gag51403/ide/3D-NEPA

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v0}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUT_JSON="${OUT_JSON:-data/shapenet_unpaired_splits_v1.json}"
SEED="${SEED:-0}"
RATIOS="${RATIOS:-0.34 0.33 0.33}"

"${PYTHON_BIN}" -m nepa3d.data.shapenet_unpaired_split \
  --cache_root "${CACHE_ROOT}" \
  --train_split "${TRAIN_SPLIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --out_json "${OUT_JSON}" \
  --seed "${SEED}" \
  --ratios ${RATIOS}
