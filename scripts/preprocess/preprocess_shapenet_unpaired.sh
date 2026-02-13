#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=8
#PBS -l walltime=02:00:00
#PBS -P gag51403
#PBS -N shapenet_unpaired_materialize
#PBS -o shapenet_unpaired_materialize.out
#PBS -e shapenet_unpaired_materialize.err

set -eu

. /etc/profile.d/modules.sh
cd /groups/gag51403/ide/3D-NEPA

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-data/shapenet_cache_v0}"
SPLIT_JSON="${SPLIT_JSON:-data/shapenet_unpaired_splits_v1.json}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_unpaired_cache_v1}"
SPLITS="${SPLITS:-train_mesh train_pc train_udf eval}"
LINK_MODE="${LINK_MODE:-symlink}"  # symlink|hardlink|copy
OVERWRITE="${OVERWRITE:-0}"

EXTRA_OVERWRITE=""
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA_OVERWRITE="--overwrite"
fi

"${PYTHON_BIN}" -m nepa3d.data.preprocess_shapenet_unpaired \
  --src_cache_root "${SRC_CACHE_ROOT}" \
  --split_json "${SPLIT_JSON}" \
  --out_root "${OUT_ROOT}" \
  --splits ${SPLITS} \
  --link_mode "${LINK_MODE}" \
  ${EXTRA_OVERWRITE}
