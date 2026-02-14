#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -P gag51403
#PBS -N nepa3d_cpac_udf
#PBS -o nepa3d_cpac_udf.out
#PBS -e nepa3d_cpac_udf.err

set -eu

. /etc/profile.d/modules.sh
cd /groups/gag51403/ide/3D-NEPA

if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SPLIT="${SPLIT:-eval}"
CKPT="${CKPT:?set CKPT=...}"
CONTEXT_BACKEND="${CONTEXT_BACKEND:-pointcloud_noray}"
N_CONTEXT="${N_CONTEXT:-256}"
N_QUERY="${N_QUERY:-256}"
HEAD_TRAIN_RATIO="${HEAD_TRAIN_RATIO:-0.2}"
HEAD_TRAIN_N="${HEAD_TRAIN_N:-0}"
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-udfgrid}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-0}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
TAU="${TAU:-0.03}"
SEED="${SEED:-0}"
MAX_SHAPES="${MAX_SHAPES:-0}"
OUT_JSON="${OUT_JSON:-results/cpac_${CONTEXT_BACKEND}_to_udf.json}"

"${PYTHON_BIN}" -m nepa3d.analysis.completion_cpac_udf \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --ckpt "${CKPT}" \
  --context_backend "${CONTEXT_BACKEND}" \
  --n_context "${N_CONTEXT}" \
  --n_query "${N_QUERY}" \
  --head_train_ratio "${HEAD_TRAIN_RATIO}" \
  --head_train_n "${HEAD_TRAIN_N}" \
  --head_train_split "${HEAD_TRAIN_SPLIT}" \
  --head_train_backend "${HEAD_TRAIN_BACKEND}" \
  --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
  --ridge_lambda "${RIDGE_LAMBDA}" \
  --tau "${TAU}" \
  --seed "${SEED}" \
  --max_shapes "${MAX_SHAPES}" \
  --out_json "${OUT_JSON}"