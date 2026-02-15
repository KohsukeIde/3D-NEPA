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
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-0}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
TAU="${TAU:-0.03}"
MAX_SHAPES="${MAX_SHAPES:-800}"
# Eval seed (SEED alias supported)
EVAL_SEED="${EVAL_SEED:-0}"
SEED="${SEED:-}"
if [ -n "${SEED}" ]; then
  EVAL_SEED="${SEED}"
fi
# New CPAC controls
DISJOINT_CONTEXT_QUERY="${DISJOINT_CONTEXT_QUERY:-1}"
CONTEXT_MODE_TRAIN="${CONTEXT_MODE_TRAIN:-normal}"   # normal / none / mismatch
CONTEXT_MODE_TEST="${CONTEXT_MODE_TEST:-normal}"     # normal / none / mismatch
MISMATCH_SHIFT="${MISMATCH_SHIFT:-1}"
REP_SOURCE="${REP_SOURCE:-h}"                        # h / zhat
OUT_JSON="${OUT_JSON:-results/cpac_${CONTEXT_BACKEND}_to_udf.json}"

ARGS="--cache_root ${CACHE_ROOT} --split ${SPLIT} --ckpt ${CKPT} \
--context_backend ${CONTEXT_BACKEND} \
--n_context ${N_CONTEXT} --n_query ${N_QUERY} \
--max_shapes ${MAX_SHAPES} \
--head_train_ratio ${HEAD_TRAIN_RATIO} \
--head_train_n ${HEAD_TRAIN_N} \
--head_train_max_shapes ${HEAD_TRAIN_MAX_SHAPES} \
--ridge_lambda ${RIDGE_LAMBDA} --tau ${TAU} \
--eval_seed ${EVAL_SEED} \
--disjoint_context_query ${DISJOINT_CONTEXT_QUERY} \
--context_mode_train ${CONTEXT_MODE_TRAIN} \
--context_mode_test ${CONTEXT_MODE_TEST} \
--mismatch_shift ${MISMATCH_SHIFT} \
--rep_source ${REP_SOURCE} \
--out_json ${OUT_JSON}"

if [ -n "${HEAD_TRAIN_SPLIT}" ]; then
  ARGS="${ARGS} --head_train_split ${HEAD_TRAIN_SPLIT}"
fi
if [ -n "${HEAD_TRAIN_BACKEND}" ]; then
  ARGS="${ARGS} --head_train_backend ${HEAD_TRAIN_BACKEND}"
fi

"${PYTHON_BIN}" -m nepa3d.analysis.completion_cpac_udf ${ARGS}
