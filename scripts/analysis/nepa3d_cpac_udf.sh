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
MAX_LEN="${MAX_LEN:--1}"   # override model max_len (pos-emb length); -1 = use checkpoint
HEAD_TRAIN_RATIO="${HEAD_TRAIN_RATIO:-0.2}"
HEAD_TRAIN_N="${HEAD_TRAIN_N:-0}"
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-0}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
RIDGE_LIPSCHITZ_LAMBDA="${RIDGE_LIPSCHITZ_LAMBDA:-0}"
RIDGE_LIPSCHITZ_PAIRS="${RIDGE_LIPSCHITZ_PAIRS:-2048}"
RIDGE_LIPSCHITZ_STEPS="${RIDGE_LIPSCHITZ_STEPS:-200}"
RIDGE_LIPSCHITZ_LR="${RIDGE_LIPSCHITZ_LR:-1e-2}"
RIDGE_LIPSCHITZ_BATCH="${RIDGE_LIPSCHITZ_BATCH:-8192}"
RIDGE_LIPSCHITZ_MAX_POINTS="${RIDGE_LIPSCHITZ_MAX_POINTS:-200000}"
RIDGE_LIPSCHITZ_SEED="${RIDGE_LIPSCHITZ_SEED:-0}"
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
QUERY_SOURCE="${QUERY_SOURCE:-pool}"                 # pool / grid
QUERY_POOL_FRAC="${QUERY_POOL_FRAC:-0.5}"           # used when QUERY_SOURCE=hybrid
GRID_SAMPLE_MODE="${GRID_SAMPLE_MODE:-uniform}"      # uniform / near_surface / stratified
GRID_NEAR_TAU="${GRID_NEAR_TAU:-0.05}"
GRID_NEAR_FRAC="${GRID_NEAR_FRAC:-0.7}"
GRID_RES_SCHEDULE="${GRID_RES_SCHEDULE:-16,32,64}"  # used when GRID_SAMPLE_MODE=coarse_to_fine
GRID_C2F_EXPAND="${GRID_C2F_EXPAND:-1}"             # used when GRID_SAMPLE_MODE=coarse_to_fine
GRID_C2F_STAGE_WEIGHTS="${GRID_C2F_STAGE_WEIGHTS:-auto}" # optional, comma-separated; "auto" = default weights
TARGET_TRANSFORM="${TARGET_TRANSFORM:-none}"         # none / trunc / log1p
TARGET_TRUNC_MAX="${TARGET_TRUNC_MAX:-0.1}"
TARGET_LOG_SCALE="${TARGET_LOG_SCALE:-0.03}"
REPORT_NEAR_TAU="${REPORT_NEAR_TAU:-0.05}"
BASELINE="${BASELINE:-none}"                         # none / nn_copy
BASELINE_ONLY="${BASELINE_ONLY:-0}"                  # 0 / 1
OUT_JSON="${OUT_JSON:-results/cpac_${CONTEXT_BACKEND}_to_udf.json}"

ARGS="--cache_root ${CACHE_ROOT} --split ${SPLIT} --ckpt ${CKPT} \
--max_len ${MAX_LEN} \
--context_backend ${CONTEXT_BACKEND} \
--n_context ${N_CONTEXT} --n_query ${N_QUERY} \
--max_shapes ${MAX_SHAPES} \
--head_train_ratio ${HEAD_TRAIN_RATIO} \
--head_train_n ${HEAD_TRAIN_N} \
--head_train_max_shapes ${HEAD_TRAIN_MAX_SHAPES} \
--ridge_lambda ${RIDGE_LAMBDA} --tau ${TAU} \
--ridge_lipschitz_lambda ${RIDGE_LIPSCHITZ_LAMBDA} \
--ridge_lipschitz_pairs ${RIDGE_LIPSCHITZ_PAIRS} \
--ridge_lipschitz_steps ${RIDGE_LIPSCHITZ_STEPS} \
--ridge_lipschitz_lr ${RIDGE_LIPSCHITZ_LR} \
--ridge_lipschitz_batch ${RIDGE_LIPSCHITZ_BATCH} \
--ridge_lipschitz_max_points ${RIDGE_LIPSCHITZ_MAX_POINTS} \
--ridge_lipschitz_seed ${RIDGE_LIPSCHITZ_SEED} \
--eval_seed ${EVAL_SEED} \
--disjoint_context_query ${DISJOINT_CONTEXT_QUERY} \
--context_mode_train ${CONTEXT_MODE_TRAIN} \
--context_mode_test ${CONTEXT_MODE_TEST} \
--mismatch_shift ${MISMATCH_SHIFT} \
--rep_source ${REP_SOURCE} \
--query_source ${QUERY_SOURCE} \
--query_pool_frac ${QUERY_POOL_FRAC} \
--grid_sample_mode ${GRID_SAMPLE_MODE} \
--grid_near_tau ${GRID_NEAR_TAU} \
--grid_near_frac ${GRID_NEAR_FRAC} \
--grid_res_schedule ${GRID_RES_SCHEDULE} \
--grid_c2f_expand ${GRID_C2F_EXPAND} \
--grid_c2f_stage_weights ${GRID_C2F_STAGE_WEIGHTS} \
--target_transform ${TARGET_TRANSFORM} \
--target_trunc_max ${TARGET_TRUNC_MAX} \
--target_log_scale ${TARGET_LOG_SCALE} \
--report_near_tau ${REPORT_NEAR_TAU} \
--baseline ${BASELINE} \
--baseline_only ${BASELINE_ONLY} \
--out_json ${OUT_JSON}"

if [ -n "${HEAD_TRAIN_SPLIT}" ]; then
  ARGS="${ARGS} --head_train_split ${HEAD_TRAIN_SPLIT}"
fi
if [ -n "${HEAD_TRAIN_BACKEND}" ]; then
  ARGS="${ARGS} --head_train_backend ${HEAD_TRAIN_BACKEND}"
fi

"${PYTHON_BIN}" -m nepa3d.analysis.completion_cpac_udf ${ARGS}
