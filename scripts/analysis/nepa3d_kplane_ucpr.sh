#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -P gag51403
#PBS -N nepa3d_kplane_ucpr
#PBS -o nepa3d_kplane_ucpr.out
#PBS -e nepa3d_kplane_ucpr.err

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
QUERY_BACKEND="${QUERY_BACKEND:-mesh}"
GALLERY_BACKEND="${GALLERY_BACKEND:-udfgrid}"
N_CONTEXT="${N_CONTEXT:-256}"
N_QUERY="${N_QUERY:-256}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_GALLERY="${EVAL_SEED_GALLERY:-${EVAL_SEED}}"
MC_K="${MC_K:-1}"
MAX_FILES="${MAX_FILES:-0}"
POOLING="${POOLING:-mean_query}"         # mean_query / plane_gap
ABLATE_QUERY_XYZ="${ABLATE_QUERY_XYZ:-0}"
ABLATE_CONTEXT_DIST="${ABLATE_CONTEXT_DIST:-0}"
DISJOINT_CONTEXT_QUERY="${DISJOINT_CONTEXT_QUERY:-1}"
CONTEXT_MODE_QUERY="${CONTEXT_MODE_QUERY:-normal}"       # normal / none / mismatch
CONTEXT_MODE_GALLERY="${CONTEXT_MODE_GALLERY:-normal}"   # normal / none / mismatch
MISMATCH_SHIFT_QUERY="${MISMATCH_SHIFT_QUERY:-1}"
MISMATCH_SHIFT_GALLERY="${MISMATCH_SHIFT_GALLERY:-1}"
TIE_BREAK_EPS="${TIE_BREAK_EPS:-1e-6}"
SANITY_CONSTANT_EMBED="${SANITY_CONSTANT_EMBED:-0}"
OUT_JSON="${OUT_JSON:-results/kplane_ucpr_${QUERY_BACKEND}_to_${GALLERY_BACKEND}.json}"

EXTRA_ARGS=""
if [ "${ABLATE_QUERY_XYZ}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --ablate_query_xyz"
fi
if [ "${ABLATE_CONTEXT_DIST}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --ablate_context_dist"
fi
if [ "${SANITY_CONSTANT_EMBED}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --sanity_constant_embed"
fi

"${PYTHON_BIN}" -m nepa3d.analysis.retrieval_kplane \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --ckpt "${CKPT}" \
  --query_backend "${QUERY_BACKEND}" \
  --gallery_backend "${GALLERY_BACKEND}" \
  --n_context "${N_CONTEXT}" \
  --n_query "${N_QUERY}" \
  --eval_seed "${EVAL_SEED}" \
  --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
  --mc_k "${MC_K}" \
  --max_files "${MAX_FILES}" \
  --pooling "${POOLING}" \
  --disjoint_context_query "${DISJOINT_CONTEXT_QUERY}" \
  --context_mode_query "${CONTEXT_MODE_QUERY}" \
  --context_mode_gallery "${CONTEXT_MODE_GALLERY}" \
  --mismatch_shift_query "${MISMATCH_SHIFT_QUERY}" \
  --mismatch_shift_gallery "${MISMATCH_SHIFT_GALLERY}" \
  --tie_break_eps "${TIE_BREAK_EPS}" \
  --out_json "${OUT_JSON}" \
  ${EXTRA_ARGS}
