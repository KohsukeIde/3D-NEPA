#!/bin/sh
#PBS -q rt_HC
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l walltime=04:00:00
#PBS -P gag51403
#PBS -N nepa3d_ucpr
#PBS -o nepa3d_ucpr.out
#PBS -e nepa3d_ucpr.err

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
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_GALLERY="${EVAL_SEED_GALLERY:-${EVAL_SEED}}"
MC_K="${MC_K:-1}"
MAX_FILES="${MAX_FILES:-0}"
# Pooling: eos / mean_a / mean_zhat
POOLING="${POOLING:-eos}"
TIE_BREAK_EPS="${TIE_BREAK_EPS:-1e-6}"
# Optional ablations (0/1)
ABLATE_POINT_XYZ="${ABLATE_POINT_XYZ:-0}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-0}"
SANITY_CONSTANT_EMBED="${SANITY_CONSTANT_EMBED:-0}"
OUT_JSON="${OUT_JSON:-results/ucpr_${QUERY_BACKEND}_to_${GALLERY_BACKEND}.json}"

EXTRA_ARGS=""
if [ "${ABLATE_POINT_XYZ}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --ablate_point_xyz"
fi
if [ "${ABLATE_POINT_DIST}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --ablate_point_dist"
fi
if [ "${SANITY_CONSTANT_EMBED}" -eq 1 ]; then
  EXTRA_ARGS="${EXTRA_ARGS} --sanity_constant_embed"
fi

"${PYTHON_BIN}" -m nepa3d.analysis.retrieval_ucpr \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --ckpt "${CKPT}" \
  --query_backend "${QUERY_BACKEND}" \
  --gallery_backend "${GALLERY_BACKEND}" \
  --eval_seed "${EVAL_SEED}" \
  --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
  --mc_k "${MC_K}" \
  --max_files "${MAX_FILES}" \
  --pooling "${POOLING}" \
  --tie_break_eps "${TIE_BREAK_EPS}" \
  --out_json "${OUT_JSON}" \
  ${EXTRA_ARGS}
