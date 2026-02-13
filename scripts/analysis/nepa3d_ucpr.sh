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
MC_K="${MC_K:-1}"
MAX_FILES="${MAX_FILES:-0}"
OUT_JSON="${OUT_JSON:-results/ucpr_${QUERY_BACKEND}_to_${GALLERY_BACKEND}.json}"

"${PYTHON_BIN}" -m nepa3d.analysis.retrieval_ucpr \
  --cache_root "${CACHE_ROOT}" \
  --split "${SPLIT}" \
  --ckpt "${CKPT}" \
  --query_backend "${QUERY_BACKEND}" \
  --gallery_backend "${GALLERY_BACKEND}" \
  --eval_seed "${EVAL_SEED}" \
  --mc_k "${MC_K}" \
  --max_files "${MAX_FILES}" \
  --out_json "${OUT_JSON}"
