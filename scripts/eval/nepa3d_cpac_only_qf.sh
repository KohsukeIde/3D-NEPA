#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=qgah50055
#PBS -N nepa3d_cpac_only

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

RUN_TAG="${RUN_TAG:?set RUN_TAG}"
CKPT="${CKPT:?set CKPT}"

UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs/eval/abcd_cpac_only}"

CPAC_SPLIT="${CPAC_SPLIT:-eval}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_HEAD_TRAIN_SPLIT="${CPAC_HEAD_TRAIN_SPLIT:-train_udf}"
CPAC_HEAD_TRAIN_BACKEND="${CPAC_HEAD_TRAIN_BACKEND:-udfgrid}"
CPAC_HEAD_TRAIN_RATIO="${CPAC_HEAD_TRAIN_RATIO:-0.2}"
CPAC_RIDGE_LAMBDA="${CPAC_RIDGE_LAMBDA:-1e-3}"
CPAC_TAU="${CPAC_TAU:-0.03}"
CPAC_EVAL_SEED="${CPAC_EVAL_SEED:-0}"

cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== CPAC-ONLY JOB INFO ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "run_tag=${RUN_TAG}"
echo "ckpt=${CKPT}"
echo "python=$(which python)"
python -V || true
echo

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing ckpt: ${CKPT}"
  exit 2
fi

if [[ ! -d "${UNPAIRED_CACHE_ROOT}" ]]; then
  echo "[error] missing unpaired cache for CPAC: ${UNPAIRED_CACHE_ROOT}"
  exit 2
fi

CPAC_JSON="${RESULTS_ROOT}/cpac_abcd_1024_${RUN_TAG}.json"
CPAC_LOG="${LOG_ROOT}/${RUN_TAG}_cpac.log"

echo "=== CPAC ==="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root "${UNPAIRED_CACHE_ROOT}" \
  --split "${CPAC_SPLIT}" \
  --ckpt "${CKPT}" \
  --context_backend pointcloud_noray \
  --n_context "${CPAC_N_CONTEXT}" \
  --n_query "${CPAC_N_QUERY}" \
  --max_shapes "${CPAC_MAX_SHAPES}" \
  --head_train_split "${CPAC_HEAD_TRAIN_SPLIT}" \
  --head_train_backend "${CPAC_HEAD_TRAIN_BACKEND}" \
  --head_train_ratio "${CPAC_HEAD_TRAIN_RATIO}" \
  --ridge_lambda "${CPAC_RIDGE_LAMBDA}" \
  --tau "${CPAC_TAU}" \
  --eval_seed "${CPAC_EVAL_SEED}" \
  --disjoint_context_query 1 \
  --context_mode_test normal \
  --rep_source h \
  --query_source pool \
  --out_json "${CPAC_JSON}" \
  2>&1 | tee "${CPAC_LOG}"

echo "=== DONE ==="
echo "cpac_json=${CPAC_JSON}"
