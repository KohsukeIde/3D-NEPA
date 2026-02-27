#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -N pointmae_scan_test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTMAE_ROOT="${POINTMAE_ROOT:-${WORKDIR}/Point-MAE}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

VARIANT="${VARIANT:-pb_t50_rs}"   # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_test_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
CFG_PROFILE="${CFG_PROFILE:-standard}"  # standard|sanity
CKPT_PATH="${CKPT_PATH:-}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae_scratch_tests}"
if [[ "${LOG_ROOT}" != /* ]]; then
  LOG_ROOT="${WORKDIR}/${LOG_ROOT}"
fi

if [[ -z "${CKPT_PATH}" ]]; then
  echo "[error] CKPT_PATH is required"
  exit 2
fi
if [[ "${CKPT_PATH}" != /* ]]; then
  CKPT_PATH="${WORKDIR}/${CKPT_PATH}"
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] CKPT_PATH not found: ${CKPT_PATH}"
  exit 2
fi

if [[ ! -d "${POINTMAE_ROOT}" ]]; then
  echo "[error] Point-MAE root not found: ${POINTMAE_ROOT}"
  exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

# Point-MAE expected ScanObjectNN layout.
mkdir -p "${POINTMAE_ROOT}/data/ScanObjectNN"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split_nobg" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"

case "${CFG_PROFILE}" in
  standard) CFG_SUFFIX="" ;;
  sanity) CFG_SUFFIX="_sanity" ;;
  *)
    echo "[error] unsupported CFG_PROFILE=${CFG_PROFILE} (standard|sanity)"
    exit 2
    ;;
esac

case "${VARIANT}" in
  pb_t50_rs|hardest) CONFIG_PATH="cfgs/finetune_scan_hardest${CFG_SUFFIX}.yaml" ;;
  obj_bg) CONFIG_PATH="cfgs/finetune_scan_objbg${CFG_SUFFIX}.yaml" ;;
  obj_only) CONFIG_PATH="cfgs/finetune_scan_objonly${CFG_SUFFIX}.yaml" ;;
  *)
    echo "[error] unsupported VARIANT=${VARIANT} (pb_t50_rs|obj_bg|obj_only)"
    exit 2
    ;;
esac

OUT_LOG="${LOG_ROOT}/${RUN_TAG}.log"

echo "=== POINT-MAE TEST (from ckpt) ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "variant=${VARIANT}"
echo "run_tag=${RUN_TAG}"
echo "config=${POINTMAE_ROOT}/${CONFIG_PATH}"
echo "ckpt=${CKPT_PATH}"
echo "log=${OUT_LOG}"
echo

cd "${POINTMAE_ROOT}"
python main.py \
  --test \
  --config "${CONFIG_PATH}" \
  --exp_name "${RUN_TAG}" \
  --ckpts "${CKPT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_LOG}"

TEST_LINE="$(grep -n "\\[TEST\\] acc =" "${OUT_LOG}" | tail -n 1 || true)"
if [[ -n "${TEST_LINE}" ]]; then
  echo "[summary] ${TEST_LINE}"
else
  echo "[warn] could not find '[TEST] acc' in ${OUT_LOG}"
fi
