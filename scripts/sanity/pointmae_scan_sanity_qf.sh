#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N pointmae_scan_sanity

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTMAE_ROOT="${POINTMAE_ROOT:-${WORKDIR}/Point-MAE}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

VARIANT="${VARIANT:-pb_t50_rs}"  # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_sanity_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae}"
mkdir -p "${LOG_ROOT}"

if [[ ! -d "${POINTMAE_ROOT}" ]]; then
  echo "[error] Point-MAE root not found: ${POINTMAE_ROOT}"
  exit 2
fi

cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

python -V
python -m pip install -q easydict tensorboardX termcolor timm==0.4.5 transforms3d matplotlib torchvision

# Point-MAE expected ScanObjectNN layout.
mkdir -p "${POINTMAE_ROOT}/data/ScanObjectNN"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split_nobg" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"

case "${VARIANT}" in
  pb_t50_rs|hardest)
    CONFIG_PATH="cfgs/finetune_scan_hardest_sanity.yaml"
    CKPT_URL="https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_hardest.pth"
    CKPT_NAME="scan_hardest.pth"
    ;;
  obj_bg)
    CONFIG_PATH="cfgs/finetune_scan_objbg_sanity.yaml"
    CKPT_URL="https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objbg.pth"
    CKPT_NAME="scan_objbg.pth"
    ;;
  obj_only)
    CONFIG_PATH="cfgs/finetune_scan_objonly_sanity.yaml"
    CKPT_URL="https://github.com/Pang-Yatian/Point-MAE/releases/download/main/scan_objonly.pth"
    CKPT_NAME="scan_objonly.pth"
    ;;
  *)
    echo "[error] unsupported VARIANT=${VARIANT} (pb_t50_rs|obj_bg|obj_only)"
    exit 2
    ;;
esac

CKPT_DIR="${POINTMAE_ROOT}/pretrained"
CKPT_PATH="${CKPT_DIR}/${CKPT_NAME}"
mkdir -p "${CKPT_DIR}"
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[download] ${CKPT_URL} -> ${CKPT_PATH}"
  curl -L --fail -o "${CKPT_PATH}" "${CKPT_URL}"
fi

OUT_LOG="${LOG_ROOT}/${RUN_TAG}.log"

echo "=== POINT-MAE SANITY ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "variant=${VARIANT}"
echo "run_tag=${RUN_TAG}"
echo "config=${CONFIG_PATH}"
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

TEST_LINE="$(grep -n "\[TEST\] acc =" "${OUT_LOG}" | tail -n 1 || true)"
if [[ -n "${TEST_LINE}" ]]; then
  echo "[summary] ${TEST_LINE}"
else
  echo "[warn] could not find '[TEST] acc' in ${OUT_LOG}"
fi
