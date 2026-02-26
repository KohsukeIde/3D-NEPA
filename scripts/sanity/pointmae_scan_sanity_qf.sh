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
USE_NEPA_CACHE="${USE_NEPA_CACHE:-0}"  # 1: build Point-MAE h5 from NEPA NPZ cache and use it
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"

VARIANT="${VARIANT:-pb_t50_rs}"  # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_sanity_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae}"
mkdir -p "${LOG_ROOT}"

to_abs_path() {
  local p="$1"
  if [[ "${p}" = /* ]]; then
    echo "${p}"
  else
    echo "${WORKDIR}/${p}"
  fi
}

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
python -m pip install -q easydict tensorboardX termcolor timm==0.4.5 transforms3d matplotlib torchvision h5py

# Point-MAE expected ScanObjectNN layout.
mkdir -p "${POINTMAE_ROOT}/data/ScanObjectNN"
if [[ "${USE_NEPA_CACHE}" == "1" ]]; then
  case "${VARIANT}" in
    pb_t50_rs)
      NEPA_CACHE_ROOT="${NEPA_CACHE_ROOT:-${WORKDIR}/data/scanobjectnn_pb_t50_rs_v3_nonorm}"
      ;;
    obj_bg)
      NEPA_CACHE_ROOT="${NEPA_CACHE_ROOT:-${WORKDIR}/data/scanobjectnn_obj_bg_v3_nonorm}"
      ;;
    obj_only)
      NEPA_CACHE_ROOT="${NEPA_CACHE_ROOT:-${WORKDIR}/data/scanobjectnn_obj_only_v3_nonorm}"
      ;;
    *)
      echo "[error] unsupported VARIANT=${VARIANT} (pb_t50_rs|obj_bg|obj_only)"
      exit 2
      ;;
  esac

  NEPA_CACHE_ROOT="$(to_abs_path "${NEPA_CACHE_ROOT}")"
  if [[ "${NEPA_CACHE_ROOT}" == *"scanobjectnn_"*"_v2" ]] && [[ "${ALLOW_SCAN_UNISCALE_V2}" != "1" ]]; then
    echo "[error] NEPA_CACHE_ROOT=${NEPA_CACHE_ROOT} is a uniscale v2 cache and is disallowed by policy."
    echo "        Use scanobjectnn_*_v3_nonorm, or set ALLOW_SCAN_UNISCALE_V2=1 for intentional legacy reruns."
    exit 2
  fi

  CACHE_H5_ROOT_BASE="${CACHE_H5_ROOT_BASE:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache}"
  CACHE_H5_ROOT_BASE="$(to_abs_path "${CACHE_H5_ROOT_BASE}")"
  CACHE_H5_OVERWRITE="${CACHE_H5_OVERWRITE:-0}"
  CACHE_H5_ROOT="${CACHE_H5_ROOT_BASE}/${VARIANT}"
  mkdir -p "${CACHE_H5_ROOT_BASE}"
  python "${WORKDIR}/scripts/sanity/build_scanobjectnn_h5_from_nepa_cache.py" \
    --cache_root "${NEPA_CACHE_ROOT}" \
    --variant "${VARIANT}" \
    --out_root "${CACHE_H5_ROOT}" \
    --overwrite "${CACHE_H5_OVERWRITE}"

  if [[ "${VARIANT}" == "obj_only" ]]; then
    ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
    ln -sfn "${CACHE_H5_ROOT}" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"
  else
    ln -sfn "${CACHE_H5_ROOT}" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
    ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split_nobg" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"
  fi
else
  ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
  ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split_nobg" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"
fi

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
echo "use_nepa_cache=${USE_NEPA_CACHE}"
if [[ "${USE_NEPA_CACHE}" == "1" ]]; then
  echo "nepa_cache_root=${NEPA_CACHE_ROOT}"
  echo "cache_h5_root=${CACHE_H5_ROOT}"
fi
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
