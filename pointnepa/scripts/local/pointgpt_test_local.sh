#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
DEFAULT_VENV_ACTIVATE="${WORKDIR}/.venv-pointgpt/bin/activate"
if [[ ! -f "${DEFAULT_VENV_ACTIVATE}" ]]; then
  DEFAULT_VENV_ACTIVATE="${WORKDIR}/.venv/bin/activate"
fi
VENV_ACTIVATE="${VENV_ACTIVATE:-${DEFAULT_VENV_ACTIVATE}}"

CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
EXP_NAME="${EXP_NAME:-pointgpt_test_local_$(date +%Y%m%d_%H%M%S)}"
CKPT_PATH="${CKPT_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TEST_VOTE_TIMES="${TEST_VOTE_TIMES:-0}"
SCAN_ROOT="${SCAN_ROOT:-${WORKDIR}/data/ScanObjectNN}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}"
  exit 2
fi
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" && ! -f "${POINTGPT_DIR}/${CONFIG_PATH}" ]]; then
  echo "[error] config not found: ${CONFIG_PATH}"
  exit 2
fi
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[error] CKPT_PATH is required"
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}"
  exit 2
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export TEST_VOTE_TIMES

mkdir -p "${POINTGPT_DIR}/data"
POINTGPT_SCAN_LINK="${POINTGPT_DIR}/data/ScanObjectNN"
if [[ -L "${POINTGPT_SCAN_LINK}" || ! -e "${POINTGPT_SCAN_LINK}" ]]; then
  ln -sfn "${SCAN_ROOT}" "${POINTGPT_SCAN_LINK}"
elif [[ ! -d "${POINTGPT_SCAN_LINK}" ]]; then
  echo "[error] PointGPT ScanObjectNN path exists and is not a directory/symlink: ${POINTGPT_SCAN_LINK}"
  exit 2
fi

CONFIG_PATH_EXEC="${CONFIG_PATH}"
if [[ -f "${POINTGPT_DIR}/${CONFIG_PATH}" ]]; then
  CONFIG_PATH_EXEC="${POINTGPT_DIR}/${CONFIG_PATH}"
fi

cd "${POINTGPT_DIR}"

echo "=== POINTGPT LOCAL TEST ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "config=${CONFIG_PATH_EXEC}"
echo "exp_name=${EXP_NAME}"
echo "ckpt_path=${CKPT_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "test_vote_times=${TEST_VOTE_TIMES}"
echo

python main.py \
  --launcher none \
  --test \
  --config "${CONFIG_PATH_EXEC}" \
  --exp_name "${EXP_NAME}" \
  --num_workers "${NUM_WORKERS}" \
  --ckpts "${CKPT_PATH}" \
  --test_vote_times "${TEST_VOTE_TIMES}" \
  ${EXTRA_ARGS}
