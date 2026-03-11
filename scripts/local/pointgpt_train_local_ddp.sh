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

CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-B/pretrain_nepa_cosine.yaml}"
EXP_NAME="${EXP_NAME:-pointgpt_local_ddp_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_local_ddp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-pointgpt,local,ddp2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}"
  exit 2
fi
if [[ ! -f "${POINTGPT_DIR}/${CONFIG_PATH}" ]]; then
  echo "[error] config not found: ${POINTGPT_DIR}/${CONFIG_PATH}"
  exit 2
fi
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torchstat") else 1)
PY
then
  python -m pip install -q torchstat
fi
if [[ "${USE_WANDB}" == "1" ]] && ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("wandb") else 1)
PY
then
  python -m pip install -q wandb
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export USE_WANDB
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_GROUP
export WANDB_RUN_NAME
export WANDB_TAGS
export WANDB_MODE
export WANDB_LOG_EVERY
export WANDB_DIR

cd "${POINTGPT_DIR}"

echo "=== POINTGPT LOCAL DDP ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "config=${CONFIG_PATH}"
echo "exp_name=${EXP_NAME}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "use_wandb=${USE_WANDB} project=${WANDB_PROJECT} group=${WANDB_GROUP} run=${WANDB_RUN_NAME} mode=${WANDB_MODE}"
echo

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --launcher pytorch \
  --config "${CONFIG_PATH}" \
  --exp_name "${EXP_NAME}" \
  --num_workers "${NUM_WORKERS}" \
  ${EXTRA_ARGS}
