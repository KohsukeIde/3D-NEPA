#!/usr/bin/env bash
set -euo pipefail

# PointGPT ScanObjectNN fine-tune from scratch. This mirrors the local DDP
# finetune wrapper but intentionally passes --scratch_model and no checkpoint.
# Defaults to DRY_RUN=1 so it is safe to inspect generated commands.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
DEFAULT_VENV_ACTIVATE="${WORKDIR}/.venv-pointgpt/bin/activate"
if [[ ! -f "${DEFAULT_VENV_ACTIVATE}" ]]; then
  DEFAULT_VENV_ACTIVATE="${WORKDIR}/.venv/bin/activate"
fi
VENV_ACTIVATE="${VENV_ACTIVATE:-${DEFAULT_VENV_ACTIVATE}}"

CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_hardest.yaml}"
EXP_NAME="${EXP_NAME:-pointgpt_scan_scratch_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29950}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_FREQ="${VAL_FREQ:-1}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-0}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
DRY_RUN="${DRY_RUN:-1}"

USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-transfer}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-posetnepa_scratch_early}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-pointgpt,scanobjectnn,scratch}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

SCAN_ROOT="${SCAN_ROOT:-${WORKDIR}/data/ScanObjectNN}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -d "${POINTGPT_DIR}" ]]; then
  echo "[error] PointGPT dir not found: ${POINTGPT_DIR}"
  exit 2
fi
if [[ ! -f "${POINTGPT_DIR}/${CONFIG_PATH}" && ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] config not found: ${CONFIG_PATH}"
  exit 2
fi

CONFIG_PATH_EXEC="${CONFIG_PATH}"
if [[ -f "${POINTGPT_DIR}/${CONFIG_PATH}" ]]; then
  CONFIG_PATH_EXEC="${POINTGPT_DIR}/${CONFIG_PATH}"
fi

CFG_STEM="$(basename "${CONFIG_PATH_EXEC%.*}")"
CFG_PARENT="$(basename "$(dirname "${CONFIG_PATH_EXEC}")")"
RESOLVED_EXPERIMENT_PATH="./experiments/${CFG_STEM}/${CFG_PARENT}/${EXP_NAME}"

echo "=== POINTGPT SCAN SCRATCH ==="
echo "date=$(date -Is)"
echo "config=${CONFIG_PATH_EXEC}"
echo "exp_name=${EXP_NAME}"
echo "resolved_experiment_path=${RESOLVED_EXPERIMENT_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "dry_run=${DRY_RUN}"

cmd=(torchrun
  --standalone
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  main.py
  --launcher pytorch
  --config "${CONFIG_PATH_EXEC}"
  --exp_name "${EXP_NAME}"
  --num_workers "${NUM_WORKERS}"
  --val_freq "${VAL_FREQ}"
  --ft_recon_weight "${FT_RECON_WEIGHT}"
  --save_last_every_epoch "${SAVE_LAST_EVERY_EPOCH}"
  --scratch_model
)

printf '[cmd] cd %q && ' "${POINTGPT_DIR}"
printf '%q ' "${cmd[@]}"
printf '%s\n' "${EXTRA_ARGS:+ ${EXTRA_ARGS}}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
  exit 2
fi
if [[ ! -f "${SCAN_ROOT}/h5_files/main_split/training_objectdataset.h5" ]]; then
  echo "[error] ScanObjectNN main_split missing under: ${SCAN_ROOT}"
  exit 2
fi
if [[ ! -f "${SCAN_ROOT}/h5_files/main_split_nobg/training_objectdataset.h5" ]]; then
  echo "[error] ScanObjectNN main_split_nobg missing under: ${SCAN_ROOT}"
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

mkdir -p "${POINTGPT_DIR}/data"
POINTGPT_SCAN_LINK="${POINTGPT_DIR}/data/ScanObjectNN"
if [[ -L "${POINTGPT_SCAN_LINK}" || ! -e "${POINTGPT_SCAN_LINK}" ]]; then
  ln -sfn "${SCAN_ROOT}" "${POINTGPT_SCAN_LINK}"
elif [[ ! -d "${POINTGPT_SCAN_LINK}" ]]; then
  echo "[error] PointGPT ScanObjectNN path exists and is not a directory/symlink: ${POINTGPT_SCAN_LINK}"
  exit 2
fi

cd "${POINTGPT_DIR}"
"${cmd[@]}" ${EXTRA_ARGS}
