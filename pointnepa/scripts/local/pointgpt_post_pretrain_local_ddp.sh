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

CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-B/post_pretrain.yaml}"
EXP_NAME="${EXP_NAME:-pointgpt_post_pretrain_local_ddp_$(date +%Y%m%d_%H%M%S)}"
CKPT_PATH="${CKPT_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29529}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-postpretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_local_ddp_postpretrain}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-pointgpt,postpretrain,local,ddp2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"
RUNTIME_META_PATH="${RUNTIME_META_PATH:-}"

HYBRID_ROOT="${HYBRID_ROOT:-${POINTGPT_DIR}/data/HybridDatasets}"
DATA_ROOT="${DATA_ROOT:-${HYBRID_ROOT}/post_pretrain}"
PC_PATH="${PC_PATH:-${HYBRID_ROOT}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CHECK_SAMPLES="${CHECK_SAMPLES:-32}"
LHY_CHECKER="${LHY_CHECKER:-${WORKDIR}/scripts/local/pointgpt_labeledhybrid_status.sh}"

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
if [[ -z "${CKPT_PATH}" ]]; then
  echo "[error] CKPT_PATH is required"
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}"
  exit 2
fi
if [[ ! -x "${LHY_CHECKER}" ]]; then
  echo "[error] labeledhybrid checker missing or not executable: ${LHY_CHECKER}"
  exit 2
fi

POINTGPT_DIR="${POINTGPT_DIR}" \
HYBRID_ROOT="${HYBRID_ROOT}" \
DATA_ROOT="${DATA_ROOT}" \
PC_PATH="${PC_PATH}" \
CHECK_SAMPLES="${CHECK_SAMPLES}" \
"${LHY_CHECKER}" >/dev/null

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
POINTGPT_HYBRID_LINK="${POINTGPT_DIR}/data/HybridDatasets"
if [[ -L "${POINTGPT_HYBRID_LINK}" || ! -e "${POINTGPT_HYBRID_LINK}" ]]; then
  ln -sfn "${HYBRID_ROOT}" "${POINTGPT_HYBRID_LINK}"
elif [[ ! -d "${POINTGPT_HYBRID_LINK}" ]]; then
  echo "[error] PointGPT HybridDatasets path exists and is not a directory/symlink: ${POINTGPT_HYBRID_LINK}"
  exit 2
fi

cd "${POINTGPT_DIR}"

CFG_STEM="$(basename "${CONFIG_PATH%.*}")"
CFG_PARENT="$(basename "$(dirname "${CONFIG_PATH}")")"
RESOLVED_EXPERIMENT_PATH="./experiments/${CFG_STEM}/${CFG_PARENT}/${EXP_NAME}"
RESOLVED_TFBOARD_PATH="./experiments/${CFG_STEM}/${CFG_PARENT}/TFBoard/${EXP_NAME}"

if [[ -n "${RUNTIME_META_PATH}" ]]; then
  mkdir -p "$(dirname "${RUNTIME_META_PATH}")"
  cat > "${RUNTIME_META_PATH}" <<EOF
CONFIG_PATH_EXEC=${POINTGPT_DIR}/${CONFIG_PATH}
CFG_STEM=${CFG_STEM}
CFG_PARENT=${CFG_PARENT}
EXPERIMENT_PATH=${RESOLVED_EXPERIMENT_PATH}
TFBOARD_PATH=${RESOLVED_TFBOARD_PATH}
EXP_NAME=${EXP_NAME}
CKPT_PATH=${CKPT_PATH}
HYBRID_ROOT=${HYBRID_ROOT}
DATA_ROOT=${DATA_ROOT}
PC_PATH=${PC_PATH}
EOF
fi

echo "=== POINTGPT LOCAL DDP POST-PRETRAIN ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "config=${POINTGPT_DIR}/${CONFIG_PATH}"
echo "exp_name=${EXP_NAME}"
echo "ckpt_path=${CKPT_PATH}"
echo "hybrid_root=${HYBRID_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "pc_path=${PC_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "resolved_experiment_path=${RESOLVED_EXPERIMENT_PATH}"
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
  --finetune_model \
  --ckpts "${CKPT_PATH}" \
  ${EXTRA_ARGS}
