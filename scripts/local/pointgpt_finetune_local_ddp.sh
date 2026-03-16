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
EXP_NAME="${EXP_NAME:-pointgpt_finetune_local_ddp_$(date +%Y%m%d_%H%M%S)}"
CKPT_PATH="${CKPT_PATH:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29539}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
VAL_FREQ="${VAL_FREQ:-1}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-0}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
NO_TEST_AS_VAL="${NO_TEST_AS_VAL:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-transfer}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_local_ddp_ft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_TAGS="${WANDB_TAGS:-pointgpt,finetune,ddp2}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"
RUNTIME_META_PATH="${RUNTIME_META_PATH:-}"

SCAN_ROOT="${SCAN_ROOT:-${WORKDIR}/data/ScanObjectNN}"
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
if [[ "${NO_TEST_AS_VAL}" != "0" && "${NO_TEST_AS_VAL}" != "1" ]]; then
  echo "[error] NO_TEST_AS_VAL must be 0 or 1 (got: ${NO_TEST_AS_VAL})"
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

CONFIG_PATH_EXEC="${POINTGPT_DIR}/${CONFIG_PATH}"
if [[ "${NO_TEST_AS_VAL}" == "1" ]]; then
  TMP_CFG_DIR="${WORKDIR}/tmp/pointgpt_finetune_cfg_override"
  mkdir -p "${TMP_CFG_DIR}"
  TMP_CFG="${TMP_CFG_DIR}/${EXP_NAME}.yaml"
  cp "${CONFIG_PATH_EXEC}" "${TMP_CFG}"
  python - "${TMP_CFG}" "${VAL_RATIO}" "${VAL_SEED}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
val_ratio = float(sys.argv[2])
val_seed = int(sys.argv[3])
if not (0.0 < val_ratio < 1.0):
    raise SystemExit(f"[error] VAL_RATIO must be in (0,1), got {val_ratio}")

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ds = cfg.get("dataset", {})
for split in ("train", "val"):
    others = ds.get(split, {}).get("others", {})
    others["val_ratio"] = val_ratio
    others["val_seed"] = val_seed
    ds[split]["others"] = others
ds["val"]["others"]["subset"] = "val"
cfg["dataset"] = ds

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  CONFIG_PATH_EXEC="${TMP_CFG}"
fi

cd "${POINTGPT_DIR}"

CFG_STEM="$(basename "${CONFIG_PATH_EXEC%.*}")"
CFG_PARENT="$(basename "$(dirname "${CONFIG_PATH_EXEC}")")"
RESOLVED_EXPERIMENT_PATH="./experiments/${CFG_STEM}/${CFG_PARENT}/${EXP_NAME}"
RESOLVED_TFBOARD_PATH="./experiments/${CFG_STEM}/${CFG_PARENT}/TFBoard/${EXP_NAME}"

if [[ -n "${RUNTIME_META_PATH}" ]]; then
  mkdir -p "$(dirname "${RUNTIME_META_PATH}")"
  cat > "${RUNTIME_META_PATH}" <<EOF
CONFIG_PATH_EXEC=${CONFIG_PATH_EXEC}
CFG_STEM=${CFG_STEM}
CFG_PARENT=${CFG_PARENT}
EXPERIMENT_PATH=${RESOLVED_EXPERIMENT_PATH}
TFBOARD_PATH=${RESOLVED_TFBOARD_PATH}
EXP_NAME=${EXP_NAME}
NO_TEST_AS_VAL=${NO_TEST_AS_VAL}
VAL_RATIO=${VAL_RATIO}
VAL_SEED=${VAL_SEED}
EOF
fi

echo "=== POINTGPT LOCAL DDP FINETUNE ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "config=${CONFIG_PATH_EXEC}"
echo "exp_name=${EXP_NAME}"
echo "ckpt_path=${CKPT_PATH}"
echo "scan_root=${SCAN_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "val_freq=${VAL_FREQ}"
echo "ft_recon_weight=${FT_RECON_WEIGHT}"
echo "save_last_every_epoch=${SAVE_LAST_EVERY_EPOCH}"
echo "no_test_as_val=${NO_TEST_AS_VAL}"
echo "resolved_experiment_path=${RESOLVED_EXPERIMENT_PATH}"
if [[ "${NO_TEST_AS_VAL}" == "1" ]]; then
  echo "policy=STRICT(train->val split)"
  echo "val_ratio=${VAL_RATIO}"
  echo "val_seed=${VAL_SEED}"
else
  echo "policy=LEGACY(test-as-val)"
fi
echo "use_wandb=${USE_WANDB} project=${WANDB_PROJECT} group=${WANDB_GROUP} run=${WANDB_RUN_NAME} mode=${WANDB_MODE}"
echo

torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --launcher pytorch \
  --config "${CONFIG_PATH_EXEC}" \
  --exp_name "${EXP_NAME}" \
  --num_workers "${NUM_WORKERS}" \
  --val_freq "${VAL_FREQ}" \
  --ft_recon_weight "${FT_RECON_WEIGHT}" \
  --save_last_every_epoch "${SAVE_LAST_EVERY_EPOCH}" \
  --finetune_model \
  --ckpts "${CKPT_PATH}" \
  ${EXTRA_ARGS}
