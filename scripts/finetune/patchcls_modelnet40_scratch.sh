#!/usr/bin/env bash
set -euo pipefail

# Patchified Transformer scratch baseline for ModelNet40.

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v2}"

RUN_NAME="${RUN_NAME:-patchcls_modelnet_scratch}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-3}"
WD="${WD:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

N_POINT="${N_POINT:-1024}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"

NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
SEED="${SEED:-0}"

SAVE_DIR="${SAVE_DIR:-runs/patchcls}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${CACHE_ROOT}" ]; then
  echo "[error] missing cache root: ${CACHE_ROOT}"
  exit 1
fi

"${PYTHON_BIN}" -m nepa3d.train.finetune_patch_cls \
  --cache_root "${CACHE_ROOT}" \
  --run_name "${RUN_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --lr "${LR}" \
  --weight_decay "${WD}" \
  --lr_scheduler cosine \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --n_point "${N_POINT}" \
  --pt_sample_mode_train random \
  --pt_sample_mode_eval fps \
  --aug_preset default \
  --val_ratio "${VAL_RATIO}" \
  --val_seed "${VAL_SEED}" \
  --val_split_mode "${VAL_SPLIT_MODE}" \
  --seed "${SEED}" \
  --num_groups "${NUM_GROUPS}" \
  --group_size "${GROUP_SIZE}" \
  --pooling cls \
  --is_causal 0 \
  --num_workers "${NUM_WORKERS}"
