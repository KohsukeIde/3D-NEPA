#!/usr/bin/env bash
set -euo pipefail

# Patchified Transformer scratch baseline for ScanObjectNN.
#
# Goal: reproduce the typical "Transformer" baseline range (~0.77-0.80+)
# before attributing gaps to NEPA pretraining details.

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
DATA_FORMAT="${DATA_FORMAT:-npz}"  # npz | scan_h5
SCAN_H5_ROOT="${SCAN_H5_ROOT:-data/ScanObjectNN/h5_files/main_split}"
SCAN_VARIANT="${SCAN_VARIANT:-pb_t50_rs}"  # auto|obj_bg|obj_only|pb_t50_rs
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"

# Training hyperparams (Point-MAE-ish defaults)
RUN_NAME="${RUN_NAME:-patchcls_scan_scratch}"
CKPT="${CKPT:-}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
LR="${LR:-1e-3}"
WD="${WD:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

N_POINT="${N_POINT:-1024}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"

NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_MODE="${BATCH_MODE:-global}"  # global | per_proc
MASTER_PORT="${MASTER_PORT:-29577}"

SAVE_DIR="${SAVE_DIR:-runs/patchcls}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ "${DATA_FORMAT}" != "npz" ]] && [[ "${DATA_FORMAT}" != "scan_h5" ]]; then
  echo "[error] DATA_FORMAT must be one of: npz | scan_h5 (got: ${DATA_FORMAT})"
  exit 1
fi
if [[ "${DATA_FORMAT}" == "npz" ]]; then
  if [ ! -d "${CACHE_ROOT}" ]; then
    echo "[error] missing cache root: ${CACHE_ROOT}"
    exit 1
  fi
  if [[ "${CACHE_ROOT}" == *"scanobjectnn_"*"_v2" ]] && [[ "${ALLOW_SCAN_UNISCALE_V2}" != "1" ]]; then
    echo "[error] CACHE_ROOT=${CACHE_ROOT} is a uniscale v2 cache and is disallowed by policy."
    echo "        Use v3_nonorm variant caches, or set ALLOW_SCAN_UNISCALE_V2=1 for intentional legacy reruns."
    exit 2
  fi
else
  if [ ! -d "${SCAN_H5_ROOT}" ]; then
    echo "[error] missing ScanObjectNN h5 root: ${SCAN_H5_ROOT}"
    exit 1
  fi
fi

ARGS=(
  -m nepa3d.train.finetune_patch_cls
  --data_format "${DATA_FORMAT}"
  --cache_root "${CACHE_ROOT}"
  --scan_h5_root "${SCAN_H5_ROOT}"
  --scan_variant "${SCAN_VARIANT}"
  --run_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --ckpt "${CKPT}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --batch_mode "${BATCH_MODE}"
  --lr "${LR}"
  --weight_decay "${WD}"
  --lr_scheduler cosine
  --warmup_epochs "${WARMUP_EPOCHS}"
  --n_point "${N_POINT}"
  --pt_sample_mode_train "${PT_SAMPLE_MODE_TRAIN}"
  --pt_sample_mode_eval "${PT_SAMPLE_MODE_EVAL}"
  --aug_preset default
  --val_ratio "${VAL_RATIO}"
  --val_seed "${VAL_SEED}"
  --val_split_mode "${VAL_SPLIT_MODE}"
  --allow_scan_uniscale_v2 "${ALLOW_SCAN_UNISCALE_V2}"
  --seed "${SEED}"
  --num_groups "${NUM_GROUPS}"
  --group_size "${GROUP_SIZE}"
  --pooling cls
  --is_causal 0
  --num_workers "${NUM_WORKERS}"
)

echo "[patchcls] data_format=${DATA_FORMAT} variant=${SCAN_VARIANT} run_name=${RUN_NAME}"
echo "[patchcls] nproc_per_node=${NPROC_PER_NODE} batch=${BATCH} batch_mode=${BATCH_MODE}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${ARGS[@]}"
fi
