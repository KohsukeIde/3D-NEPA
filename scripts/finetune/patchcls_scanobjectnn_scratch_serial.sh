#!/usr/bin/env bash
set -euo pipefail

# PTv3-style serial patch grouping baseline:
# Morton sort -> contiguous chunk groups -> mini-PointNet.
#
# This wraps patchcls_scanobjectnn_scratch.sh and only overrides patch grouping.

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

export WORKDIR="${ROOT_DIR}"

# Defaults focused on 1024-point classification parity.
export CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
export DATA_FORMAT="${DATA_FORMAT:-npz}"
export SCAN_VARIANT="${SCAN_VARIANT:-pb_t50_rs}"
export RUN_NAME="${RUN_NAME:-patchcls_scan_serial_scratch}"
export SAVE_DIR="${SAVE_DIR:-runs/sanity/patchcls}"

export N_POINT="${N_POINT:-1024}"
export GROUP_SIZE="${GROUP_SIZE:-16}"   # 1024 / 16 = 64 serial patches
export NUM_GROUPS="${NUM_GROUPS:-64}"   # pos/metadata alignment
export PATCH_EMBED="${PATCH_EMBED:-serial}"
export SERIAL_ORDER="${SERIAL_ORDER:-morton}"
export SERIAL_BITS="${SERIAL_BITS:-10}"
export SERIAL_SHUFFLE_WITHIN_PATCH="${SERIAL_SHUFFLE_WITHIN_PATCH:-0}"

# Keep PM-aligned recipe by default.
export EPOCHS="${EPOCHS:-300}"
export BATCH="${BATCH:-64}"
export BATCH_MODE="${BATCH_MODE:-global}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export LR="${LR:-5e-4}"
export WD="${WD:-0.05}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
export AUG_PRESET="${AUG_PRESET:-pointmae}"
export VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
export VAL_RATIO="${VAL_RATIO:-0.1}"
export VAL_SEED="${VAL_SEED:-0}"
export SEED="${SEED:-0}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
export PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"

exec "${ROOT_DIR}/scripts/finetune/patchcls_scanobjectnn_scratch.sh"

