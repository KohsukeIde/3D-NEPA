#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
ROOT="${ROOT:-${WORKDIR}/data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
CKPT_PATH="${CKPT_PATH:-}"
RUN_NAME="${RUN_NAME:-pgpt_s_shapenetpart_ft}"
EPOCH="${EPOCH:-300}"
WARMUP_EPOCH="${WARMUP_EPOCH:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
MODEL_NAME="${MODEL_NAME:-PointGPT_S}"
SEED="${SEED:-0}"
GROUP_MODE="${GROUP_MODE:-fps_knn}"
GROUP_RADIUS="${GROUP_RADIUS:-0.22}"
GROUP_VOXEL_GRID="${GROUP_VOXEL_GRID:-6}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "[error] CKPT_PATH is required"
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] checkpoint not found: ${CKPT_PATH}"
  exit 2
fi
if [[ ! -d "${ROOT}" ]]; then
  echo "[error] ShapeNetPart root not found: ${ROOT}"
  exit 2
fi

export PYTHONPATH="${POINTGPT_DIR}:${PYTHONPATH:-}"

cd "${POINTGPT_DIR}/segmentation"

exec python main.py \
  --model pt \
  --model_name "${MODEL_NAME}" \
  --ckpts "${CKPT_PATH}" \
  --root "${ROOT}" \
  --epoch "${EPOCH}" \
  --warmup_epoch "${WARMUP_EPOCH}" \
  --batch_size "${BATCH_SIZE}" \
  --learning_rate "${LEARNING_RATE}" \
  --log_dir "${RUN_NAME}" \
  --seed "${SEED}" \
  --group_mode "${GROUP_MODE}" \
  --group_radius "${GROUP_RADIUS}" \
  --group_voxel_grid "${GROUP_VOXEL_GRID}"
