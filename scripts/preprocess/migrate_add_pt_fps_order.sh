#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   CACHE_ROOT=data/scanobjectnn_main_split_v2 SPLITS=train,test FPS_K=2048 WORKERS=16 \
#     bash scripts/preprocess/migrate_add_pt_fps_order.sh
#
# Existing cache backfill for observed-point FPS order:
#   CACHE_ROOT=data/scanobjectnn_main_split_v2 SPLITS=train,test FPS_K=2048 \
#   PT_KEY=pc_xyz OUT_KEY=pc_fps_order WORKERS=16 \
#     bash scripts/preprocess/migrate_add_pt_fps_order.sh

CACHE_ROOT=${CACHE_ROOT:-}
SPLITS=${SPLITS:-train,test}
FPS_K=${FPS_K:-2048}
PT_KEY=${PT_KEY:-pt_xyz_pool}
OUT_KEY=${OUT_KEY:-pt_fps_order}
WORKERS=${WORKERS:-8}
OVERWRITE=${OVERWRITE:-0}
WRITE_MODE=${WRITE_MODE:-append}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}
LOG_EVERY=${LOG_EVERY:-1000}
CHUNKSIZE=${CHUNKSIZE:-32}

if [[ -z "${CACHE_ROOT}" ]]; then
  echo "ERROR: CACHE_ROOT is empty" >&2
  exit 2
fi

args=(
  --cache_root "${CACHE_ROOT}"
  --splits "${SPLITS}"
  --fps_k "${FPS_K}"
  --pt_key "${PT_KEY}"
  --out_key "${OUT_KEY}"
  --workers "${WORKERS}"
  --write_mode "${WRITE_MODE}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --log_every "${LOG_EVERY}"
  --chunksize "${CHUNKSIZE}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  args+=(--overwrite)
fi

.venv/bin/python -u -m nepa3d.data.migrate_add_pt_fps_order "${args[@]}"
