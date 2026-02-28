#!/bin/bash
# PBS worker script: backfill RFPS order bank into cached npz files.
set -euo pipefail
export PYTHONUNBUFFERED=1

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
cd "${WORKDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load cuda/12.9 2>/dev/null || true
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v0}"
SPLITS="${SPLITS:-train}"
PT_KEY="${PT_KEY:-pc_xyz}"
OUT_KEY="${OUT_KEY:-pc_rfps_order_bank}"
RFPS_K="${RFPS_K:-1024}"
RFPS_M="${RFPS_M:-4096}"
BANK_SIZE="${BANK_SIZE:-8}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-24}"
WRITE_MODE="${WRITE_MODE:-append}"
OVERWRITE="${OVERWRITE:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
LOG_EVERY="${LOG_EVERY:-1000}"
CHUNKSIZE="${CHUNKSIZE:-32}"

echo "=== RFPS BANK BACKFILL ==="
echo "date=$(date -Iseconds)"
echo "host=$(hostname)"
echo "cache_root=${CACHE_ROOT} splits=${SPLITS}"
echo "pt_key=${PT_KEY} out_key=${OUT_KEY}"
echo "rfps_k=${RFPS_K} rfps_m=${RFPS_M} bank_size=${BANK_SIZE}"
echo "workers=${WORKERS} write_mode=${WRITE_MODE} overwrite=${OVERWRITE}"
echo "num_shards=${NUM_SHARDS} shard_id=${SHARD_ID}"

cmd=(
  python -u -m nepa3d.data.migrate_add_pt_rfps_order_bank
  --cache_root "${CACHE_ROOT}"
  --splits "${SPLITS}"
  --pt_key "${PT_KEY}"
  --out_key "${OUT_KEY}"
  --rfps_k "${RFPS_K}"
  --rfps_m "${RFPS_M}"
  --bank_size "${BANK_SIZE}"
  --seed "${SEED}"
  --workers "${WORKERS}"
  --write_mode "${WRITE_MODE}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --log_every "${LOG_EVERY}"
  --chunksize "${CHUNKSIZE}"
)
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "+ ${cmd[*]}"
"${cmd[@]}"

echo "done $(date -Iseconds)"
