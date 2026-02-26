#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PRE_PID_FILE="${PRE_PID_FILE:-logs/preprocess/shapenet_v0/preprocess.pid}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v0}"
MIN_TRAIN_NPZ="${MIN_TRAIN_NPZ:-30000}"
SLEEP_SEC="${SLEEP_SEC:-60}"

if [ ! -f "${PRE_PID_FILE}" ]; then
  # backward compatibility with previous flat log layout
  if [ -f "logs/shapenet_preprocess.pid" ]; then
    PRE_PID_FILE="logs/shapenet_preprocess.pid"
  fi
fi

if [ ! -f "${PRE_PID_FILE}" ]; then
  echo "[error] preprocess pid file not found: ${PRE_PID_FILE}"
  exit 1
fi

pid="$(cat "${PRE_PID_FILE}")"
echo "[wait] waiting preprocess pid=${pid}"

while kill -0 "${pid}" 2>/dev/null; do
  sleep "${SLEEP_SEC}"
done

train_count="$(find "${CACHE_ROOT}/train" -name '*.npz' 2>/dev/null | wc -l || true)"
test_count="$(find "${CACHE_ROOT}/test" -name '*.npz' 2>/dev/null | wc -l || true)"
echo "[done] preprocess finished train=${train_count} test=${test_count}"

if [ "${train_count}" -lt "${MIN_TRAIN_NPZ}" ]; then
  echo "[abort] train npz too small (<${MIN_TRAIN_NPZ}); skip pretrain launch"
  exit 1
fi

echo "[start] launching ShapeNet simple pretrain"
CACHE_ROOT="${CACHE_ROOT}" bash scripts/pretrain/run_shapenet_simple_local.sh
