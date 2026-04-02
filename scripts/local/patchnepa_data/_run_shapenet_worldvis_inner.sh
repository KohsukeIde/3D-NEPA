#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/shapenet_worldvis_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/shapenet_worldvis_local.pid}"
NICE_LEVEL="${NICE_LEVEL:-10}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

mkdir -p "${LOG_ROOT}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}

trap cleanup EXIT

echo "$$" > "${PID_FILE}"

{
  echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S %Z') start tmux worldvis build"
  echo "[launcher] out_root=${OUT_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/shapenet_cache_v2_20260401_worldvis}"
  echo "[launcher] workers=${WORKERS:-16} timeout_sec=${TASK_TIMEOUT_SEC:-900}"
  echo "[launcher] nice=${NICE_LEVEL} ionice_class=${IONICE_CLASS} ionice_level=${IONICE_LEVEL}"
} >> "${LOG_FILE}"

cmd=(bash "${ROOT_DIR}/scripts/preprocess/preprocess_shapenet_v2.sh")
if command -v ionice >/dev/null 2>&1; then
  cmd=(ionice -c "${IONICE_CLASS}" -n "${IONICE_LEVEL}" "${cmd[@]}")
fi
cmd=(nice -n "${NICE_LEVEL}" "${cmd[@]}")

"${cmd[@]}" >> "${LOG_FILE}" 2>&1
