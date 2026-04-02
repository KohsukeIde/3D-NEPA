#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.pid}"
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
  echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S %Z') start ScanObjectNN protocol variant build"
  echo "[launcher] obj_bg_cache=${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
  echo "[launcher] obj_only_cache=${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
  echo "[launcher] pb_t50_rs_cache=${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
  echo "[launcher] workers=${WORKERS:-8} pt_fps_workers=${PT_FPS_WORKERS:-8}"
  echo "[launcher] normalize_pc=${NORMALIZE_PC:-0} query_bbox_mode=${QUERY_BBOX_MODE:-auto}"
  echo "[launcher] nice=${NICE_LEVEL} ionice_class=${IONICE_CLASS} ionice_level=${IONICE_LEVEL}"
} >> "${LOG_FILE}"

cmd=(bash "${ROOT_DIR}/scripts/preprocess/preprocess_scanobjectnn_protocol_variants.sh")
if command -v ionice >/dev/null 2>&1; then
  cmd=(ionice -c "${IONICE_CLASS}" -n "${IONICE_LEVEL}" "${cmd[@]}")
fi
cmd=(nice -n "${NICE_LEVEL}" "${cmd[@]}")

"${cmd[@]}" >> "${LOG_FILE}" 2>&1
