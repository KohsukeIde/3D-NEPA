#!/usr/bin/env bash
set -eu

# Run ScanObjectNN few-shot tables for protocol variants.
# Default variants:
#   - obj_bg
#   - obj_only
#   - pb_t50_rs
#
# Default methods are the pretrain-independent subset:
#   scratch + shapenet_nepa + shapenet_mesh_udf_nepa
# You can add mix methods via METHODS env if desired.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

VARIANTS="${VARIANTS:-obj_bg obj_only pb_t50_rs}"
METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-0}"

BASE_RUN_ROOT="${BASE_RUN_ROOT:-runs/scan_variants}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-logs/finetune/scan_variants}"

variant_cache_root() {
  case "$1" in
    obj_bg) echo "data/scanobjectnn_obj_bg_v2" ;;
    obj_only) echo "data/scanobjectnn_obj_only_v2" ;;
    pb_t50_rs) echo "data/scanobjectnn_pb_t50_rs_v2" ;;
    *)
      echo "[error] unknown variant: $1" >&2
      exit 1
      ;;
  esac
}

for v in ${VARIANTS}; do
  CACHE_ROOT="$(variant_cache_root "${v}")"
  RUN_ROOT="${BASE_RUN_ROOT}/${v}"
  LOG_ROOT="${BASE_LOG_ROOT}/${v}/jobs"
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
  echo "[variant] ${v}"
  echo "  cache_root=${CACHE_ROOT}"
  echo "  run_root=${RUN_ROOT}"
  echo "  log_root=${LOG_ROOT}"
  CACHE_ROOT="${CACHE_ROOT}" RUN_ROOT="${RUN_ROOT}" LOG_ROOT="${LOG_ROOT}" METHODS="${METHODS}" ABLATE_POINT_DIST="${ABLATE_POINT_DIST}" \
    bash scripts/finetune/run_scanobjectnn_m1_table_local.sh
done
