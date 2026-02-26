#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/preprocess_scanobjectnn_protocol_variants.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
JOB_NAME="${JOB_NAME:-preprocess_scanobjectnn_variants_v3_nonorm}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

MAIN_SPLIT_DIR="${MAIN_SPLIT_DIR:-data/ScanObjectNN/h5_files/main_split}"
MAIN_SPLIT_NO_BG_DIR="${MAIN_SPLIT_NO_BG_DIR:-data/ScanObjectNN/h5_files/main_split_nobg}"
VARIANT_H5_ROOT="${VARIANT_H5_ROOT:-data/ScanObjectNN/h5_files_protocol_variants}"
OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"

PT_POOL="${PT_POOL:-4000}"
RAY_POOL="${RAY_POOL:-256}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
NORMALIZE_PC="${NORMALIZE_PC:-0}"
QUERY_BBOX_MODE="${QUERY_BBOX_MODE:-auto}"
QUERY_BBOX_PAD="${QUERY_BBOX_PAD:-0.0}"
ALLOW_DUPLICATE_STEMS="${ALLOW_DUPLICATE_STEMS:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

ENABLE_PT_FPS_ORDER="${ENABLE_PT_FPS_ORDER:-1}"
PT_FPS_SPLITS="${PT_FPS_SPLITS:-train:test}"
PT_FPS_KEY="${PT_FPS_KEY:-pt_fps_order}"
PT_FPS_K="${PT_FPS_K:-2048}"
PT_FPS_WORKERS="${PT_FPS_WORKERS:-32}"
PT_FPS_WRITE_MODE="${PT_FPS_WRITE_MODE:-append}"
PT_FPS_OVERWRITE="${PT_FPS_OVERWRITE:-0}"

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/scanobjectnn_variants_v3_nonorm}"
mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "MAIN_SPLIT_DIR=${MAIN_SPLIT_DIR}"
  "MAIN_SPLIT_NO_BG_DIR=${MAIN_SPLIT_NO_BG_DIR}"
  "VARIANT_H5_ROOT=${VARIANT_H5_ROOT}"
  "OBJ_BG_CACHE=${OBJ_BG_CACHE}"
  "OBJ_ONLY_CACHE=${OBJ_ONLY_CACHE}"
  "PB_T50_RS_CACHE=${PB_T50_RS_CACHE}"
  "PT_POOL=${PT_POOL}"
  "RAY_POOL=${RAY_POOL}"
  "PT_SURFACE_RATIO=${PT_SURFACE_RATIO}"
  "PT_SURFACE_SIGMA=${PT_SURFACE_SIGMA}"
  "SEED=${SEED}"
  "WORKERS=${WORKERS}"
  "NORMALIZE_PC=${NORMALIZE_PC}"
  "QUERY_BBOX_MODE=${QUERY_BBOX_MODE}"
  "QUERY_BBOX_PAD=${QUERY_BBOX_PAD}"
  "ALLOW_DUPLICATE_STEMS=${ALLOW_DUPLICATE_STEMS}"
  "SKIP_PREPROCESS=${SKIP_PREPROCESS}"
  "ENABLE_PT_FPS_ORDER=${ENABLE_PT_FPS_ORDER}"
  "PT_FPS_SPLITS=${PT_FPS_SPLITS}"
  "PT_FPS_KEY=${PT_FPS_KEY}"
  "PT_FPS_K=${PT_FPS_K}"
  "PT_FPS_WORKERS=${PT_FPS_WORKERS}"
  "PT_FPS_WRITE_MODE=${PT_FPS_WRITE_MODE}"
  "PT_FPS_OVERWRITE=${PT_FPS_OVERWRITE}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${LOG_DIR}/${JOB_NAME}.out"
  -e "${LOG_DIR}/${JOB_NAME}.err"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

echo "[submit] ${JOB_NAME}"
echo "[submit] obj_bg=${OBJ_BG_CACHE}"
echo "[submit] obj_only=${OBJ_ONLY_CACHE}"
echo "[submit] pb_t50_rs=${PB_T50_RS_CACHE}"
echo "[submit] normalize_pc=${NORMALIZE_PC} query_bbox_mode=${QUERY_BBOX_MODE} query_bbox_pad=${QUERY_BBOX_PAD}"
echo "[submit] skip_preprocess=${SKIP_PREPROCESS} pt_fps_backfill=${ENABLE_PT_FPS_ORDER} fps_k=${PT_FPS_K}"
"${cmd[@]}"
