#!/usr/bin/env bash
set -eu

# Build paper-style ScanObjectNN variant caches:
#   - OBJ-BG     (main_split objectdataset)
#   - OBJ-ONLY   (main_split_nobg objectdataset)
#   - PB_T50_RS  (main_split augmentedrot_scale75)
#
# This script creates small variant-specific h5 roots via symlink, then runs
# preprocess_scanobjectnn.py per variant so train/test are cleanly separated.

if [[ -n "${WORKDIR:-}" ]] && [[ -f "${WORKDIR}/nepa3d/data/preprocess_scanobjectnn.py" ]]; then
  ROOT_DIR="${WORKDIR}"
elif [[ -n "${PBS_O_WORKDIR:-}" ]] && [[ -f "${PBS_O_WORKDIR}/nepa3d/data/preprocess_scanobjectnn.py" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

MAIN_SPLIT_DIR="${MAIN_SPLIT_DIR:-data/ScanObjectNN/h5_files/main_split}"
MAIN_SPLIT_NO_BG_DIR="${MAIN_SPLIT_NO_BG_DIR:-data/ScanObjectNN/h5_files/main_split_nobg}"
VARIANT_H5_ROOT="${VARIANT_H5_ROOT:-data/ScanObjectNN/h5_files_protocol_variants}"

OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v2}"
OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v2}"
PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v2}"

PT_POOL="${PT_POOL:-4000}"
RAY_POOL="${RAY_POOL:-256}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
NORMALIZE_PC="${NORMALIZE_PC:-1}"
QUERY_BBOX_MODE="${QUERY_BBOX_MODE:-unit}"
QUERY_BBOX_PAD="${QUERY_BBOX_PAD:-0.0}"
ALLOW_DUPLICATE_STEMS="${ALLOW_DUPLICATE_STEMS:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# Backfill pt_fps_order so NEPA-full eval with pt_sample_mode=fps does not
# fall back to on-the-fly FPS.
ENABLE_PT_FPS_ORDER="${ENABLE_PT_FPS_ORDER:-1}"
PT_FPS_SPLITS="${PT_FPS_SPLITS:-train:test}"
PT_FPS_KEY="${PT_FPS_KEY:-pt_fps_order}"
PT_FPS_K="${PT_FPS_K:-2048}"
PT_FPS_WORKERS="${PT_FPS_WORKERS:-32}"
PT_FPS_WRITE_MODE="${PT_FPS_WRITE_MODE:-append}"
PT_FPS_OVERWRITE="${PT_FPS_OVERWRITE:-0}"

mkdir -p "${VARIANT_H5_ROOT}"

make_variant_root() {
  local variant="$1"
  local src_dir="$2"
  local train_file="$3"
  local test_file="$4"
  local dst="${VARIANT_H5_ROOT}/${variant}"
  mkdir -p "${dst}"
  rm -f "${dst}"/*.h5
  ln -sf "$(realpath "${src_dir}/${train_file}")" "${dst}/${train_file}"
  ln -sf "$(realpath "${src_dir}/${test_file}")" "${dst}/${test_file}"
  echo "${dst}"
}

run_preprocess() {
  local scan_root="$1"
  local out_root="$2"
  local -a extra_args=()
  if [ "${ALLOW_DUPLICATE_STEMS}" = "1" ]; then
    extra_args+=(--allow_duplicate_stems)
  fi
  echo "[start] scan_root=${scan_root} out_root=${out_root}"
  "${PYTHON_BIN}" -u -m nepa3d.data.preprocess_scanobjectnn \
    --scan_root "${scan_root}" \
    --out_root "${out_root}" \
    --split all \
    --pt_pool "${PT_POOL}" \
    --ray_pool "${RAY_POOL}" \
    --pt_surface_ratio "${PT_SURFACE_RATIO}" \
    --pt_surface_sigma "${PT_SURFACE_SIGMA}" \
    --seed "${SEED}" \
    --workers "${WORKERS}" \
    --normalize_pc "${NORMALIZE_PC}" \
    --query_bbox_mode "${QUERY_BBOX_MODE}" \
    --query_bbox_pad "${QUERY_BBOX_PAD}" \
    "${extra_args[@]}"
  echo "[done ] out_root=${out_root}"
}

run_pt_fps_backfill() {
  local cache_root="$1"
  local fps_splits="${PT_FPS_SPLITS//:/,}"
  if [[ "${ENABLE_PT_FPS_ORDER}" != "1" ]]; then
    return 0
  fi
  echo "[start] pt_fps backfill cache_root=${cache_root} splits=${fps_splits} fps_k=${PT_FPS_K}"
  local -a fps_args=(
    --cache_root "${cache_root}"
    --splits "${fps_splits}"
    --pt_key "pt_xyz_pool"
    --out_key "${PT_FPS_KEY}"
    --fps_k "${PT_FPS_K}"
    --workers "${PT_FPS_WORKERS}"
    --write_mode "${PT_FPS_WRITE_MODE}"
  )
  if [[ "${PT_FPS_OVERWRITE}" == "1" ]]; then
    fps_args+=(--overwrite)
  fi
  "${PYTHON_BIN}" -u -m nepa3d.data.migrate_add_pt_fps_order "${fps_args[@]}"
  echo "[done ] pt_fps backfill cache_root=${cache_root}"
}

OBJ_BG_ROOT="$(make_variant_root "obj_bg" "${MAIN_SPLIT_DIR}" "training_objectdataset.h5" "test_objectdataset.h5")"
OBJ_ONLY_ROOT="$(make_variant_root "obj_only" "${MAIN_SPLIT_NO_BG_DIR}" "training_objectdataset.h5" "test_objectdataset.h5")"
PB_T50_RS_ROOT="$(make_variant_root "pb_t50_rs" "${MAIN_SPLIT_DIR}" "training_objectdataset_augmentedrot_scale75.h5" "test_objectdataset_augmentedrot_scale75.h5")"

if [[ "${SKIP_PREPROCESS}" != "1" ]]; then
  run_preprocess "${OBJ_BG_ROOT}" "${OBJ_BG_CACHE}"
  run_preprocess "${OBJ_ONLY_ROOT}" "${OBJ_ONLY_CACHE}"
  run_preprocess "${PB_T50_RS_ROOT}" "${PB_T50_RS_CACHE}"
else
  echo "[skip] preprocess disabled (SKIP_PREPROCESS=1)"
fi

run_pt_fps_backfill "${OBJ_BG_CACHE}"
run_pt_fps_backfill "${OBJ_ONLY_CACHE}"
run_pt_fps_backfill "${PB_T50_RS_CACHE}"

echo "[summary]"
echo "  preprocess normalize_pc=${NORMALIZE_PC} query_bbox_mode=${QUERY_BBOX_MODE} query_bbox_pad=${QUERY_BBOX_PAD}"
for d in "${OBJ_BG_CACHE}" "${OBJ_ONLY_CACHE}" "${PB_T50_RS_CACHE}"; do
  tr_n="$(find "${d}/train" -type f -name '*.npz' 2>/dev/null | wc -l || true)"
  te_n="$(find "${d}/test" -type f -name '*.npz' 2>/dev/null | wc -l || true)"
  echo "  - ${d}: train=${tr_n} test=${te_n} pt_fps_order=${PT_FPS_KEY}"
done
