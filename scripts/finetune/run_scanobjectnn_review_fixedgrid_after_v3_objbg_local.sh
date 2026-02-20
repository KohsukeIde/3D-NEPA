#!/usr/bin/env bash
set -eu

# Run fixed-grid diagnostics after corrected fair FT v3 (obj_bg) finishes.
# This is an internal diagnostic chain (v0-style query-token setting), not a
# PointGPT-style fair-comparison mainline.
#
# Stages:
#   G1) fixed-grid query + mean_no_special pooling
#   G2) fixed-grid query + BOS pooling (global-query proxy)
#
# Default sweep is intentionally small for quick validation:
#   variant=obj_bg, K=0, seeds=0,1,2, methods=5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

count_lastpt() {
  local run_root="$1"
  find "${run_root}" -type f -name last.pt 2>/dev/null | wc -l
}

wait_for_v3_objbg() {
  local v3_run_root="$1"
  local expected="$2"
  local pid_file="$3"

  while true; do
    local done
    done="$(count_lastpt "${v3_run_root}")"
    echo "[wait-v3] done=${done}/${expected}"
    if [ "${done}" -ge "${expected}" ]; then
      echo "[wait-v3] v3 obj_bg complete"
      break
    fi

    if [ -f "${pid_file}" ]; then
      local pid
      pid="$(cat "${pid_file}" 2>/dev/null || true)"
      if [ -n "${pid}" ] && ! ps -p "${pid}" >/dev/null 2>&1; then
        echo "[wait-v3][warn] pid file exists but process is not running: ${pid}"
      fi
    fi
    sleep 60
  done
}

count_words() {
  # shellcheck disable=SC2086
  set -- $1
  echo $#
}

METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"
SEEDS="${SEEDS:-0 1 2}"
K_LIST="${K_LIST:-0}"
VARIANTS="${VARIANTS:-obj_bg}"

V3_RUN_ROOT="${V3_RUN_ROOT:-runs/scan_variants_review_ft_fair_pcxyz2k_fpsfix_v3/obj_bg}"
V3_EXPECTED="${V3_EXPECTED:-75}"
V3_PID_FILE="${V3_PID_FILE:-logs/finetune/scan_variants_review_ft_fair_pcxyz2k_fpsfix_v3_objbg/pipeline.pid}"

wait_for_v3_objbg "${V3_RUN_ROOT}" "${V3_EXPECTED}" "${V3_PID_FILE}"

nm="$(count_words "${METHODS}")"
ns="$(count_words "${SEEDS}")"
nk="$(count_words "${K_LIST}")"
nv="$(count_words "${VARIANTS}")"
EXPECTED=$((nm * ns * nk * nv))

COMMON_ENV=(
  "VARIANTS=${VARIANTS}"
  "METHODS=${METHODS}"
  "SEEDS=${SEEDS}"
  "K_LIST=${K_LIST}"
  "BACKEND=${BACKEND:-pointcloud_noray}"
  "N_POINT=${N_POINT:-256}"
  "N_RAY=${N_RAY:-0}"
  "ALLOW_SCALE_UP=${ALLOW_SCALE_UP:-0}"
  "CLS_IS_CAUSAL=${CLS_IS_CAUSAL:-0}"
  "MC_EVAL_K=${MC_EVAL_K:-10}"
  "MC_EVAL_K_VAL=${MC_EVAL_K_VAL:-1}"
  "MC_EVAL_K_TEST=${MC_EVAL_K_TEST:-10}"
  "PT_XYZ_KEY=${PT_XYZ_KEY:-pt_xyz_pool}"
  "PT_DIST_KEY=${PT_DIST_KEY:-pt_dist_pool}"
  "PT_FPS_KEY=${PT_FPS_KEY:-pt_fps_order}"
  "PT_RFPS_M=${PT_RFPS_M:-4096}"
  "AUG_PRESET=${AUG_PRESET:-scanobjectnn}"
  "AUG_ROTATE_Z=${AUG_ROTATE_Z:-0}"
  "AUG_EVAL=${AUG_EVAL:-0}"
  "ABLATE_POINT_DIST=${ABLATE_POINT_DIST:-0}"
)

echo "[stage-g1] fixed-grid + mean_no_special"
G1_BASE_RUN_ROOT="${G1_BASE_RUN_ROOT:-runs/scan_variants_review_ft_bidir_poolfix_fixedgrid_v0diag}"
G1_BASE_LOG_ROOT="${G1_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_bidir_poolfix_fixedgrid_v0diag}"
g1_done="$(count_lastpt "${G1_BASE_RUN_ROOT}")"
echo "[stage-g1] done=${g1_done}/${EXPECTED}"
if [ "${g1_done}" -lt "${EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    BASE_RUN_ROOT="${G1_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${G1_BASE_LOG_ROOT}" \
    PT_SAMPLE_MODE_TRAIN="fixed_grid" \
    PT_SAMPLE_MODE_EVAL="fixed_grid" \
    CLS_POOLING="mean_no_special" \
    RUN_SUFFIX="${RUN_SUFFIX_G1:-_fixed_grid}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
g1_done="$(count_lastpt "${G1_BASE_RUN_ROOT}")"
echo "[stage-g1] done=${g1_done}/${EXPECTED}"
if [ "${g1_done}" -lt "${EXPECTED}" ]; then
  echo "[error] stage-g1 incomplete"
  exit 1
fi

echo "[stage-g2] fixed-grid + bos (global-query proxy)"
G2_BASE_RUN_ROOT="${G2_BASE_RUN_ROOT:-runs/scan_variants_review_ft_bidir_poolfix_fixedgrid_bos_v0diag}"
G2_BASE_LOG_ROOT="${G2_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_bidir_poolfix_fixedgrid_bos_v0diag}"
g2_done="$(count_lastpt "${G2_BASE_RUN_ROOT}")"
echo "[stage-g2] done=${g2_done}/${EXPECTED}"
if [ "${g2_done}" -lt "${EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    BASE_RUN_ROOT="${G2_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${G2_BASE_LOG_ROOT}" \
    PT_SAMPLE_MODE_TRAIN="fixed_grid" \
    PT_SAMPLE_MODE_EVAL="fixed_grid" \
    CLS_POOLING="bos" \
    RUN_SUFFIX="${RUN_SUFFIX_G2:-_fixed_grid_bos}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
g2_done="$(count_lastpt "${G2_BASE_RUN_ROOT}")"
echo "[stage-g2] done=${g2_done}/${EXPECTED}"
if [ "${g2_done}" -lt "${EXPECTED}" ]; then
  echo "[error] stage-g2 incomplete"
  exit 1
fi

echo "[done] fixed-grid diagnostics complete"
