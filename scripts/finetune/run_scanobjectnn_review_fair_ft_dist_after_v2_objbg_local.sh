#!/usr/bin/env bash
set -eu

# After current v2 (xyz-only, obj_bg) completes, run dist-enabled obj_bg tables:
#   1) v1-style sampling: train=random, eval=random
#   2) v2-style sampling: train=random, eval=fps

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

wait_for_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [ ! -f "${pid_file}" ]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [ -z "${pid}" ]; then
    return 0
  fi
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "[wait] ${label} pid=${pid}"
    while ps -p "${pid}" >/dev/null 2>&1; do
      sleep 30
    done
    echo "[wait] ${label} finished"
  fi
}

count_done_variant() {
  local run_root="$1"
  local variant="$2"
  find "${run_root}/${variant}" -type f -name last.pt 2>/dev/null | wc -l
}

METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"
VARIANT="${VARIANT:-obj_bg}"
EXPECTED="${EXPECTED:-75}" # 5 methods x 5 K x 3 seeds

WAIT_PID_FILE="${WAIT_PID_FILE:-logs/finetune/scan_variants_review_fair_ft_chain_v2_objbg/pipeline.pid}"
wait_for_pid_file "${WAIT_PID_FILE}" "fair_ft_v2_objbg_xyzonly"

COMMON_ENV=(
  "BACKEND=${BACKEND:-pointcloud_noray}"
  "N_POINT=${N_POINT:-2048}"
  "N_RAY=${N_RAY:-0}"
  "ALLOW_SCALE_UP=${ALLOW_SCALE_UP:-1}"
  "CLS_IS_CAUSAL=${CLS_IS_CAUSAL:-0}"
  "CLS_POOLING=${CLS_POOLING:-mean_pts}"
  "MC_EVAL_K=${MC_EVAL_K:-1}"
  "MC_EVAL_K_VAL=${MC_EVAL_K_VAL:-1}"
  "MC_EVAL_K_TEST=${MC_EVAL_K_TEST:-1}"
  "PT_XYZ_KEY=${PT_XYZ_KEY:-pc_xyz}"
  "PT_DIST_KEY=${PT_DIST_KEY:-pt_dist_pool}"
  "PT_FPS_KEY=${PT_FPS_KEY:-pt_fps_order}"
  "PT_RFPS_M=${PT_RFPS_M:-4096}"
  "ABLATE_POINT_DIST=${ABLATE_POINT_DIST:-0}"
  "AUG_PRESET=${AUG_PRESET:-scanobjectnn}"
  "AUG_ROTATE_Z=${AUG_ROTATE_Z:-0}"
  "AUG_EVAL=${AUG_EVAL:-0}"
)

# Stage D1: dist-enabled v1-style sampling (eval=random)
D1_RUN_ROOT="${D1_RUN_ROOT:-runs/scan_variants_review_ft_fair_pcxyz2k_dist_v1}"
D1_LOG_ROOT="${D1_LOG_ROOT:-logs/finetune/scan_variants_review_ft_fair_pcxyz2k_dist_v1}"

echo "[stage-d1] dist-enabled v1-style (${VARIANT})"
d1_done="$(count_done_variant "${D1_RUN_ROOT}" "${VARIANT}")"
echo "[stage-d1] done=${d1_done}/${EXPECTED}"
if [ "${d1_done}" -lt "${EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    "PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN_D1:-random}" \
    "PT_SAMPLE_MODE_EVAL=${PT_SAMPLE_MODE_EVAL_D1:-random}" \
    VARIANTS="${VARIANT}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${D1_RUN_ROOT}" \
    BASE_LOG_ROOT="${D1_LOG_ROOT}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
d1_done="$(count_done_variant "${D1_RUN_ROOT}" "${VARIANT}")"
echo "[stage-d1] done=${d1_done}/${EXPECTED}"
if [ "${d1_done}" -lt "${EXPECTED}" ]; then
  echo "[error] stage-d1 incomplete"
  exit 1
fi

# Stage D2: dist-enabled v2-style sampling (eval=fps)
D2_RUN_ROOT="${D2_RUN_ROOT:-runs/scan_variants_review_ft_fair_pcxyz2k_dist_fps_v2}"
D2_LOG_ROOT="${D2_LOG_ROOT:-logs/finetune/scan_variants_review_ft_fair_pcxyz2k_dist_fps_v2}"

echo "[stage-d2] dist-enabled v2-style (${VARIANT})"
d2_done="$(count_done_variant "${D2_RUN_ROOT}" "${VARIANT}")"
echo "[stage-d2] done=${d2_done}/${EXPECTED}"
if [ "${d2_done}" -lt "${EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    "PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN_D2:-random}" \
    "PT_SAMPLE_MODE_EVAL=${PT_SAMPLE_MODE_EVAL_D2:-fps}" \
    VARIANTS="${VARIANT}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${D2_RUN_ROOT}" \
    BASE_LOG_ROOT="${D2_LOG_ROOT}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
d2_done="$(count_done_variant "${D2_RUN_ROOT}" "${VARIANT}")"
echo "[stage-d2] done=${d2_done}/${EXPECTED}"
if [ "${d2_done}" -lt "${EXPECTED}" ]; then
  echo "[error] stage-d2 incomplete"
  exit 1
fi

echo "[done] dist-enabled v1/v2 obj_bg chain completed"

