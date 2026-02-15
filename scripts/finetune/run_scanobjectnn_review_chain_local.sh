#!/usr/bin/env bash
set -eu

# Review-response chain (resume-safe):
#   stage1: full fine-tune tables on ScanObjectNN core3 variants
#   stage2: linear-probe tables on the same settings
#
# Key defaults for reviewer concerns:
#   - uses protocol variants (obj_bg/obj_only/pb_t50_rs)
#   - includes mix methods in comparison
#   - uses N_RAY=0 for pointcloud_noray backend (unless explicitly overridden)

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

count_done() {
  local run_root="$1"
  find "${run_root}" -type f -name last.pt 2>/dev/null | wc -l
}

VARIANTS="${VARIANTS:-obj_bg obj_only pb_t50_rs}"
METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"
EXPECTED_FULL="${EXPECTED_FULL:-225}"  # 3 variants x 5 methods x 5 K x 3 seeds

# Stage1: full fine-tune
FT_BASE_RUN_ROOT="${FT_BASE_RUN_ROOT:-runs/scan_variants_review_ft_nray0}"
FT_BASE_LOG_ROOT="${FT_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_nray0}"
FT_PIPELINE_PID="${FT_BASE_LOG_ROOT}/pipeline.pid"

ft_done="$(count_done "${FT_BASE_RUN_ROOT}")"
echo "[stage1/full-ft] done=${ft_done}/${EXPECTED_FULL}"
if [ "${ft_done}" -lt "${EXPECTED_FULL}" ]; then
  if [ -f "${FT_PIPELINE_PID}" ] && ps -p "$(cat "${FT_PIPELINE_PID}")" >/dev/null 2>&1; then
    echo "[stage1/full-ft] attach existing pid=$(cat "${FT_PIPELINE_PID}")"
  else
    echo "[stage1/full-ft] launch"
    VARIANTS="${VARIANTS}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${FT_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${FT_BASE_LOG_ROOT}" \
    LOG_DIR="${FT_BASE_LOG_ROOT}" \
    BACKEND="${BACKEND:-pointcloud_noray}" \
    N_RAY="${N_RAY:-0}" \
      bash scripts/finetune/launch_scanobjectnn_variant_tables_local.sh
  fi
  wait_for_pid_file "${FT_PIPELINE_PID}" "review_full_ft"
fi

ft_done="$(count_done "${FT_BASE_RUN_ROOT}")"
echo "[stage1/full-ft] done=${ft_done}/${EXPECTED_FULL}"
if [ "${ft_done}" -lt "${EXPECTED_FULL}" ]; then
  echo "[error] stage1 full fine-tune not complete"
  exit 1
fi

# Stage2: linear probe
LP_BASE_RUN_ROOT="${LP_BASE_RUN_ROOT:-runs/scan_variants_review_lp_nray0}"
LP_BASE_LOG_ROOT="${LP_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_lp_nray0}"
LP_PIPELINE_PID="${LP_BASE_LOG_ROOT}/pipeline.pid"

lp_done="$(count_done "${LP_BASE_RUN_ROOT}")"
echo "[stage2/linear-probe] done=${lp_done}/${EXPECTED_FULL}"
if [ "${lp_done}" -lt "${EXPECTED_FULL}" ]; then
  if [ -f "${LP_PIPELINE_PID}" ] && ps -p "$(cat "${LP_PIPELINE_PID}")" >/dev/null 2>&1; then
    echo "[stage2/linear-probe] attach existing pid=$(cat "${LP_PIPELINE_PID}")"
  else
    echo "[stage2/linear-probe] launch"
    VARIANTS="${VARIANTS}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${LP_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${LP_BASE_LOG_ROOT}" \
    LOG_DIR="${LP_BASE_LOG_ROOT}" \
    BACKEND="${BACKEND:-pointcloud_noray}" \
    N_RAY="${N_RAY:-0}" \
    FREEZE_BACKBONE=1 \
    RUN_SUFFIX="${RUN_SUFFIX:-_lp}" \
      bash scripts/finetune/launch_scanobjectnn_variant_tables_local.sh
  fi
  wait_for_pid_file "${LP_PIPELINE_PID}" "review_linear_probe"
fi

lp_done="$(count_done "${LP_BASE_RUN_ROOT}")"
echo "[stage2/linear-probe] done=${lp_done}/${EXPECTED_FULL}"
if [ "${lp_done}" -lt "${EXPECTED_FULL}" ]; then
  echo "[error] stage2 linear probe not complete"
  exit 1
fi

echo "[done] review-response chain completed"
