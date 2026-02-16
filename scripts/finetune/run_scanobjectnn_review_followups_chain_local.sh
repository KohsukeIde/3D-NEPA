#!/usr/bin/env bash
set -eu

# Follow-up chain for reviewer-facing ScanObjectNN experiments (resume-safe).
#
# Stage A: K=1 seed expansion on unstable variants (obj_only, pb_t50_rs)
# Stage B: dist ablation (zero pt_dist) on obj_bg at K=0/20
# Stage C: QA+dualmask checkpoint spot-check on obj_bg at K=0/20
#
# Notes:
# - Keeps N_POINT unchanged (default 256).
# - Uses pointcloud_noray + N_RAY=0 for paper-safe classification.

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

# 0) Wait current pipelines if requested.
WAIT_PID_FILES="${WAIT_PID_FILES:-logs/finetune/scan_variants_review_chain/pipeline.pid logs/finetune/modelnet40_pointgpt_protocol/pipeline.pid}"
for f in ${WAIT_PID_FILES}; do
  wait_for_pid_file "${f}" "$(basename "$(dirname "${f}")")"
done

METHODS_ALL="${METHODS_ALL:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"

COMMON_ENV=(
  "BACKEND=${BACKEND:-pointcloud_noray}"
  "N_RAY=${N_RAY:-0}"
  "N_POINT=${N_POINT:-256}"
)

# Stage A: K=1 seed expansion.
echo "[stageA] K=1 seed expansion (obj_only, pb_t50_rs; seeds 3..9)"
env \
  "${COMMON_ENV[@]}" \
  VARIANTS="${STAGEA_VARIANTS:-obj_only pb_t50_rs}" \
  METHODS="${STAGEA_METHODS:-${METHODS_ALL}}" \
  SEEDS="${STAGEA_SEEDS:-3 4 5 6 7 8 9}" \
  K_LIST="${STAGEA_K_LIST:-1}" \
  BASE_RUN_ROOT="${STAGEA_BASE_RUN_ROOT:-runs/scan_variants_review_ft_nray0}" \
  BASE_LOG_ROOT="${STAGEA_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_k1_seedexp}" \
  bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh

# Stage B: dist ablation (xyz only).
echo "[stageB] dist ablation (obj_bg; shapenet_mix_nepa; K=0/20)"
env \
  "${COMMON_ENV[@]}" \
  VARIANTS="${STAGEB_VARIANTS:-obj_bg}" \
  METHODS="${STAGEB_METHODS:-shapenet_mix_nepa}" \
  SEEDS="${STAGEB_SEEDS:-0 1 2}" \
  K_LIST="${STAGEB_K_LIST:-0 20}" \
  ABLATE_POINT_DIST=1 \
  RUN_SUFFIX="${STAGEB_RUN_SUFFIX:-_ablate_dist}" \
  BASE_RUN_ROOT="${STAGEB_BASE_RUN_ROOT:-runs/scan_variants_review_ft_nray0}" \
  BASE_LOG_ROOT="${STAGEB_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_distablate}" \
  bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh

# Stage C: QA+dualmask pretrain checkpoint check.
QA_DUAL_CKPT="${STAGEC_QA_DUAL_CKPT:-runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt}"
if [ ! -f "${QA_DUAL_CKPT}" ]; then
  echo "[error] missing QA+dualmask checkpoint: ${QA_DUAL_CKPT}"
  exit 1
fi
echo "[stageC] QA+dualmask checkpoint check (obj_bg; shapenet_mix_nepa; K=0/20)"
env \
  "${COMMON_ENV[@]}" \
  VARIANTS="${STAGEC_VARIANTS:-obj_bg}" \
  METHODS="${STAGEC_METHODS:-shapenet_mix_nepa}" \
  SEEDS="${STAGEC_SEEDS:-0 1 2}" \
  K_LIST="${STAGEC_K_LIST:-0 20}" \
  SHAPENET_MIX_NEPA_CKPT="${QA_DUAL_CKPT}" \
  RUN_SUFFIX="${STAGEC_RUN_SUFFIX:-_qa_dualmask}" \
  BASE_RUN_ROOT="${STAGEC_BASE_RUN_ROOT:-runs/scan_variants_review_ft_nray0}" \
  BASE_LOG_ROOT="${STAGEC_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_qa_dualmask}" \
  bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh

echo "[done] scanobjectnn review follow-ups completed"
