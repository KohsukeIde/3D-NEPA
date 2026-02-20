#!/usr/bin/env bash
set -eu

# ScanObjectNN fair-protocol FT chain (no linear probe)
#
# Defaults are aligned to reviewer feedback / recent protocols:
# - point-only input from observed surface points: pt_xyz_key=pc_xyz
# - xyz-only channel for fair baseline: --ablate_point_dist
# - bidirectional attention, pointcloud_noray
# - n_point=2048 with allow_scale_up=1
# - simple rotation augmentation preset for ScanObjectNN
# - vote disabled by default (mc_eval_k_test=1)

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

count_done_variants() {
  local run_root="$1"
  shift
  local total=0
  local v
  for v in "$@"; do
    local c
    c="$(find "${run_root}/${v}" -type f -name last.pt 2>/dev/null | wc -l)"
    total=$((total + c))
  done
  echo "${total}"
}

WAIT_PID_FILES="${WAIT_PID_FILES:-logs/finetune/scan_variants_review_poolfix_chain/pipeline.pid logs/finetune/scan_variants_review_followups_chain_bidir/pipeline.pid}"
for f in ${WAIT_PID_FILES}; do
  wait_for_pid_file "${f}" "$(basename "$(dirname "${f}")")"
done

METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"

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
  "PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN:-random}"
  "PT_SAMPLE_MODE_EVAL=${PT_SAMPLE_MODE_EVAL:-fps}"
  "PT_FPS_KEY=${PT_FPS_KEY:-auto}"
  "PT_RFPS_M=${PT_RFPS_M:-4096}"
  "ABLATE_POINT_DIST=${ABLATE_POINT_DIST:-1}"
  "AUG_PRESET=${AUG_PRESET:-scanobjectnn}"
  "AUG_ROTATE_Z=${AUG_ROTATE_Z:-0}"
  "AUG_EVAL=${AUG_EVAL:-0}"
)

FT_BASE_RUN_ROOT="${FT_BASE_RUN_ROOT:-runs/scan_variants_review_ft_fair_pcxyz2k_v1}"
FT_BASE_LOG_ROOT="${FT_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_fair_pcxyz2k_v1}"

STAGE1_VARIANTS="${STAGE1_VARIANTS:-obj_bg}"
STAGE2_VARIANTS="${STAGE2_VARIANTS:-obj_only pb_t50_rs}"

STAGE1_EXPECTED="${STAGE1_EXPECTED:-75}"   # 1 variant x 5 methods x 5 K x 3 seeds
STAGE2_EXPECTED="${STAGE2_EXPECTED:-150}"  # 2 variants x 5 methods x 5 K x 3 seeds

echo "[stage1] full FT fair protocol (obj_bg first)"
done1="$(count_done_variants "${FT_BASE_RUN_ROOT}" ${STAGE1_VARIANTS})"
echo "[stage1] done=${done1}/${STAGE1_EXPECTED}"
if [ "${done1}" -lt "${STAGE1_EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    VARIANTS="${STAGE1_VARIANTS}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${FT_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${FT_BASE_LOG_ROOT}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
done1="$(count_done_variants "${FT_BASE_RUN_ROOT}" ${STAGE1_VARIANTS})"
echo "[stage1] done=${done1}/${STAGE1_EXPECTED}"
if [ "${done1}" -lt "${STAGE1_EXPECTED}" ]; then
  echo "[error] stage1 incomplete"
  exit 1
fi

echo "[stage2] full FT fair protocol (obj_only + pb_t50_rs)"
done2="$(count_done_variants "${FT_BASE_RUN_ROOT}" ${STAGE2_VARIANTS})"
echo "[stage2] done=${done2}/${STAGE2_EXPECTED}"
if [ "${done2}" -lt "${STAGE2_EXPECTED}" ]; then
  env \
    "${COMMON_ENV[@]}" \
    VARIANTS="${STAGE2_VARIANTS}" \
    METHODS="${METHODS}" \
    BASE_RUN_ROOT="${FT_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${FT_BASE_LOG_ROOT}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
fi
done2="$(count_done_variants "${FT_BASE_RUN_ROOT}" ${STAGE2_VARIANTS})"
echo "[stage2] done=${done2}/${STAGE2_EXPECTED}"
if [ "${done2}" -lt "${STAGE2_EXPECTED}" ]; then
  echo "[error] stage2 incomplete"
  exit 1
fi

echo "[done] fair-protocol FT chain completed"
