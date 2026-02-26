#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

ts() { date +"%F %T"; }

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/analysis/pending_feedback_all_chain_${RUN_ID}}"
mkdir -p "${LOG_DIR}"

echo "[$(ts)] [stage0] pending-feedback all-chain start (run_id=${RUN_ID})"

###############################################################################
# Stage-1: encdec non-collapse / topo-coord recheck with fresh pretrains
###############################################################################
echo "[$(ts)] [stage1] encdec fresh pair (proj vs bbox) start"

MODEL1_TAG="${MODEL1_TAG:-nepa_impl_encdec_plusgut_projfresh2_s0_e50}"
MODEL1_SAVE_DIR="${MODEL1_SAVE_DIR:-runs/eccv_upmix_nepa_impl_encdec_plusgut_projfresh2_s0}"
MODEL2_TAG="${MODEL2_TAG:-nepa_impl_encdec_plusgut_bboxfresh2_s0_e50}"
MODEL2_SAVE_DIR="${MODEL2_SAVE_DIR:-runs/eccv_upmix_nepa_impl_encdec_plusgut_bboxfresh2_s0}"

env \
  MODEL1_TAG="${MODEL1_TAG}" \
  MODEL1_GPU="${MODEL1_GPU:-0}" \
  MODEL1_ARCH="encdec" \
  MODEL1_QA_LAYOUT="split" \
  MODEL1_TOPO_K="${MODEL1_TOPO_K:-16}" \
  MODEL1_TOPO_RAY_COORD="proj" \
  MODEL1_TOPO_RAY_BBOX="${MODEL1_TOPO_RAY_BBOX:-0.5}" \
  MODEL1_SAVE_DIR="${MODEL1_SAVE_DIR}" \
  MODEL2_TAG="${MODEL2_TAG}" \
  MODEL2_GPU="${MODEL2_GPU:-1}" \
  MODEL2_ARCH="encdec" \
  MODEL2_QA_LAYOUT="split" \
  MODEL2_TOPO_K="${MODEL2_TOPO_K:-16}" \
  MODEL2_TOPO_RAY_COORD="bbox" \
  MODEL2_TOPO_RAY_BBOX="${MODEL2_TOPO_RAY_BBOX:-0.5}" \
  MODEL2_SAVE_DIR="${MODEL2_SAVE_DIR}" \
  INCLUDE_PT_GRAD="${INCLUDE_PT_GRAD:-1}" \
  PT_GRAD_MODE="${PT_GRAD_MODE:-log}" \
  PT_GRAD_EPS="${PT_GRAD_EPS:-1e-3}" \
  PT_GRAD_CLIP="${PT_GRAD_CLIP:-10.0}" \
  PT_GRAD_ORIENT="${PT_GRAD_ORIENT:-ray}" \
  INCLUDE_RAY_UNC="${INCLUDE_RAY_UNC:-1}" \
  RAY_UNC_K="${RAY_UNC_K:-8}" \
  RAY_UNC_MODE="${RAY_UNC_MODE:-normal_var}" \
  bash scripts/analysis/run_impl_update_chain_local.sh \
  2>&1 | tee "${LOG_DIR}/stage1_impl_update_encdec_pair.log"

echo "[$(ts)] [stage1] done"

###############################################################################
# Stage-2: ScanObjectNN 2048/FPS re-optimization on available cache
###############################################################################
echo "[$(ts)] [stage2] scanobjectnn 2048/FPS chain start"

CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_cache_v2}"
MIX_NEPA_CKPT="${MIX_NEPA_CKPT:-runs/eccv_upmix_nepa_s0/ckpt_ep049.pt}"
MIX_MAE_CKPT="${MIX_MAE_CKPT:-runs/eccv_upmix_mae_s0/ckpt_ep049.pt}"

if [ ! -d "${CACHE_ROOT}" ]; then
  echo "[$(ts)] [stage2][error] missing cache root: ${CACHE_ROOT}"
  exit 1
fi
if [ ! -f "${MIX_NEPA_CKPT}" ]; then
  echo "[$(ts)] [stage2][error] missing ckpt: ${MIX_NEPA_CKPT}"
  exit 1
fi
if [ ! -f "${MIX_MAE_CKPT}" ]; then
  echo "[$(ts)] [stage2][error] missing ckpt: ${MIX_MAE_CKPT}"
  exit 1
fi

env \
  CACHE_ROOT="${CACHE_ROOT}" \
  BACKEND="pointcloud_noray" \
  METHODS="${METHODS:-scratch shapenet_mix_nepa shapenet_mix_mae}" \
  SHAPENET_MIX_NEPA_CKPT="${MIX_NEPA_CKPT}" \
  SHAPENET_MIX_MAE_CKPT="${MIX_MAE_CKPT}" \
  SEEDS="${SEEDS:-0 1 2}" \
  K_LIST="${K_LIST:-0 1 5 10 20}" \
  N_POINT="${N_POINT:-2048}" \
  N_RAY="${N_RAY:-0}" \
  ALLOW_SCALE_UP="${ALLOW_SCALE_UP:-1}" \
  CLS_IS_CAUSAL="${CLS_IS_CAUSAL:-0}" \
  CLS_POOLING="${CLS_POOLING:-mean_pts}" \
  PT_XYZ_KEY="${PT_XYZ_KEY:-pc_xyz}" \
  PT_DIST_KEY="${PT_DIST_KEY:-pt_dist_pool}" \
  PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}" \
  PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}" \
  PT_FPS_KEY="${PT_FPS_KEY:-auto}" \
  PT_RFPS_M="${PT_RFPS_M:-4096}" \
  ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}" \
  MC_EVAL_K="${MC_EVAL_K:-1}" \
  MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}" \
  MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-1}" \
  AUG_PRESET="${AUG_PRESET:-scanobjectnn}" \
  AUG_ROTATE_Z="${AUG_ROTATE_Z:-0}" \
  AUG_EVAL="${AUG_EVAL:-0}" \
  RUN_ROOT="${RUN_ROOT:-runs/scan_variants_review_ft_fair_pcxyz2k_cachev2_v1}" \
  LOG_ROOT="${FT_LOG_ROOT:-logs/finetune/scan_variants_review_ft_fair_pcxyz2k_cachev2_v1/jobs}" \
  bash scripts/finetune/run_scanobjectnn_m1_table_local.sh \
  2>&1 | tee "${LOG_DIR}/stage2_scanobjectnn_2048fps.log"

echo "[$(ts)] [stage2] done"

echo "[$(ts)] [done] pending-feedback all-chain completed"

