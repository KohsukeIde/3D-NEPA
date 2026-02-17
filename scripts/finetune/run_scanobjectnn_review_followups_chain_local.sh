#!/usr/bin/env bash
set -eu

# Follow-up chain for reviewer-facing ScanObjectNN experiments (resume-safe).
#
# Stage A: K=1 seed expansion on unstable variants (obj_only, pb_t50_rs)
# Stage B: dist ablation (zero pt_dist) on obj_bg at K=0/20
# Stage C: QA+dualmask checkpoint spot-check on obj_bg (single ablation; default K=20)
#
# Notes:
# - Keeps N_POINT unchanged (default 256).
# - Uses pointcloud_noray + N_RAY=0 for paper-safe classification.
# - Uses vote-10 at test time by default (MC_EVAL_K_TEST=10).

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
# Default includes both current bidirectional chains and older chain names.
WAIT_PID_FILES="${WAIT_PID_FILES:-logs/finetune/scan_variants_review_chain_bidir/pipeline.pid logs/finetune/after_review_modelnet_chain_bidir/pipeline.pid logs/finetune/scan_variants_review_chain/pipeline.pid logs/finetune/modelnet40_pointgpt_protocol/pipeline.pid}"
for f in ${WAIT_PID_FILES}; do
  wait_for_pid_file "${f}" "$(basename "$(dirname "${f}")")"
done

METHODS_ALL="${METHODS_ALL:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"

COMMON_ENV=(
  "BACKEND=${BACKEND:-pointcloud_noray}"
  "N_RAY=${N_RAY:-0}"
  "N_POINT=${N_POINT:-256}"
  "MC_EVAL_K=${MC_EVAL_K:-10}"
  "MC_EVAL_K_VAL=${MC_EVAL_K_VAL:-1}"
  "MC_EVAL_K_TEST=${MC_EVAL_K_TEST:-10}"
  "CLS_IS_CAUSAL=${CLS_IS_CAUSAL:-0}"
  "CLS_POOLING=${CLS_POOLING:-mean_a}"
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
  STAGEC_AUTO_PRETRAIN="${STAGEC_AUTO_PRETRAIN:-1}"
  if [ "${STAGEC_AUTO_PRETRAIN}" != "1" ]; then
    echo "[warn] missing QA+dualmask checkpoint: ${QA_DUAL_CKPT}"
    echo "[warn] skipping stageC (set STAGEC_QA_DUAL_CKPT=<path> or STAGEC_AUTO_PRETRAIN=1 and relaunch)"
    echo "[done] scanobjectnn review follow-ups completed (stageC skipped)"
    exit 0
  fi

  QA_PRETRAIN_SAVE_DIR="${STAGEC_QA_PRETRAIN_SAVE_DIR:-runs/eccv_upmix_nepa_qa_dualmask_s0}"
  QA_PRETRAIN_LOG="${STAGEC_QA_PRETRAIN_LOG:-logs/pretrain/eccv_qa_dualmask/upmix_nepa_qa_dualmask_s0_bs96_resume.log}"
  QA_PRETRAIN_GPU="${STAGEC_QA_PRETRAIN_GPU:-0}"
  QA_PRETRAIN_MIX_CONFIG="${STAGEC_QA_PRETRAIN_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix.yaml}"
  QA_PRETRAIN_MIX_NUM_SAMPLES="${STAGEC_QA_PRETRAIN_MIX_NUM_SAMPLES:-200000}"
  QA_PRETRAIN_MIX_SEED="${STAGEC_QA_PRETRAIN_MIX_SEED:-0}"
  QA_PRETRAIN_EPOCHS="${STAGEC_QA_PRETRAIN_EPOCHS:-50}"
  QA_PRETRAIN_BATCH="${STAGEC_QA_PRETRAIN_BATCH:-96}"
  QA_PRETRAIN_N_POINT="${STAGEC_QA_PRETRAIN_N_POINT:-256}"
  QA_PRETRAIN_N_RAY="${STAGEC_QA_PRETRAIN_N_RAY:-256}"
  QA_PRETRAIN_NUM_WORKERS="${STAGEC_QA_PRETRAIN_NUM_WORKERS:-6}"
  QA_PRETRAIN_SEED="${STAGEC_QA_PRETRAIN_SEED:-0}"

  mkdir -p "$(dirname "${QA_PRETRAIN_LOG}")"
  echo "[stageC] missing checkpoint -> auto pretrain resume"
  echo "[stageC] log=${QA_PRETRAIN_LOG}"
  CUDA_VISIBLE_DEVICES="${QA_PRETRAIN_GPU}" .venv/bin/python -u -m nepa3d.train.pretrain \
    --mix_config "${QA_PRETRAIN_MIX_CONFIG}" \
    --mix_num_samples "${QA_PRETRAIN_MIX_NUM_SAMPLES}" \
    --mix_seed "${QA_PRETRAIN_MIX_SEED}" \
    --objective nepa \
    --qa_tokens 1 \
    --dual_mask_near 0.4 \
    --dual_mask_far 0.1 \
    --dual_mask_window 32 \
    --dual_mask_warmup_frac 0.05 \
    --epochs "${QA_PRETRAIN_EPOCHS}" \
    --batch "${QA_PRETRAIN_BATCH}" \
    --n_point "${QA_PRETRAIN_N_POINT}" \
    --n_ray "${QA_PRETRAIN_N_RAY}" \
    --num_workers "${QA_PRETRAIN_NUM_WORKERS}" \
    --save_every 1 \
    --save_last 1 \
    --auto_resume 1 \
    --resume "${QA_PRETRAIN_SAVE_DIR}/last.pt" \
    --save_dir "${QA_PRETRAIN_SAVE_DIR}" \
    --seed "${QA_PRETRAIN_SEED}" \
    > "${QA_PRETRAIN_LOG}" 2>&1

  QA_DUAL_CKPT="${QA_PRETRAIN_SAVE_DIR}/ckpt_ep049.pt"
  if [ ! -f "${QA_DUAL_CKPT}" ]; then
    echo "[error] stageC auto pretrain finished but checkpoint not found: ${QA_DUAL_CKPT}"
    exit 1
  fi
fi
echo "[stageC] QA+dualmask checkpoint check (single ablation: mean_a vs eos)"
STAGEC_POOLINGS="${STAGEC_POOLINGS:-mean_a eos}"
STAGEC_RUN_SUFFIX_BASE="${STAGEC_RUN_SUFFIX_BASE:-_qa_dualmask}"
for pool in ${STAGEC_POOLINGS}; do
  echo "[stageC] run cls_pooling=${pool}"
  env \
    "${COMMON_ENV[@]}" \
    CLS_POOLING="${pool}" \
    VARIANTS="${STAGEC_VARIANTS:-obj_bg}" \
    METHODS="${STAGEC_METHODS:-shapenet_mix_nepa}" \
    SEEDS="${STAGEC_SEEDS:-0}" \
    K_LIST="${STAGEC_K_LIST:-20}" \
    SHAPENET_MIX_NEPA_CKPT="${QA_DUAL_CKPT}" \
    RUN_SUFFIX="${STAGEC_RUN_SUFFIX_BASE}_${pool}" \
    BASE_RUN_ROOT="${STAGEC_BASE_RUN_ROOT:-runs/scan_variants_review_ft_nray0}" \
    BASE_LOG_ROOT="${STAGEC_BASE_LOG_ROOT:-logs/finetune/scan_variants_review_ft_qa_dualmask}" \
    bash scripts/finetune/run_scanobjectnn_variant_tables_local.sh
done

echo "[done] scanobjectnn review follow-ups completed"
