#!/bin/bash
set -euo pipefail

# Extension matrix after the 12-job operational minimum.
# Target: add uncovered jobs from the 26-job plan.
#
# Added groups (total 20 jobs):
#   1) SOTA-fair A/B fps, LR=5e-4 across 3 variants                      -> 6
#   2) NEPA-full A_fps, LR in {1e-4,5e-4} across 3 variants              -> 6
#   3) SOTA-fair A/B rfps+aug, pb_t50_rs, LR in {1e-4,5e-4}             -> 4
#   4) Aug compare (B_fps, pb_t50_rs, best LR): none vs scanobjectnn     -> 2
#   5) drop_path compare (B_fps, pb_t50_rs, best LR): base vs dp         -> 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_VARIANTS="${SCRIPT_DIR}/submit_sotafair_variants_llrd_droppath_ablation_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -x "${SUBMIT_VARIANTS}" ]]; then
  echo "[error] missing executable: ${SUBMIT_VARIANTS}"
  exit 1
fi

RUN_SET_BASE="${RUN_SET_BASE:-variant_ext_$(date +%Y%m%d_%H%M%S)}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

# Common eval controls
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-0}"
RUN_CPAC="${RUN_CPAC:-0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
EPOCHS_CLS="${EPOCHS_CLS:-300}"
USE_FC_NORM="${USE_FC_NORM:-1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
AUG_EVAL="${AUG_EVAL:-1}"
VARIANTS_ALL="${VARIANTS_ALL:-obj_bg,obj_only,pb_t50_rs}"

# Checkpoints
CKPT_A_FPS="${CKPT_A_FPS:-${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt}"
CKPT_B_FPS="${CKPT_B_FPS:-${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt}"
CKPT_A_RFPS_AUG="${CKPT_A_RFPS_AUG:-${WORKDIR}/runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runA/last.pt}"
CKPT_B_RFPS_AUG="${CKPT_B_RFPS_AUG:-${WORKDIR}/runs/pretrain_ab_1024_rfps_aug_rfps_aug_ab_20260225_171018_runB/last.pt}"

# Best-setting assumptions for targeted follow-ups
BEST_LR="${BEST_LR:-5e-4}"
BEST_AUG_PRESET="${BEST_AUG_PRESET:-scanobjectnn}"

submit_case() {
  local tag="$1"
  shift
  echo "[submit-case] ${tag}"
  env \
    WORKDIR="${WORKDIR}" \
    QSUB_DEPEND="${QSUB_DEPEND}" \
    RUN_SET_BASE_PREFIX="${RUN_SET_BASE}_${tag}" \
    RUN_SCAN="${RUN_SCAN}" \
    RUN_MODELNET="${RUN_MODELNET}" \
    RUN_CPAC="${RUN_CPAC}" \
    VAL_SPLIT_MODE="${VAL_SPLIT_MODE}" \
    EPOCHS_CLS="${EPOCHS_CLS}" \
    USE_FC_NORM="${USE_FC_NORM}" \
    LABEL_SMOOTHING="${LABEL_SMOOTHING}" \
    AUG_EVAL="${AUG_EVAL}" \
    "$@" \
    bash "${SUBMIT_VARIANTS}"
}

# 1) SOTA-fair A/B fps, LR=5e-4 across 3 variants -> 6
submit_case "sotafair_ab_fps_lr5e4" \
  VARIANTS="${VARIANTS_ALL}" \
  RUN_IDS="A,B" \
  ABLATIONS="base" \
  LR_CLS="5e-4" \
  CKPT_RUNA="${CKPT_A_FPS}" \
  CKPT_RUNB="${CKPT_B_FPS}"

# 2) NEPA-full A_fps, LR=1e-4 across 3 variants -> 3
submit_case "nepafull_a_fps_lr1e4" \
  VARIANTS="${VARIANTS_ALL}" \
  RUN_IDS="A" \
  ABLATIONS="base" \
  LR_CLS="1e-4" \
  CKPT_RUNA="${CKPT_A_FPS}" \
  PT_XYZ_KEY_CLS="pt_xyz_pool" \
  PT_DIST_KEY_CLS="pt_dist_pool" \
  ABLATE_POINT_DIST="0" \
  POINT_ORDER_MODE="fps"

# 2) NEPA-full A_fps, LR=5e-4 across 3 variants -> 3
submit_case "nepafull_a_fps_lr5e4" \
  VARIANTS="${VARIANTS_ALL}" \
  RUN_IDS="A" \
  ABLATIONS="base" \
  LR_CLS="5e-4" \
  CKPT_RUNA="${CKPT_A_FPS}" \
  PT_XYZ_KEY_CLS="pt_xyz_pool" \
  PT_DIST_KEY_CLS="pt_dist_pool" \
  ABLATE_POINT_DIST="0" \
  POINT_ORDER_MODE="fps"

# 3) SOTA-fair A/B rfps+aug, pb_t50_rs, LR=1e-4 -> 2
submit_case "sotafair_ab_rfpsaug_pb_lr1e4" \
  VARIANTS="pb_t50_rs" \
  RUN_IDS="A,B" \
  ABLATIONS="base" \
  LR_CLS="1e-4" \
  CKPT_RUNA="${CKPT_A_RFPS_AUG}" \
  CKPT_RUNB="${CKPT_B_RFPS_AUG}"

# 3) SOTA-fair A/B rfps+aug, pb_t50_rs, LR=5e-4 -> 2
submit_case "sotafair_ab_rfpsaug_pb_lr5e4" \
  VARIANTS="pb_t50_rs" \
  RUN_IDS="A,B" \
  ABLATIONS="base" \
  LR_CLS="5e-4" \
  CKPT_RUNA="${CKPT_A_RFPS_AUG}" \
  CKPT_RUNB="${CKPT_B_RFPS_AUG}"

# 4) Aug compare (B_fps, pb_t50_rs, best LR): none vs scanobjectnn -> 2
submit_case "sotafair_b_fps_pb_augnone_best" \
  VARIANTS="pb_t50_rs" \
  RUN_IDS="B" \
  ABLATIONS="base" \
  LR_CLS="${BEST_LR}" \
  SCAN_AUG_PRESET="none" \
  CKPT_RUNB="${CKPT_B_FPS}"

submit_case "sotafair_b_fps_pb_augscan_best" \
  VARIANTS="pb_t50_rs" \
  RUN_IDS="B" \
  ABLATIONS="base" \
  LR_CLS="${BEST_LR}" \
  SCAN_AUG_PRESET="scanobjectnn" \
  CKPT_RUNB="${CKPT_B_FPS}"

# 5) drop_path compare (B_fps, pb_t50_rs, best LR): base vs dp -> 2
submit_case "sotafair_b_fps_pb_dp_best" \
  VARIANTS="pb_t50_rs" \
  RUN_IDS="B" \
  ABLATIONS="base,dp" \
  LR_CLS="${BEST_LR}" \
  SCAN_AUG_PRESET="${BEST_AUG_PRESET}" \
  DROP_PATH_ONLY="0.1" \
  CKPT_RUNB="${CKPT_B_FPS}"

echo "[summary] expected_added_jobs=20 (plus existing minimum set)"
