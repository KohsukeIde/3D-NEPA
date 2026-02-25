#!/bin/bash
set -euo pipefail

# Submit compact A/B RFPS pretrain jobs for dual-mask ON/OFF comparison.
#
# Matrix:
#   - runs: A, B
#   - dual-mask: off, on
#   - points/rays: 256 base (A: n_point=256,n_ray=256; B: n_point=256,n_ray=0)
#
# Usage:
#   bash scripts/pretrain/submit_pretrain_ab_rfps_aug_dualmask256_qf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_multinode_pbsdsh.sh"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"

NODES_PER_RUN="${NODES_PER_RUN:-2}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
SEED_BASE="${SEED_BASE:-700}"
MAX_LEN_256="${MAX_LEN_256:-1200}"
DROP_PATH="${DROP_PATH:-0.0}"
PT_RFPS_M="${PT_RFPS_M:-1024}"
QA_LAYOUT="${QA_LAYOUT:-interleave}"
SEQUENCE_MODE="${SEQUENCE_MODE:-block}"
EVENT_ORDER_MODE="${EVENT_ORDER_MODE:-morton}"
RAY_ORDER_MODE="${RAY_ORDER_MODE:-theta_phi}"
RAY_ANCHOR_MISS_T="${RAY_ANCHOR_MISS_T:-4.0}"
RAY_VIEW_TOL="${RAY_VIEW_TOL:-1e-6}"
TYPE_SPECIFIC_POS="${TYPE_SPECIFIC_POS:-0}"

AUG_ROTATE_Z="${AUG_ROTATE_Z:-1}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-0.8}"
AUG_SCALE_MAX="${AUG_SCALE_MAX:-1.25}"
AUG_TRANSLATE="${AUG_TRANSLATE:-0.0}"
AUG_JITTER_SIGMA="${AUG_JITTER_SIGMA:-0.01}"
AUG_JITTER_CLIP="${AUG_JITTER_CLIP:-0.05}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-0}"

DUAL_MASK_NEAR_ON="${DUAL_MASK_NEAR_ON:-0.4}"
DUAL_MASK_FAR_ON="${DUAL_MASK_FAR_ON:-0.1}"
DUAL_MASK_WINDOW_ON="${DUAL_MASK_WINDOW_ON:-32}"
DUAL_MASK_WARMUP_FRAC_ON="${DUAL_MASK_WARMUP_FRAC_ON:-0.05}"
DUAL_MASK_TYPE_AWARE_ON="${DUAL_MASK_TYPE_AWARE_ON:-0}"
DUAL_MASK_WINDOW_SCALE_ON="${DUAL_MASK_WINDOW_SCALE_ON:-linear}"
DUAL_MASK_WINDOW_REF_TOTAL_ON="${DUAL_MASK_WINDOW_REF_TOTAL_ON:--1}"

RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-runs/pretrain_ab_256_rfps_aug_dualmask_${RUN_SET}}"
QSUB_DEPEND="${QSUB_DEPEND:-}"
JOB_IDS_OUT="${JOB_IDS_OUT:-}"

if [[ -n "${JOB_IDS_OUT}" ]]; then
  mkdir -p "$(dirname "${JOB_IDS_OUT}")"
  : > "${JOB_IDS_OUT}"
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found in PATH."
  exit 1
fi

submit() {
  local run_id="$1"
  local vars="$2"
  echo "[submit] ${run_id}"
  local cmd=(
    qsub
    -l "rt_QF=${NODES_PER_RUN}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "nepa3d_${run_id}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_id},${vars}"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  local job_id
  job_id="$("${cmd[@]}")"
  echo "[job] ${run_id} ${job_id}"
  if [[ -n "${JOB_IDS_OUT}" ]]; then
    echo "${run_id} ${job_id}" >> "${JOB_IDS_OUT}"
  fi
}

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},N_POINT=256,MAX_LEN=${MAX_LEN_256},PT_SAMPLE_MODE_TRAIN=rfps,PT_FPS_KEY=auto,PT_RFPS_M=${PT_RFPS_M},QA_LAYOUT=${QA_LAYOUT},SEQUENCE_MODE=${SEQUENCE_MODE},EVENT_ORDER_MODE=${EVENT_ORDER_MODE},RAY_ORDER_MODE=${RAY_ORDER_MODE},RAY_ANCHOR_MISS_T=${RAY_ANCHOR_MISS_T},RAY_VIEW_TOL=${RAY_VIEW_TOL},TYPE_SPECIFIC_POS=${TYPE_SPECIFIC_POS},DROP_PATH=${DROP_PATH},AUG_ROTATE_Z=${AUG_ROTATE_Z},AUG_SCALE_MIN=${AUG_SCALE_MIN},AUG_SCALE_MAX=${AUG_SCALE_MAX},AUG_TRANSLATE=${AUG_TRANSLATE},AUG_JITTER_SIGMA=${AUG_JITTER_SIGMA},AUG_JITTER_CLIP=${AUG_JITTER_CLIP},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST}"
DM_OFF="DUAL_MASK_NEAR=0.0,DUAL_MASK_FAR=0.0,DUAL_MASK_WINDOW=${DUAL_MASK_WINDOW_ON},DUAL_MASK_WARMUP_FRAC=${DUAL_MASK_WARMUP_FRAC_ON},DUAL_MASK_TYPE_AWARE=${DUAL_MASK_TYPE_AWARE_ON},DUAL_MASK_WINDOW_SCALE=${DUAL_MASK_WINDOW_SCALE_ON},DUAL_MASK_WINDOW_REF_TOTAL=${DUAL_MASK_WINDOW_REF_TOTAL_ON}"
DM_ON="DUAL_MASK_NEAR=${DUAL_MASK_NEAR_ON},DUAL_MASK_FAR=${DUAL_MASK_FAR_ON},DUAL_MASK_WINDOW=${DUAL_MASK_WINDOW_ON},DUAL_MASK_WARMUP_FRAC=${DUAL_MASK_WARMUP_FRAC_ON},DUAL_MASK_TYPE_AWARE=${DUAL_MASK_TYPE_AWARE_ON},DUAL_MASK_WINDOW_SCALE=${DUAL_MASK_WINDOW_SCALE_ON},DUAL_MASK_WINDOW_REF_TOTAL=${DUAL_MASK_WINDOW_REF_TOTAL_ON}"

# Run A: NEPA-full style pretrain inputs.
submit "runA_rfps256_dmoff_${RUN_SET}" "${COMMON},${DM_OFF},SEED=$((SEED_BASE+0)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=256,QA_TOKENS=1,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=${SAVE_ROOT}_runA_dmoff"
submit "runA_rfps256_dmon_${RUN_SET}"  "${COMMON},${DM_ON},SEED=$((SEED_BASE+1)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=256,QA_TOKENS=1,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=${SAVE_ROOT}_runA_dmon"

# Run B: SOTA-fair style pretrain inputs.
submit "runB_rfps256_dmoff_${RUN_SET}" "${COMMON},${DM_OFF},SEED=$((SEED_BASE+2)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml,N_RAY=0,QA_TOKENS=0,PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=${SAVE_ROOT}_runB_dmoff"
submit "runB_rfps256_dmon_${RUN_SET}"  "${COMMON},${DM_ON},SEED=$((SEED_BASE+3)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml,N_RAY=0,QA_TOKENS=0,PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=${SAVE_ROOT}_runB_dmon"

echo "[done] submitted A/B 256 RFPS+aug dual-mask ON/OFF pretrain jobs"
