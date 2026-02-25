#!/bin/bash
set -euo pipefail

# Submit A/B pretrain jobs with RFPS + mild geometric augmentation.
#
# Default target:
#   - Run A (NEPA-full pretrain setting)
#   - Run B (SOTA-fair pretrain setting)
# with:
#   PT_SAMPLE_MODE_TRAIN=rfps
#   aug_rotate_z=1
#   aug_scale in [0.8, 1.25]
#   aug_translate=0.0
#   aug_jitter_sigma=0.01
#   aug_jitter_clip=0.05
#
# Example:
#   bash scripts/pretrain/submit_pretrain_ab_rfps_aug_qf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_multinode_pbsdsh.sh"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"

NODES_PER_RUN="${NODES_PER_RUN:-2}"
WALLTIME="${WALLTIME:-48:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
SEED_BASE="${SEED_BASE:-300}"
MAX_LEN_AB="${MAX_LEN_AB:-4500}"
DROP_PATH="${DROP_PATH:-0.0}"
PT_RFPS_M="${PT_RFPS_M:-4096}"

AUG_ROTATE_Z="${AUG_ROTATE_Z:-1}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-0.8}"
AUG_SCALE_MAX="${AUG_SCALE_MAX:-1.25}"
AUG_TRANSLATE="${AUG_TRANSLATE:-0.0}"
AUG_JITTER_SIGMA="${AUG_JITTER_SIGMA:-0.01}"
AUG_JITTER_CLIP="${AUG_JITTER_CLIP:-0.05}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-0}"

RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-runs/pretrain_ab_1024_rfps_aug_${RUN_SET}}"
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
  cmd=(
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

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},N_POINT=1024,MAX_LEN=${MAX_LEN_AB},PT_SAMPLE_MODE_TRAIN=rfps,PT_FPS_KEY=auto,PT_RFPS_M=${PT_RFPS_M},DROP_PATH=${DROP_PATH},AUG_ROTATE_Z=${AUG_ROTATE_Z},AUG_SCALE_MIN=${AUG_SCALE_MIN},AUG_SCALE_MAX=${AUG_SCALE_MAX},AUG_TRANSLATE=${AUG_TRANSLATE},AUG_JITTER_SIGMA=${AUG_JITTER_SIGMA},AUG_JITTER_CLIP=${AUG_JITTER_CLIP},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST}"

submit "runA_rfps_aug_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+0)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=1024,QA_TOKENS=1,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=${SAVE_ROOT}_runA"

submit "runB_rfps_aug_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+1)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml,N_RAY=0,QA_TOKENS=0,PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=${SAVE_ROOT}_runB"

echo "[done] submitted A/B RFPS+aug pretrain jobs"
