#!/bin/bash
set -euo pipefail

# Submit A/B/C/D pretrain jobs in parallel.
# Example:
#   NODES_PER_RUN=2 WALLTIME=24:00:00 BATCH=16 EPOCHS=100 bash scripts/pretrain/submit_pretrain_abcd_qf.sh

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
SEED_BASE="${SEED_BASE:-100}"
MAX_LEN_ABCD="${MAX_LEN_ABCD:-4500}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-fps}"
DROP_PATH="${DROP_PATH:-0.0}"
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
    -N "nepa3d_${run_id}" \
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_id},${vars}" \
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

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN},PT_FPS_KEY=auto,N_POINT=1024,DROP_PATH=${DROP_PATH}"

submit "runA" "${COMMON},SEED=$((SEED_BASE+0)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=1024,QA_TOKENS=1,MAX_LEN=${MAX_LEN_ABCD},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_abcd_1024_runA"

submit "runB" "${COMMON},SEED=$((SEED_BASE+1)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml,N_RAY=0,QA_TOKENS=0,MAX_LEN=${MAX_LEN_ABCD},PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_abcd_1024_runB"

submit "runC" "${COMMON},SEED=$((SEED_BASE+2)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=1024,QA_TOKENS=1,MAX_LEN=${MAX_LEN_ABCD},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_abcd_1024_runC"

submit "runD" "${COMMON},SEED=$((SEED_BASE+3)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=0,QA_TOKENS=1,MAX_LEN=${MAX_LEN_ABCD},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_abcd_1024_runD"

echo "[done] submitted A/B/C/D"
