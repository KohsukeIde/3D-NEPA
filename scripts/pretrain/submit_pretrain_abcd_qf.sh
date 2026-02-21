#!/bin/bash
set -euo pipefail

# Submit A/B/C/D pretrain jobs in parallel.
# Example:
#   NODES_PER_RUN=2 WALLTIME=24:00:00 BATCH=16 EPOCHS=100 bash scripts/pretrain/submit_pretrain_abcd_qf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_multinode_pbsdsh.sh"
NODES_PER_RUN="${NODES_PER_RUN:-2}"
WALLTIME="${WALLTIME:-48:00:00}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
SEED_BASE="${SEED_BASE:-100}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

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
    -N "nepa3d_${run_id}" \
    -v "RUN_TAG=${run_id},${vars}" \
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},PT_SAMPLE_MODE_TRAIN=fps,PT_FPS_KEY=auto,N_POINT=1024"

submit "runA" "${COMMON},SEED=$((SEED_BASE+0)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=1024,QA_TOKENS=1,MAX_LEN=4500,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_abcd_1024_runA"

submit "runB" "${COMMON},SEED=$((SEED_BASE+1)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_scan_pointcloud_mainsplit.yaml,N_RAY=0,QA_TOKENS=0,MAX_LEN=2500,PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_abcd_1024_runB"

submit "runC" "${COMMON},SEED=$((SEED_BASE+2)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=1024,QA_TOKENS=1,MAX_LEN=4500,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_abcd_1024_runC"

submit "runD" "${COMMON},SEED=$((SEED_BASE+3)),MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_RAY=0,QA_TOKENS=1,MAX_LEN=2500,PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_abcd_1024_runD"

echo "[done] submitted A/B/C/D"
