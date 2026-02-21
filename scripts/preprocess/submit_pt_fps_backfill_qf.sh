#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/migrate_pt_fps_order_multinode_pbsdsh.sh"

NODES_PER_JOB="${NODES_PER_JOB:-2}"
WALLTIME="${WALLTIME:-12:00:00}"
WORKERS="${WORKERS:-32}"
FPS_K="${FPS_K:-2048}"
# Keep SPLITS default in migrate script to avoid commas in qsub -v payload.

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found"
  exit 1
fi

echo "[submit] shapenet pt_fps_order"
QID1=$(qsub -l "rt_QF=${NODES_PER_JOB}" -l "walltime=${WALLTIME}" \
  -N "ptfps_shapenet" \
  -v "RUN_TAG=shapenet,CACHE_ROOT=data/shapenet_cache_v0,PT_KEY=pt_xyz_pool,OUT_KEY=pt_fps_order,FPS_K=${FPS_K},WORKERS=${WORKERS}" \
  "${SCRIPT}")
echo "${QID1}"

echo "[submit] scanobjectnn pt_fps_order"
QID2=$(qsub -l "rt_QF=${NODES_PER_JOB}" -l "walltime=${WALLTIME}" \
  -N "ptfps_scanmain" \
  -v "RUN_TAG=scanmain,CACHE_ROOT=data/scanobjectnn_main_split_v2,PT_KEY=pt_xyz_pool,OUT_KEY=pt_fps_order,FPS_K=${FPS_K},WORKERS=${WORKERS}" \
  "${SCRIPT}")
echo "${QID2}"

echo "[done] submitted: ${QID1} ${QID2}"
