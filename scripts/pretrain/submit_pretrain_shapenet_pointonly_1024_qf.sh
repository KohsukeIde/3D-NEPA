#!/bin/bash
set -euo pipefail

# Submit ShapeNet pointcloud-only 1024 pretrain jobs.
# Default is B-profile only (XYZ-only).
#
# Example:
#   bash scripts/pretrain/submit_pretrain_shapenet_pointonly_1024_qf.sh

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
SEED_BASE="${SEED_BASE:-500}"
MAX_LEN="${MAX_LEN:-2500}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-fps}"
DROP_PATH="${DROP_PATH:-0.0}"
RUN_SET="${RUN_SET:-shapenet_pointonly1024_$(date +%Y%m%d_%H%M%S)}"
RUN_IDS="${RUN_IDS:-B}"
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

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN},PT_FPS_KEY=auto,N_POINT=1024,DROP_PATH=${DROP_PATH},MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only.yaml"

IFS=',' read -r -a _run_arr <<< "${RUN_IDS}"
_n_submit=0
for _rid in "${_run_arr[@]}"; do
  _rid="$(echo "${_rid}" | xargs | tr '[:lower:]' '[:upper:]')"
  case "${_rid}" in
    B)
      submit "runB_shapenetpc_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+1)),N_RAY=0,QA_TOKENS=0,MAX_LEN=${MAX_LEN},PT_XYZ_KEY=pc_xyz,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=1,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_shapenet_pointonly_1024_${RUN_SET}_runB"
      _n_submit=$((_n_submit + 1))
      ;;
    "")
      ;;
    *)
      echo "[error] unknown RUN_IDS entry: ${_rid} (use comma-separated B)"
      exit 2
      ;;
  esac
done

if [[ "${_n_submit}" -le 0 ]]; then
  echo "[error] no runs selected by RUN_IDS=${RUN_IDS}"
  exit 3
fi

echo "[done] submitted ${_n_submit} ShapeNet pointcloud-only 1024 pretrain jobs"
