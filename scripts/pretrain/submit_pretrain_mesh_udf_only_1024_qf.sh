#!/bin/bash
set -euo pipefail

# Submit ShapeNet mesh+UDF-only 1024 pretrain jobs.
# Default is A-profile only; optional C/D can be enabled via RUN_IDS.
#
# Examples:
#   bash scripts/pretrain/submit_pretrain_mesh_udf_only_1024_qf.sh
#   RUN_IDS=A,C,D bash scripts/pretrain/submit_pretrain_mesh_udf_only_1024_qf.sh

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
SEED_BASE="${SEED_BASE:-400}"
MAX_LEN="${MAX_LEN:-4500}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-fps}"
DROP_PATH="${DROP_PATH:-0.0}"
RUN_SET="${RUN_SET:-meshudf_only1024_$(date +%Y%m%d_%H%M%S)}"
RUN_IDS="${RUN_IDS:-A}"
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

COMMON="NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN},PT_FPS_KEY=auto,N_POINT=1024,DROP_PATH=${DROP_PATH},MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf.yaml"

IFS=',' read -r -a _run_arr <<< "${RUN_IDS}"
_n_submit=0
for _rid in "${_run_arr[@]}"; do
  _rid="$(echo "${_rid}" | xargs | tr '[:lower:]' '[:upper:]')"
  case "${_rid}" in
    A)
      submit "runA_meshudf_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+0)),N_RAY=1024,QA_TOKENS=1,MAX_LEN=${MAX_LEN},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_meshudf_only_1024_${RUN_SET}_runA"
      _n_submit=$((_n_submit + 1))
      ;;
    C)
      submit "runC_meshudf_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+2)),N_RAY=1024,QA_TOKENS=1,MAX_LEN=${MAX_LEN},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=morton,SAVE_DIR=runs/pretrain_meshudf_only_1024_${RUN_SET}_runC"
      _n_submit=$((_n_submit + 1))
      ;;
    D)
      submit "runD_meshudf_${RUN_SET}" "${COMMON},SEED=$((SEED_BASE+3)),N_RAY=0,QA_TOKENS=1,MAX_LEN=${MAX_LEN},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,POINT_ORDER_MODE=fps,SAVE_DIR=runs/pretrain_meshudf_only_1024_${RUN_SET}_runD"
      _n_submit=$((_n_submit + 1))
      ;;
    "")
      ;;
    *)
      echo "[error] unknown RUN_IDS entry: ${_rid} (use comma-separated A,C,D)"
      exit 2
      ;;
  esac
done

if [[ "${_n_submit}" -le 0 ]]; then
  echo "[error] no runs selected by RUN_IDS=${RUN_IDS}"
  exit 3
fi

echo "[done] submitted ${_n_submit} mesh+udf-only 1024 pretrain jobs"
