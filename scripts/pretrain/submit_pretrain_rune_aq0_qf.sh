#!/bin/bash
set -euo pipefail

# Submit two focused pretrain runs:
#   - Run E: Run A settings + point_order_mode=random
#   - Run A_q0: Run A settings + qa_tokens=0
#
# Example:
#   NODES_PER_RUN=2 WALLTIME=48:00:00 BATCH=16 EPOCHS=100 \
#   bash scripts/pretrain/submit_pretrain_rune_aq0_qf.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_multinode_pbsdsh.sh"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
NODES_PER_RUN="${NODES_PER_RUN:-2}"
WALLTIME="${WALLTIME:-48:00:00}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
SEED_BASE="${SEED_BASE:-500}"
MAX_LEN="${MAX_LEN:-4500}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_ROOT="${SAVE_ROOT:-runs/pretrain_rune_aq0_1024_${STAMP}}"
QSUB_DEPEND="${QSUB_DEPEND:-}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"

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
    -N "nepa3d_${run_id}"
    -v "RUN_TAG=${run_id},${vars}"
  )
  if [[ -n "${GROUP_LIST}" ]]; then
    cmd+=( -W "group_list=${GROUP_LIST}" )
  fi
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

COMMON="WORKDIR=${WORKDIR},NUM_WORKERS=${NUM_WORKERS},BATCH=${BATCH},EPOCHS=${EPOCHS},LR=${LR},MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_POINT=1024,N_RAY=1024,MAX_LEN=${MAX_LEN},PT_XYZ_KEY=pt_xyz_pool,PT_DIST_KEY=pt_dist_pool,ABLATE_POINT_DIST=0,PT_SAMPLE_MODE_TRAIN=fps,PT_FPS_KEY=auto"

# Run E: order ablation against A/C (random token order)
submit "runE" "${COMMON},SEED=$((SEED_BASE+0)),QA_TOKENS=1,POINT_ORDER_MODE=random,SAVE_DIR=${SAVE_ROOT}_runE"

# Run A_q0: qa_tokens ablation against Run A (keep A-style ordering)
submit "runAq0" "${COMMON},SEED=$((SEED_BASE+1)),QA_TOKENS=0,POINT_ORDER_MODE=fps,SAVE_DIR=${SAVE_ROOT}_runAq0"

echo "[done] submitted Run E and Run A_q0"
