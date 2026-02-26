#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/analysis/scale_objpres_pair_${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo "[$(date +"%F %T")] launch pair run_id=${RUN_ID}" | tee "${LOG_DIR}/launch.log"

# GPU0: linear scaling
nohup env \
  RUN_TAG="eccv_upmix_nepa_qa_dualmask_scale_objpres_lin_s0_${RUN_ID}" \
  GPU_ID=0 \
  DUAL_MASK_WINDOW_SCALE=linear \
  LR=1e-4 \
  BATCH=24 \
  bash scripts/analysis/run_scale_objpres_retry_local.sh \
  > "${LOG_DIR}/gpu0_linear.log" 2>&1 &
echo $! > "${LOG_DIR}/gpu0_linear.pid"

# GPU1: sqrt scaling
nohup env \
  RUN_TAG="eccv_upmix_nepa_qa_dualmask_scale_objpres_sqrt_s0_${RUN_ID}" \
  GPU_ID=1 \
  DUAL_MASK_WINDOW_SCALE=sqrt \
  LR=1e-4 \
  BATCH=24 \
  bash scripts/analysis/run_scale_objpres_retry_local.sh \
  > "${LOG_DIR}/gpu1_sqrt.log" 2>&1 &
echo $! > "${LOG_DIR}/gpu1_sqrt.pid"

echo "[$(date +"%F %T")] launched" | tee -a "${LOG_DIR}/launch.log"
echo "gpu0 pid: $(cat "${LOG_DIR}/gpu0_linear.pid")" | tee -a "${LOG_DIR}/launch.log"
echo "gpu1 pid: $(cat "${LOG_DIR}/gpu1_sqrt.pid")" | tee -a "${LOG_DIR}/launch.log"
