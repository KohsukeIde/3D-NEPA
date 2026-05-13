#!/usr/bin/env bash
set -euo pipefail

# Full chain driver. Defaults to Stage 1 only because Stage 2/3 are expensive.
# Set RUN_STAGE2=1 and RUN_STAGE3=1 when the earlier summaries justify it.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-0}"
RUN_STAGE3="${RUN_STAGE3:-0}"

if [[ "${RUN_STAGE1}" == "1" ]]; then
  echo "=== running Stage 1 diagnostic lock ==="
  bash "${SCRIPT_DIR}/run_stage1_local.sh"
fi

if [[ "${RUN_STAGE2}" == "1" ]]; then
  echo "=== running Stage 2 wider order sweep ==="
  bash "${SCRIPT_DIR}/run_stage2_order_sweep.sh"
fi

if [[ "${RUN_STAGE3}" == "1" ]]; then
  echo "=== running Stage 3 confirmation sweep ==="
  bash "${SCRIPT_DIR}/run_stage3_confirm.sh"
fi

echo "[done] chain complete"
