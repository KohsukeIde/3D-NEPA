#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MANIFEST="${MANIFEST:-${WORKDIR}/posetnepa_mask_order_preflight/generated/manifest.json}"
PRETRAIN_WRAPPER="${PRETRAIN_WRAPPER:-${WORKDIR}/pointnepa/scripts/local/pointgpt_train_local_ddp.sh}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ORDERS="${ORDERS:-}"      # optional comma-separated subset
MASKS="${MASKS:-}"        # optional comma-separated subset, e.g. 0.0,0.7
USE_WANDB="${USE_WANDB:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[error] manifest not found: ${MANIFEST}"
  echo "Run 01_make_mask_order_configs.py first."
  exit 2
fi
if [[ ! -x "${PRETRAIN_WRAPPER}" ]]; then
  echo "[error] pretrain wrapper missing or not executable: ${PRETRAIN_WRAPPER}"
  exit 2
fi

mapfile -t ROWS < <("${PYTHON_BIN}" - "${MANIFEST}" "${ORDERS}" "${MASKS}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
orders = {x.strip() for x in sys.argv[2].split(',') if x.strip()}
masks = {float(x.strip()) for x in sys.argv[3].split(',') if x.strip()}
for e in m['runs']:
    if orders and e['order'] not in orders:
        continue
    if masks and float(e['mask_ratio']) not in masks:
        continue
    print('\t'.join([e['run_id'], e['order'], str(e['mask_ratio']), e['pretrain_config'], e['pretrain_exp']]))
PY
)

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "[error] no manifest rows selected"
  echo "orders=${ORDERS:-<all>} masks=${MASKS:-<all>}"
  exit 2
fi

idx=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r RUN_ID ORDER MASK CONFIG_PATH BASE_EXP <<< "${row}"
  EXP_NAME="${BASE_EXP}_${RUN_TAG}"
  MASTER_PORT=$((MASTER_PORT_BASE + idx))
  echo "=== PRETRAIN run=${RUN_ID} order=${ORDER} mask=${MASK} config=${CONFIG_PATH} exp=${EXP_NAME} ==="
  cmd=(env
    WORKDIR="${WORKDIR}"
    CONFIG_PATH="${CONFIG_PATH}"
    EXP_NAME="${EXP_NAME}"
    USE_WANDB="${USE_WANDB}"
    NPROC_PER_NODE="${NPROC_PER_NODE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    NUM_WORKERS="${NUM_WORKERS}"
    MASTER_PORT="${MASTER_PORT}"
    WANDB_GROUP="posetnepa_mask_order_pretrain"
    WANDB_TAGS="posetnepa,mask_order,pretrain,order_${ORDER},mask_${MASK}"
    "${PRETRAIN_WRAPPER}"
  )
  printf '[cmd] '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
  idx=$((idx + 1))
done
