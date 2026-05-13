#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
MANIFEST="${MANIFEST:-${WORKDIR}/posetnepa_mask_order_preflight/generated/manifest.json}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${WORKDIR}/pointnepa/scripts/local/pointgpt_finetune_local_ddp.sh}"
RUN_TAG="${RUN_TAG:-}"  # if empty, auto-detect latest matching pretrain per run
ORDERS="${ORDERS:-}"
MASKS="${MASKS:-}"
FT_SPLITS="${FT_SPLITS:-hardest}"  # hardest,objbg,objonly
USE_WANDB="${USE_WANDB:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29800}"
DRY_RUN="${DRY_RUN:-0}"
VAL_FREQ="${VAL_FREQ:-1}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-0}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "${MANIFEST}" ]]; then echo "[error] manifest not found: ${MANIFEST}"; exit 2; fi
if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then echo "[error] finetune wrapper missing: ${FINETUNE_WRAPPER}"; exit 2; fi

find_ckpt() {
  local cfg_rel="$1" base_exp="$2"
  local cfg_stem cfg_parent pattern latest
  cfg_stem="$(basename "${cfg_rel%.*}")"
  cfg_parent="$(basename "$(dirname "${cfg_rel}")")"
  if [[ -n "${RUN_TAG}" ]]; then
    pattern="${POINTGPT_DIR}/experiments/${cfg_stem}/${cfg_parent}/${base_exp}_${RUN_TAG}/ckpt-last.pth"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "${pattern}"
      return 0
    fi
    [[ -f "${pattern}" ]] && { echo "${pattern}"; return 0; }
  fi
  latest="$(ls -1dt "${POINTGPT_DIR}/experiments/${cfg_stem}/${cfg_parent}/${base_exp}"_*/ckpt-last.pth 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "[error] no ckpt found for ${base_exp} under ${cfg_stem}/${cfg_parent}" >&2
    return 1
  fi
  echo "${latest}"
}

mapfile -t ROWS < <("${PYTHON_BIN}" - "${MANIFEST}" "${ORDERS}" "${MASKS}" "${FT_SPLITS}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
orders = {x.strip() for x in sys.argv[2].split(',') if x.strip()}
masks = {float(x.strip()) for x in sys.argv[3].split(',') if x.strip()}
aliases = {
    "obj_bg": "objbg",
    "objbg": "objbg",
    "obj_only": "objonly",
    "objonly": "objonly",
    "hardest": "hardest",
    "pb_t50_rs": "hardest",
    "pb": "hardest",
}
splits = []
for raw in [x.strip() for x in sys.argv[4].split(',') if x.strip()]:
    key = raw.lower()
    if key not in aliases:
        raise SystemExit(f"[error] unsupported FT split: {raw}")
    splits.append(aliases[key])
for e in m['runs']:
    if orders and e['order'] not in orders:
        continue
    if masks and float(e['mask_ratio']) not in masks:
        continue
    for split in splits:
        print('\t'.join([e['run_id'], e['order'], str(e['mask_ratio']), split, e['pretrain_config'], e['pretrain_exp'], e['finetune_configs'][split], e['finetune_exp_prefix']]))
PY
)

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "[error] no finetune rows selected"
  echo "orders=${ORDERS:-<all>} masks=${MASKS:-<all>} ft_splits=${FT_SPLITS}"
  exit 2
fi

idx=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r RUN_ID ORDER MASK SPLIT PRE_CFG PRE_EXP FT_CFG FT_PREFIX <<< "${row}"
  CKPT_PATH="$(find_ckpt "${PRE_CFG}" "${PRE_EXP}")"
  CKPT_RUN="$(basename "$(dirname "${CKPT_PATH}")")"
  EXP_NAME="${FT_PREFIX}_${SPLIT}_from_${CKPT_RUN}"
  MASTER_PORT=$((MASTER_PORT_BASE + idx))
  echo "=== FINETUNE run=${RUN_ID} split=${SPLIT} ckpt=${CKPT_PATH} exp=${EXP_NAME} ==="
  cmd=(env
    WORKDIR="${WORKDIR}"
    CONFIG_PATH="${FT_CFG}"
    EXP_NAME="${EXP_NAME}"
    CKPT_PATH="${CKPT_PATH}"
    USE_WANDB="${USE_WANDB}"
    NPROC_PER_NODE="${NPROC_PER_NODE}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
    NUM_WORKERS="${NUM_WORKERS}"
    MASTER_PORT="${MASTER_PORT}"
    VAL_FREQ="${VAL_FREQ}"
    FT_RECON_WEIGHT="${FT_RECON_WEIGHT}"
    SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH}"
    WANDB_GROUP="posetnepa_mask_order_finetune"
    WANDB_TAGS="posetnepa,mask_order,finetune,order_${ORDER},mask_${MASK},split_${SPLIT}"
    "${FINETUNE_WRAPPER}"
  )
  printf '[cmd] '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}"
  fi
  idx=$((idx + 1))
done
