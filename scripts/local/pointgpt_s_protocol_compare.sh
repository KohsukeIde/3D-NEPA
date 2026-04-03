#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
GENERIC_SCRIPT="${SCRIPT_DIR}/pointgpt_protocol_compare.sh"

RUN_TAG="${RUN_TAG:-pointgpt_protocol_compare_official_s_$(date +%Y%m%d_%H%M%S)}"
CKPT_PATH="${CKPT_PATH:-${POINTGPT_DIR}/checkpoints/official/pointgpt_s_pretrain_official.pth}"

if [[ ! -x "${GENERIC_SCRIPT}" ]]; then
  echo "[error] missing generic protocol compare script: ${GENERIC_SCRIPT}"
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] official PointGPT-S checkpoint missing: ${CKPT_PATH}"
  exit 2
fi

SOURCE_LABEL="${SOURCE_LABEL:-pointgpt_s_pretrain_official}" \
CKPT_PATH="${CKPT_PATH}" \
OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objbg.yaml}" \
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objonly.yaml}" \
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_hardest.yaml}" \
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-300}" \
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-300}" \
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-300}" \
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-3}" \
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_protocol_compare_s}" \
RUN_TAG="${RUN_TAG}" \
bash "${GENERIC_SCRIPT}"
