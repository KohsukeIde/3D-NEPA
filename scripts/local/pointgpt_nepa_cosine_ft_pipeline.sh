#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_finetune_local_ddp.sh}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointgpt_nepa_cosine_ft_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_nepa_cosine_ft}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NEPA_CKPT_PATH="${NEPA_CKPT_PATH:-${POINTGPT_DIR}/experiments/pretrain_nepa_cosine_shapenet_cache_v0/PointGPT-B/pointgpt_nepa_cosine_shapenet_cache_v0_online_retry_20260306_234205/ckpt-last.pth}"
OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-30}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-50}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-30}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-0}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-transfer}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_nepa_cosine_ft}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then
  echo "[error] finetune wrapper missing or not executable: ${FINETUNE_WRAPPER}"
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python bin missing or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -f "${NEPA_CKPT_PATH}" ]]; then
  echo "[error] nepa checkpoint missing: ${NEPA_CKPT_PATH}"
  exit 2
fi

exp_path_from_cfg() {
  local cfg_path="$1"
  local exp_name="$2"
  local cfg_stem
  local cfg_parent
  cfg_stem="$(basename "${cfg_path%.*}")"
  cfg_parent="$(basename "$(dirname "${cfg_path}")")"
  printf '%s/experiments/%s/%s/%s\n' "${POINTGPT_DIR}" "${cfg_stem}" "${cfg_parent}" "${exp_name}"
}

ckpt_epoch() {
  local ckpt_path="$1"
  "${PYTHON_BIN}" - "$ckpt_path" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu")
print(int(state.get("epoch", -1)))
PY
}

best_acc() {
  local ckpt_path="$1"
  "${PYTHON_BIN}" - "$ckpt_path" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu")
best = state.get("best_metrics", {})
if hasattr(best, "state_dict"):
    best = best.state_dict()
print(float(best.get("acc", float("nan"))))
PY
}

wait_for_completion() {
  local label="$1"
  local exp_name="$2"
  local exp_path="$3"
  local expected_epoch="$4"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local log_path=""

  echo "[wait] ${label}: exp_name=${exp_name}"
  echo "[wait] ${label}: exp_path=${exp_path}"

  while true; do
    if compgen -G "${exp_path}/*.log" >/dev/null; then
      log_path="$(ls -1 "${exp_path}"/*.log | sort | tail -n 1)"
    fi

    local alive=0
    if pgrep -af -- "--exp_name ${exp_name}" >/dev/null; then
      alive=1
    fi

    if [[ "${alive}" == "0" ]]; then
      if [[ ! -f "${ckpt_path}" ]]; then
        echo "[error] ${label}: process ended without ckpt-last: ${ckpt_path}"
        [[ -n "${log_path}" ]] && tail -n 80 "${log_path}"
        exit 1
      fi
      local epoch
      epoch="$(ckpt_epoch "${ckpt_path}")"
      if (( epoch < expected_epoch )); then
        echo "[error] ${label}: process ended at epoch=${epoch}, expected >= ${expected_epoch}"
        [[ -n "${log_path}" ]] && tail -n 80 "${log_path}"
        exit 1
      fi
      echo "[done] ${label}: epoch=${epoch} ckpt=${ckpt_path}"
      return 0
    fi

    if [[ -n "${log_path}" ]]; then
      echo "[wait] ${label}: alive=1 latest=$(tail -n 1 "${log_path}")"
    else
      echo "[wait] ${label}: alive=1 log=missing"
    fi
    sleep "${POLL_SEC}"
  done
}

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local tags="$3"
  local master_port="$4"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_GROUP="${WANDB_GROUP}" \
  WANDB_RUN_NAME="${exp_name}" \
  WANDB_TAGS="${tags}" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  FT_RECON_WEIGHT="${FT_RECON_WEIGHT}" \
  SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${master_port}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  CKPT_PATH="${NEPA_CKPT_PATH}" \
  VAL_FREQ="${FT_VAL_FREQ}" \
  "${FINETUNE_WRAPPER}"
}

OBJ_BG_EXP_NAME="pointgpt_ft_objbg_from_nepa_cosine_${RUN_TAG}"
OBJ_ONLY_EXP_NAME="pointgpt_ft_objonly_from_nepa_cosine_${RUN_TAG}"
HARDEST_EXP_NAME="pointgpt_ft_hardest_from_nepa_cosine_${RUN_TAG}"

OBJ_BG_EXP_PATH="$(exp_path_from_cfg "${OBJ_BG_CONFIG_PATH}" "${OBJ_BG_EXP_NAME}")"
OBJ_ONLY_EXP_PATH="$(exp_path_from_cfg "${OBJ_ONLY_CONFIG_PATH}" "${OBJ_ONLY_EXP_NAME}")"
HARDEST_EXP_PATH="$(exp_path_from_cfg "${HARDEST_CONFIG_PATH}" "${HARDEST_EXP_NAME}")"

echo "=== POINTGPT NEPA COSINE FT PIPELINE ==="
echo "date=$(date -Is)"
echo "run_tag=${RUN_TAG}"
echo "nepa_ckpt_path=${NEPA_CKPT_PATH}"
echo "ft_recon_weight=${FT_RECON_WEIGHT}"
echo "save_last_every_epoch=${SAVE_LAST_EVERY_EPOCH}"
echo "obj_bg_exp_name=${OBJ_BG_EXP_NAME}"
echo "objonly_exp_name=${OBJ_ONLY_EXP_NAME}"
echo "hardest_exp_name=${HARDEST_EXP_NAME}"
echo

run_finetune_sync \
  "${OBJ_BG_CONFIG_PATH}" \
  "${OBJ_BG_EXP_NAME}" \
  "pointgpt,finetune,objbg,ddp2,nepa_cosine,cls_only" \
  "29551"
wait_for_completion "ft_nepa_cosine_objbg" "${OBJ_BG_EXP_NAME}" "${OBJ_BG_EXP_PATH}" "${FT_MAX_EPOCH_OBJ_BG}"

run_finetune_sync \
  "${OBJ_ONLY_CONFIG_PATH}" \
  "${OBJ_ONLY_EXP_NAME}" \
  "pointgpt,finetune,objonly,ddp2,nepa_cosine,cls_only" \
  "29553"
wait_for_completion "ft_nepa_cosine_objonly" "${OBJ_ONLY_EXP_NAME}" "${OBJ_ONLY_EXP_PATH}" "${FT_MAX_EPOCH_OBJ_ONLY}"

run_finetune_sync \
  "${HARDEST_CONFIG_PATH}" \
  "${HARDEST_EXP_NAME}" \
  "pointgpt,finetune,hardest,ddp2,nepa_cosine,cls_only" \
  "29555"
wait_for_completion "ft_nepa_cosine_hardest" "${HARDEST_EXP_NAME}" "${HARDEST_EXP_PATH}" "${FT_MAX_EPOCH_HARDEST}"

OBJ_BG_ACC="$(best_acc "${OBJ_BG_EXP_PATH}/ckpt-last.pth")"
OBJ_ONLY_ACC="$(best_acc "${OBJ_ONLY_EXP_PATH}/ckpt-last.pth")"
HARDEST_ACC="$(best_acc "${HARDEST_EXP_PATH}/ckpt-last.pth")"

SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${SUMMARY_PATH}" <<EOF
# PointGPT NEPA Cosine FT Summary

- run_tag: \`${RUN_TAG}\`
- date: \`$(date -Is)\`
- source_ckpt: \`${NEPA_CKPT_PATH}\`
- ft_recon_weight: \`${FT_RECON_WEIGHT}\`
- save_last_every_epoch: \`${SAVE_LAST_EVERY_EPOCH}\`

## Results

- obj_bg:
  - exp: \`${OBJ_BG_EXP_NAME}\`
  - path: \`${OBJ_BG_EXP_PATH}\`
  - best_acc: \`${OBJ_BG_ACC}\`
- objonly:
  - exp: \`${OBJ_ONLY_EXP_NAME}\`
  - path: \`${OBJ_ONLY_EXP_PATH}\`
  - best_acc: \`${OBJ_ONLY_ACC}\`
- hardest:
  - exp: \`${HARDEST_EXP_NAME}\`
  - path: \`${HARDEST_EXP_PATH}\`
  - best_acc: \`${HARDEST_ACC}\`
EOF

echo "[done] summary written to ${SUMMARY_PATH}"
