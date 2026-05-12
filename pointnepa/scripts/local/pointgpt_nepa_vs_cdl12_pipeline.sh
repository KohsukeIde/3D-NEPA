#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
PRETRAIN_WRAPPER="${PRETRAIN_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_train_local_ddp.sh}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_finetune_local_ddp.sh}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointgpt_nepa_vs_cdl12_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointgpt_nepa_vs_cdl12}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NEPA_CONFIG_PATH="${NEPA_CONFIG_PATH:-cfgs/PointGPT-B/pretrain_nepa_cosine_shapenet_cache_v0.yaml}"
NEPA_EXP_NAME="${NEPA_EXP_NAME:-}"
NEPA_MAX_EPOCH="${NEPA_MAX_EPOCH:-300}"

CDL12_CONFIG_PATH="${CDL12_CONFIG_PATH:-cfgs/PointGPT-B/pretrain_cdl12_shapenet_cache_v0.yaml}"
CDL12_EXP_NAME="${CDL12_EXP_NAME:-pointgpt_cdl12_shapenet_cache_v0_online_${RUN_TAG}}"
CDL12_MAX_EPOCH="${CDL12_MAX_EPOCH:-300}"

FT_CONFIG_PATH="${FT_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
FT_MAX_EPOCH="${FT_MAX_EPOCH:-30}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-pointgpt-pretrain}"
PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-pointgpt_local_ddp_full}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_local_ddp_ft_compare}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"

if [[ -z "${NEPA_EXP_NAME}" ]]; then
  echo "[error] NEPA_EXP_NAME is required so the pipeline can watch the current run"
  exit 2
fi
if [[ ! -x "${PRETRAIN_WRAPPER}" ]]; then
  echo "[error] pretrain wrapper missing or not executable: ${PRETRAIN_WRAPPER}"
  exit 2
fi
if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then
  echo "[error] finetune wrapper missing or not executable: ${FINETUNE_WRAPPER}"
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python bin missing or not executable: ${PYTHON_BIN}"
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
acc = best.get("acc", float("nan"))
print(float(acc))
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

run_pretrain_sync() {
  local config_path="$1"
  local exp_name="$2"
  local project="$3"
  local group="$4"
  local tags="$5"
  local master_port="$6"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${project}" \
  WANDB_GROUP="${group}" \
  WANDB_RUN_NAME="${exp_name}" \
  WANDB_TAGS="${tags}" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${master_port}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  "${PRETRAIN_WRAPPER}"
}

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local ckpt_path="$3"
  local project="$4"
  local group="$5"
  local tags="$6"
  local master_port="$7"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${project}" \
  WANDB_GROUP="${group}" \
  WANDB_RUN_NAME="${exp_name}" \
  WANDB_TAGS="${tags}" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${master_port}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  CKPT_PATH="${ckpt_path}" \
  VAL_FREQ="${FT_VAL_FREQ}" \
  "${FINETUNE_WRAPPER}"
}

NEPA_EXP_PATH="$(exp_path_from_cfg "${NEPA_CONFIG_PATH}" "${NEPA_EXP_NAME}")"
CDL12_EXP_PATH="$(exp_path_from_cfg "${CDL12_CONFIG_PATH}" "${CDL12_EXP_NAME}")"
FT_NEPA_EXP_NAME="pointgpt_ft_objbg_from_${NEPA_EXP_NAME}_${RUN_TAG}"
FT_CDL12_EXP_NAME="pointgpt_ft_objbg_from_${CDL12_EXP_NAME}_${RUN_TAG}"
FT_NEPA_EXP_PATH="$(exp_path_from_cfg "${FT_CONFIG_PATH}" "${FT_NEPA_EXP_NAME}")"
FT_CDL12_EXP_PATH="$(exp_path_from_cfg "${FT_CONFIG_PATH}" "${FT_CDL12_EXP_NAME}")"

echo "=== POINTGPT NEPA VS CDL12 PIPELINE ==="
echo "date=$(date -Is)"
echo "run_tag=${RUN_TAG}"
echo "nepa_exp_name=${NEPA_EXP_NAME}"
echo "nepa_exp_path=${NEPA_EXP_PATH}"
echo "cdl12_exp_name=${CDL12_EXP_NAME}"
echo "cdl12_exp_path=${CDL12_EXP_PATH}"
echo "ft_config_path=${FT_CONFIG_PATH}"
echo

wait_for_completion "nepa_cosine_pretrain" "${NEPA_EXP_NAME}" "${NEPA_EXP_PATH}" "${NEPA_MAX_EPOCH}"
NEPA_CKPT="${NEPA_EXP_PATH}/ckpt-last.pth"

run_pretrain_sync \
  "${CDL12_CONFIG_PATH}" \
  "${CDL12_EXP_NAME}" \
  "${PRETRAIN_WANDB_PROJECT}" \
  "${PRETRAIN_WANDB_GROUP}" \
  "pointgpt,cdl12,full,ddp2,online,shapenet_cache_v0" \
  "29541"
wait_for_completion "cdl12_pretrain" "${CDL12_EXP_NAME}" "${CDL12_EXP_PATH}" "${CDL12_MAX_EPOCH}"
CDL12_CKPT="${CDL12_EXP_PATH}/ckpt-last.pth"

run_finetune_sync \
  "${FT_CONFIG_PATH}" \
  "${FT_NEPA_EXP_NAME}" \
  "${NEPA_CKPT}" \
  "${FT_WANDB_PROJECT}" \
  "${FT_WANDB_GROUP}" \
  "pointgpt,finetune,objbg,ddp2,nepa_cosine" \
  "29543"
wait_for_completion "ft_nepa_cosine_objbg" "${FT_NEPA_EXP_NAME}" "${FT_NEPA_EXP_PATH}" "${FT_MAX_EPOCH}"

run_finetune_sync \
  "${FT_CONFIG_PATH}" \
  "${FT_CDL12_EXP_NAME}" \
  "${CDL12_CKPT}" \
  "${FT_WANDB_PROJECT}" \
  "${FT_WANDB_GROUP}" \
  "pointgpt,finetune,objbg,ddp2,cdl12" \
  "29545"
wait_for_completion "ft_cdl12_objbg" "${FT_CDL12_EXP_NAME}" "${FT_CDL12_EXP_PATH}" "${FT_MAX_EPOCH}"

NEPA_FT_CKPT="${FT_NEPA_EXP_PATH}/ckpt-last.pth"
CDL12_FT_CKPT="${FT_CDL12_EXP_PATH}/ckpt-last.pth"
NEPA_FT_ACC="$(best_acc "${NEPA_FT_CKPT}")"
CDL12_FT_ACC="$(best_acc "${CDL12_FT_CKPT}")"

SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${SUMMARY_PATH}" <<EOF
# PointGPT NEPA vs CDL12 Pipeline Summary

- run_tag: \`${RUN_TAG}\`
- date: \`$(date -Is)\`

## Pretrain

- nepa_cosine:
  - exp: \`${NEPA_EXP_NAME}\`
  - path: \`${NEPA_EXP_PATH}\`
  - ckpt: \`${NEPA_CKPT}\`
- cdl12:
  - exp: \`${CDL12_EXP_NAME}\`
  - path: \`${CDL12_EXP_PATH}\`
  - ckpt: \`${CDL12_CKPT}\`

## Fine-tune (`obj_bg`)

- nepa_cosine:
  - exp: \`${FT_NEPA_EXP_NAME}\`
  - path: \`${FT_NEPA_EXP_PATH}\`
  - best_acc: \`${NEPA_FT_ACC}\`
- cdl12:
  - exp: \`${FT_CDL12_EXP_NAME}\`
  - path: \`${FT_CDL12_EXP_PATH}\`
  - best_acc: \`${CDL12_FT_ACC}\`

## Delta

- \`nepa_cosine - cdl12 = $("${PYTHON_BIN}" - <<PY
nepa = float("${NEPA_FT_ACC}")
cdl12 = float("${CDL12_FT_ACC}")
print(nepa - cdl12)
PY
)\`
EOF

echo "[done] summary written to ${SUMMARY_PATH}"
