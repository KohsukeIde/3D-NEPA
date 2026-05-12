#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
PRETRAIN_WRAPPER="${PRETRAIN_WRAPPER:-${SCRIPT_DIR}/pointgpt_train_local_ddp.sh}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${SCRIPT_DIR}/pointgpt_finetune_local_ddp.sh}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointnepa_s_vitshift_maskoff_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_s_pointnepa_vitshift_ablation}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

BASELINE_SUMMARY_PATH="${BASELINE_SUMMARY_PATH:-${WORKDIR}/logs/local/pointgpt_s_pointnepa_mask_ablation/pointnepa_s_maskoff_20260403_212525_summary.md}"
POINTNEPA_CONFIG_PATH="${POINTNEPA_CONFIG_PATH:-cfgs/PointGPT-S/pretrain_nepa_cosine_vitshift_shapenet_cache_v0_nomask.yaml}"
POINTNEPA_EXP_NAME="${POINTNEPA_EXP_NAME:-pointgpt_s_nepa_cosine_vitshift_shapenet_cache_v0_nomask_${RUN_TAG}}"
POINTNEPA_MAX_EPOCH="${POINTNEPA_MAX_EPOCH:-300}"

OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-300}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-300}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-300}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-pointgpt-pretrain}"
PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-pointnepa_s_pretrain}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointnepa_s_ft}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${PRETRAIN_WRAPPER}" ]]; then
  echo "[error] missing pretrain wrapper: ${PRETRAIN_WRAPPER}"
  exit 2
fi
if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then
  echo "[error] missing finetune wrapper: ${FINETUNE_WRAPPER}"
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] missing python bin: ${PYTHON_BIN}"
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

meta_path_for_exp() {
  local exp_name="$1"
  printf '%s/%s.meta.env\n' "${LOG_ROOT}" "${exp_name}"
}

load_meta_for_exp() {
  local exp_name="$1"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"
  if [[ ! -f "${meta_path}" ]]; then
    echo "[error] runtime meta missing for ${exp_name}: ${meta_path}"
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${meta_path}"
}

resolve_existing_exp_path() {
  local exp_name="$1"
  local fallback="$2"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"
  if [[ -f "${meta_path}" ]]; then
    # shellcheck disable=SC1090
    source "${meta_path}"
    local resolved="${EXPERIMENT_PATH}"
    if [[ "${resolved}" == ./* ]]; then
      resolved="${POINTGPT_DIR}/${resolved#./}"
    elif [[ "${resolved}" != /* ]]; then
      resolved="${POINTGPT_DIR}/${resolved}"
    fi
    printf '%s\n' "${resolved}"
    return 0
  fi

  local found=""
  while IFS= read -r candidate; do
    [[ -z "${candidate}" ]] && continue
    if [[ "${candidate}" == *"/experiments/config/"* ]]; then
      continue
    fi
    if [[ -f "${candidate}/ckpt-last.pth" || -f "${candidate}/ckpt-best.pth" ]]; then
      found="${candidate}"
      break
    fi
  done < <(find "${POINTGPT_DIR}/experiments" -type d -name "${exp_name}" | sort)
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return 0
  fi

  printf '%s\n' "${fallback}"
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

prepare_resume_ckpt() {
  local exp_path="$1"
  local last_ckpt="${exp_path}/ckpt-last.pth"
  local best_ckpt="${exp_path}/ckpt-best.pth"
  if [[ -f "${last_ckpt}" ]]; then
    printf '%s\n' "${last_ckpt}"
    return 0
  fi
  if [[ -f "${best_ckpt}" ]]; then
    cp -f "${best_ckpt}" "${last_ckpt}"
    echo "[resume] promoted ckpt-best -> ckpt-last at ${exp_path}" >&2
    printf '%s\n' "${last_ckpt}"
    return 0
  fi
  return 1
}

wait_for_file() {
  local path="$1"
  while [[ ! -f "${path}" ]]; do
    echo "[wait] missing file: ${path}"
    sleep "${POLL_SEC}"
  done
}

wait_for_exp_completion() {
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
  local tags="$3"
  local master_port="$4"
  local meta_path="$5"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT}" \
  WANDB_GROUP="${PRETRAIN_WANDB_GROUP}" \
  WANDB_RUN_NAME="${exp_name}" \
  WANDB_TAGS="${tags}" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${master_port}" \
  RUNTIME_META_PATH="${meta_path}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  "${PRETRAIN_WRAPPER}"
}

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local ckpt_path="$3"
  local tags="$4"
  local master_port="$5"
  local meta_path="$6"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${FT_WANDB_PROJECT}" \
  WANDB_GROUP="${FT_WANDB_GROUP}" \
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
  RUNTIME_META_PATH="${meta_path}" \
  CKPT_PATH="${ckpt_path}" \
  VAL_FREQ="${FT_VAL_FREQ}" \
  "${FINETUNE_WRAPPER}"
}

ensure_pretrain() {
  local exp_path
  local meta_path
  exp_path="$(exp_path_from_cfg "${POINTNEPA_CONFIG_PATH}" "${POINTNEPA_EXP_NAME}")"
  exp_path="$(resolve_existing_exp_path "${POINTNEPA_EXP_NAME}" "${exp_path}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch=-1

  if [[ -f "${ckpt_path}" ]]; then
    epoch="$(ckpt_epoch "${ckpt_path}")"
  fi

  if (( epoch >= POINTNEPA_MAX_EPOCH )); then
    echo "[done] pointnepa_vitshift_pretrain already complete: epoch=${epoch}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  if pgrep -af -- "--exp_name ${POINTNEPA_EXP_NAME}" >/dev/null; then
    wait_for_exp_completion "pointnepa_vitshift_pretrain_existing" "${POINTNEPA_EXP_NAME}" "${exp_path}" "${POINTNEPA_MAX_EPOCH}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  echo "[launch] pointnepa_vitshift_pretrain: exp_name=${POINTNEPA_EXP_NAME}" >&2
  meta_path="$(meta_path_for_exp "${POINTNEPA_EXP_NAME}")"
  run_pretrain_sync \
    "${POINTNEPA_CONFIG_PATH}" \
    "${POINTNEPA_EXP_NAME}" \
    "pointgpt,s,pointnepa,pretrain,vitshift,maskoff,ddp2" \
    "29681" \
    "${meta_path}" >&2
  load_meta_for_exp "${POINTNEPA_EXP_NAME}"
  exp_path="${EXPERIMENT_PATH}"
  if [[ "${exp_path}" == ./* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path#./}"
  elif [[ "${exp_path}" != /* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path}"
  fi
  wait_for_exp_completion "pointnepa_vitshift_pretrain" "${POINTNEPA_EXP_NAME}" "${exp_path}" "${POINTNEPA_MAX_EPOCH}" >&2
  printf '%s\n' "${ckpt_path}"
}

ensure_finetune_stage() {
  local label="$1"
  local config_path="$2"
  local exp_name="$3"
  local exp_path="$4"
  local expected_epoch="$5"
  local source_ckpt_path="$6"
  local tags="$7"
  local master_port="$8"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"
  exp_path="$(resolve_existing_exp_path "${exp_name}" "${exp_path}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch=-1
  local resume_args=""
  if prepare_resume_ckpt "${exp_path}" >/dev/null 2>&1; then
    ckpt_path="${exp_path}/ckpt-last.pth"
    epoch="$(ckpt_epoch "${ckpt_path}")"
    resume_args="--resume"
  fi

  if (( epoch >= expected_epoch )); then
    echo "[done] ${label} already complete: epoch=${epoch}" >&2
    return 0
  fi

  if pgrep -af -- "--exp_name ${exp_name}" >/dev/null; then
    wait_for_exp_completion "${label}_existing" "${exp_name}" "${exp_path}" "${expected_epoch}" >&2
    return 0
  fi

  echo "[launch] ${label}: exp_name=${exp_name}" >&2
  EXTRA_ARGS="${resume_args}" \
  run_finetune_sync \
    "${config_path}" \
    "${exp_name}" \
    "${source_ckpt_path}" \
    "${tags}" \
    "${master_port}" \
    "${meta_path}"
  load_meta_for_exp "${exp_name}"
  exp_path="${EXPERIMENT_PATH}"
  if [[ "${exp_path}" == ./* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path#./}"
  elif [[ "${exp_path}" != /* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path}"
  fi
  wait_for_exp_completion "${label}" "${exp_name}" "${exp_path}" "${expected_epoch}" >&2
}

wait_for_file "${BASELINE_SUMMARY_PATH}.done"
POINTNEPA_CKPT_PATH="$(ensure_pretrain)"

OBJ_BG_EXP_NAME="pointgpt_ft_objbg_from_pointnepa_vitshift_maskoff_clsonly_${RUN_TAG}"
OBJ_ONLY_EXP_NAME="pointgpt_ft_objonly_from_pointnepa_vitshift_maskoff_clsonly_${RUN_TAG}"
HARDEST_EXP_NAME="pointgpt_ft_hardest_from_pointnepa_vitshift_maskoff_clsonly_${RUN_TAG}"

OBJ_BG_EXP_PATH="$(exp_path_from_cfg "${OBJ_BG_CONFIG_PATH}" "${OBJ_BG_EXP_NAME}")"
OBJ_ONLY_EXP_PATH="$(exp_path_from_cfg "${OBJ_ONLY_CONFIG_PATH}" "${OBJ_ONLY_EXP_NAME}")"
HARDEST_EXP_PATH="$(exp_path_from_cfg "${HARDEST_CONFIG_PATH}" "${HARDEST_EXP_NAME}")"

ensure_finetune_stage \
  "pointnepa_vitshift_maskoff_clsonly_objbg" \
  "${OBJ_BG_CONFIG_PATH}" \
  "${OBJ_BG_EXP_NAME}" \
  "${OBJ_BG_EXP_PATH}" \
  "${FT_MAX_EPOCH_OBJ_BG}" \
  "${POINTNEPA_CKPT_PATH}" \
  "pointgpt,s,finetune,objbg,pointnepa,vitshift,maskoff,clsonly" \
  "29683"

ensure_finetune_stage \
  "pointnepa_vitshift_maskoff_clsonly_objonly" \
  "${OBJ_ONLY_CONFIG_PATH}" \
  "${OBJ_ONLY_EXP_NAME}" \
  "${OBJ_ONLY_EXP_PATH}" \
  "${FT_MAX_EPOCH_OBJ_ONLY}" \
  "${POINTNEPA_CKPT_PATH}" \
  "pointgpt,s,finetune,objonly,pointnepa,vitshift,maskoff,clsonly" \
  "29685"

ensure_finetune_stage \
  "pointnepa_vitshift_maskoff_clsonly_hardest" \
  "${HARDEST_CONFIG_PATH}" \
  "${HARDEST_EXP_NAME}" \
  "${HARDEST_EXP_PATH}" \
  "${FT_MAX_EPOCH_HARDEST}" \
  "${POINTNEPA_CKPT_PATH}" \
  "pointgpt,s,finetune,hardest,pointnepa,vitshift,maskoff,clsonly" \
  "29687"

SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${SUMMARY_PATH}" <<EOF
# PointNEPA-S Vit-Shift Mask-Off Summary

- run_tag: \`${RUN_TAG}\`
- baseline_maskoff_summary: \`${BASELINE_SUMMARY_PATH}\`
- source_label: \`pointnepa_vitshift_maskoff\`
- pretrain_config: \`${POINTNEPA_CONFIG_PATH}\`
- mask_ratio: \`0.0\`
- shift_mode: \`loss-side shift (vit-style)\`
- source_ckpt: \`${POINTNEPA_CKPT_PATH}\`
- ft_label: \`clsonly\`
- ft_recon_weight: \`${FT_RECON_WEIGHT}\`

## Results

- obj_bg:
  - exp: \`${OBJ_BG_EXP_NAME}\`
  - path: \`${OBJ_BG_EXP_PATH}\`
  - best_acc: \`$(best_acc "${OBJ_BG_EXP_PATH}/ckpt-last.pth")\`
- objonly:
  - exp: \`${OBJ_ONLY_EXP_NAME}\`
  - path: \`${OBJ_ONLY_EXP_PATH}\`
  - best_acc: \`$(best_acc "${OBJ_ONLY_EXP_PATH}/ckpt-last.pth")\`
- hardest:
  - exp: \`${HARDEST_EXP_NAME}\`
  - path: \`${HARDEST_EXP_PATH}\`
  - best_acc: \`$(best_acc "${HARDEST_EXP_PATH}/ckpt-last.pth")\`
EOF

DONE_PATH="${SUMMARY_PATH}.done"
cat > "${DONE_PATH}" <<EOF
run_tag=${RUN_TAG}
date=$(date -Is)
summary=${SUMMARY_PATH}
EOF

echo "[done] summary=${SUMMARY_PATH}"
