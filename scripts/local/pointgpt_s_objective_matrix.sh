#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
PRETRAIN_WRAPPER="${SCRIPT_DIR}/pointgpt_train_local_ddp.sh"
MATRIX_SCRIPT="${SCRIPT_DIR}/pointgpt_ft_recipe_matrix_2x2.sh"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointgpt_s_ft_recipe_matrix_2x2_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_s_objective_matrix}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_s_ft_recipe_matrix_2x2}"
POLL_SEC="${POLL_SEC:-60}"
WAIT_FOR_PGREP_PATTERN="${WAIT_FOR_PGREP_PATTERN:-pointgpt_protocol_compare_official_s_}"
PROTOCOL_RUN_TAG="${PROTOCOL_RUN_TAG:-pointgpt_protocol_compare_official_s_20260318}"
PROTOCOL_LOG_ROOT="${PROTOCOL_LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_protocol_compare}"
PROTOCOL_SOURCE_LABEL="${PROTOCOL_SOURCE_LABEL:-pointgpt_s_pretrain_official}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NEPA_CONFIG_PATH="${NEPA_CONFIG_PATH:-cfgs/PointGPT-S/pretrain_nepa_cosine_shapenet_cache_v0.yaml}"
NEPA_EXP_NAME="${NEPA_EXP_NAME:-pointgpt_s_nepa_cosine_shapenet_cache_v0_${RUN_TAG}}"
NEPA_MAX_EPOCH="${NEPA_MAX_EPOCH:-300}"

CDL12_CONFIG_PATH="${CDL12_CONFIG_PATH:-cfgs/PointGPT-S/pretrain_cdl12_shapenet_cache_v0.yaml}"
CDL12_EXP_NAME="${CDL12_EXP_NAME:-pointgpt_s_cdl12_shapenet_cache_v0_${RUN_TAG}}"
CDL12_MAX_EPOCH="${CDL12_MAX_EPOCH:-300}"

OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-300}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-300}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-300}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-pointgpt-pretrain}"
PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-pointgpt_s_matrix_pretrain}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_s_matrix_ft}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}" "${MATRIX_LOG_ROOT}"

if [[ ! -x "${PRETRAIN_WRAPPER}" ]]; then
  echo "[error] missing pretrain wrapper: ${PRETRAIN_WRAPPER}"
  exit 2
fi
if [[ ! -x "${MATRIX_SCRIPT}" ]]; then
  echo "[error] missing matrix script: ${MATRIX_SCRIPT}"
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

wait_for_protocol_compare_completion() {
  local run_tag="$1"
  local source_label="$2"
  local summary_path="${PROTOCOL_LOG_ROOT}/${run_tag}_summary.md"
  local done_path="${summary_path}.done"
  local expected_test_log="${PROTOCOL_LOG_ROOT}/pointgpt_ft_hardest_from_${source_label}_pointgptft_strict_${run_tag}_test.log"

  while true; do
    local alive=0
    if [[ -n "${WAIT_FOR_PGREP_PATTERN}" ]] && pgrep -af -- "${WAIT_FOR_PGREP_PATTERN}" >/dev/null; then
      alive=1
    fi

    if [[ -f "${done_path}" ]]; then
      echo "[done] protocol compare completion stamp found: ${done_path}"
      return 0
    fi

    if [[ "${alive}" == "0" ]]; then
      if [[ -f "${summary_path}" && -f "${expected_test_log}" ]]; then
        echo "[done] protocol compare outputs found after process exit"
        return 0
      fi
      echo "[error] protocol compare ended without expected outputs"
      echo "[error] summary_path=${summary_path}"
      echo "[error] expected_test_log=${expected_test_log}"
      exit 1
    fi

    echo "[wait] official compare still running: run_tag=${run_tag}"
    sleep "${POLL_SEC}"
  done
}

wait_for_pattern() {
  local pattern="$1"
  if [[ -z "${pattern}" ]]; then
    return 0
  fi
  while pgrep -af -- "${pattern}" >/dev/null; do
    echo "[wait] existing process still running: ${pattern}"
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

ensure_nepa_pretrain() {
  local exp_path
  exp_path="$(exp_path_from_cfg "${NEPA_CONFIG_PATH}" "${NEPA_EXP_NAME}")"
  exp_path="$(resolve_existing_exp_path "${NEPA_EXP_NAME}" "${exp_path}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch=-1

  if [[ -f "${ckpt_path}" ]]; then
    epoch="$(ckpt_epoch "${ckpt_path}")"
  fi

  if (( epoch >= NEPA_MAX_EPOCH )); then
    echo "[done] nepa_pretrain already complete: epoch=${epoch}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  if pgrep -af -- "--exp_name ${NEPA_EXP_NAME}" >/dev/null; then
    wait_for_exp_completion "nepa_pretrain_existing" "${NEPA_EXP_NAME}" "${exp_path}" "${NEPA_MAX_EPOCH}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  local extra_args=""
  local meta_path
  meta_path="$(meta_path_for_exp "${NEPA_EXP_NAME}")"
  if [[ -f "${ckpt_path}" ]]; then
    extra_args="--resume"
    echo "[resume] nepa_pretrain: epoch=${epoch} exp_name=${NEPA_EXP_NAME}"
  else
    echo "[launch] nepa_pretrain: exp_name=${NEPA_EXP_NAME}"
  fi

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT}" \
  WANDB_GROUP="${PRETRAIN_WANDB_GROUP}" \
  WANDB_RUN_NAME="${NEPA_EXP_NAME}" \
  WANDB_TAGS="pointgpt,s,pretrain,nepa_cosine,shapenet_cache_v0,ddp2" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="29661" \
  RUNTIME_META_PATH="${meta_path}" \
  CONFIG_PATH="${NEPA_CONFIG_PATH}" \
  EXP_NAME="${NEPA_EXP_NAME}" \
  EXTRA_ARGS="${extra_args}" \
  "${PRETRAIN_WRAPPER}"

  load_meta_for_exp "${NEPA_EXP_NAME}"
  exp_path="${EXPERIMENT_PATH}"
  if [[ "${exp_path}" == ./* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path#./}"
  elif [[ "${exp_path}" != /* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path}"
  fi
  wait_for_exp_completion "nepa_pretrain" "${NEPA_EXP_NAME}" "${exp_path}" "${NEPA_MAX_EPOCH}" >&2
  printf '%s\n' "${ckpt_path}"
}

wait_for_protocol_compare_completion "${PROTOCOL_RUN_TAG}" "${PROTOCOL_SOURCE_LABEL}"
NEPA_CKPT_PATH="$(ensure_nepa_pretrain)"

RUN_TAG="${RUN_TAG}" \
LOG_ROOT="${MATRIX_LOG_ROOT}" \
POLL_SEC="${POLL_SEC}" \
NUM_WORKERS="${NUM_WORKERS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NEPA_CKPT_PATH="${NEPA_CKPT_PATH}" \
CDL12_CONFIG_PATH="${CDL12_CONFIG_PATH}" \
CDL12_EXP_NAME="${CDL12_EXP_NAME}" \
CDL12_MAX_EPOCH="${CDL12_MAX_EPOCH}" \
OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH}" \
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH}" \
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH}" \
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG}" \
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY}" \
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST}" \
FT_VAL_FREQ="${FT_VAL_FREQ}" \
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH}" \
USE_WANDB="${USE_WANDB}" \
WANDB_ENTITY="${WANDB_ENTITY}" \
PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT}" \
PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP}" \
FT_WANDB_PROJECT="${FT_WANDB_PROJECT}" \
FT_WANDB_GROUP="${FT_WANDB_GROUP}" \
WANDB_MODE="${WANDB_MODE}" \
WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
WANDB_DIR="${WANDB_DIR}" \
bash "${MATRIX_SCRIPT}"

SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${SUMMARY_PATH}" <<EOF
# PointGPT-S Objective Matrix Orchestrator

- run_tag: \`${RUN_TAG}\`
- date: \`$(date -Is)\`
- protocol_run_tag: \`${PROTOCOL_RUN_TAG}\`
- nepa_ckpt: \`${NEPA_CKPT_PATH}\`
- matrix_summary: \`${MATRIX_LOG_ROOT}/${RUN_TAG}_summary.md\`
EOF

DONE_PATH="${SUMMARY_PATH}.done"
cat > "${DONE_PATH}" <<EOF
run_tag=${RUN_TAG}
date=$(date -Is)
summary=${SUMMARY_PATH}
EOF

echo "[done] summary=${SUMMARY_PATH}"
