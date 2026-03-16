#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_finetune_local_ddp.sh}"
TEST_WRAPPER="${TEST_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_test_local.sh}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointgpt_protocol_compare_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_protocol_compare}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TEST_CUDA_VISIBLE_DEVICES="${TEST_CUDA_VISIBLE_DEVICES:-0}"

SOURCE_LABEL="${SOURCE_LABEL:-cdl12_local}"
CKPT_PATH="${CKPT_PATH:-${POINTGPT_DIR}/experiments/pretrain_cdl12_shapenet_cache_v0/PointGPT-B/pointgpt_cdl12_shapenet_cache_v0_online_pointgpt_nepa_vs_cdl12_objbg_resume_20260311_1306/ckpt-last.pth}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-3}"
TEST_VOTE_TIMES="${TEST_VOTE_TIMES:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"

OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-30}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-50}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-30}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_protocol_compare}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

WAIT_FOR_PGREP_PATTERN="${WAIT_FOR_PGREP_PATTERN:-}"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then
  echo "[error] finetune wrapper missing or not executable: ${FINETUNE_WRAPPER}"
  exit 2
fi
if [[ ! -x "${TEST_WRAPPER}" ]]; then
  echo "[error] test wrapper missing or not executable: ${TEST_WRAPPER}"
  exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python bin missing or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[error] source checkpoint missing: ${CKPT_PATH}"
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

  found="$(find "${POINTGPT_DIR}/experiments" -type d -name "${exp_name}" | sort | rg -v '/experiments/config/' | head -n 1 || true)"
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

ckpt_epoch_if_exists() {
  local ckpt_path="$1"
  if [[ -f "${ckpt_path}" ]]; then
    ckpt_epoch "${ckpt_path}"
  else
    printf '%s\n' "-1"
  fi
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

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local tags="$3"
  local master_port="$4"
  local no_test_as_val="$5"
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
  NO_TEST_AS_VAL="${no_test_as_val}" \
  VAL_RATIO="${VAL_RATIO}" \
  VAL_SEED="${VAL_SEED}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  MASTER_PORT="${master_port}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  RUNTIME_META_PATH="${meta_path}" \
  CKPT_PATH="${CKPT_PATH}" \
  "${FINETUNE_WRAPPER}"
}

run_test_sync() {
  local config_path="$1"
  local exp_name="$2"
  local ckpt_path="$3"
  local log_path="$4"

  CUDA_VISIBLE_DEVICES="${TEST_CUDA_VISIBLE_DEVICES}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  TEST_VOTE_TIMES="${TEST_VOTE_TIMES}" \
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  CKPT_PATH="${ckpt_path}" \
  "${TEST_WRAPPER}" 2>&1 | tee "${log_path}"
}

extract_plain_test_acc() {
  local log_path="$1"
  "${PYTHON_BIN}" - "$log_path" <<'PY'
import re
import sys

text = open(sys.argv[1], "r").read().splitlines()
acc = float("nan")
for line in text:
    m = re.search(r"\[TEST\] acc = ([0-9.]+)", line)
    if m:
        acc = float(m.group(1))
print(acc)
PY
}

run_protocol_variant() {
  local variant="$1"
  local config_path="$2"
  local expected_epoch="$3"
  local policy_label="$4"
  local no_test_as_val="$5"
  local port_train="$6"

  local exp_name="pointgpt_ft_${variant}_from_${SOURCE_LABEL}_pointgptft_${policy_label}_${RUN_TAG}"
  local exp_path
  exp_path="$(resolve_existing_exp_path "${exp_name}" "$(exp_path_from_cfg "${config_path}" "${exp_name}")")"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"

  local existing_epoch
  existing_epoch="$(ckpt_epoch_if_exists "${exp_path}/ckpt-last.pth")"
  if (( existing_epoch < expected_epoch )); then
    run_finetune_sync \
      "${config_path}" \
      "${exp_name}" \
      "pointgpt,protocol-compare,${SOURCE_LABEL},${variant},${policy_label}" \
      "${port_train}" \
      "${no_test_as_val}" \
      "${meta_path}"
    exp_path="$(resolve_existing_exp_path "${exp_name}" "${exp_path}")"
  fi
  wait_for_exp_completion "${variant}-${policy_label}" "${exp_name}" "${exp_path}" "${expected_epoch}"

  local best_ckpt="${exp_path}/ckpt-best.pth"
  if [[ ! -f "${best_ckpt}" ]]; then
    best_ckpt="${exp_path}/ckpt-last.pth"
  fi

  local test_exp_name="${exp_name}_test"
  local test_log="${LOG_ROOT}/${test_exp_name}.log"
  run_test_sync "${exp_path}/config.yaml" "${test_exp_name}" "${best_ckpt}" "${test_log}"

  cat <<EOF
### ${variant} / ${policy_label}

- train_exp: \`${exp_name}\`
- train_path: \`${exp_path}\`
- best_val_metric: \`$(best_acc "${exp_path}/ckpt-last.pth")\`
- test_ckpt: \`${best_ckpt}\`
- test_log: \`${test_log}\`
- test_acc_plain: \`$(extract_plain_test_acc "${test_log}")\`

EOF
}

wait_for_pattern "${WAIT_FOR_PGREP_PATTERN}"

SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
{
  echo "# PointGPT Protocol Compare"
  echo
  echo "- date: $(date -Is)"
  echo "- source_label: \`${SOURCE_LABEL}\`"
  echo "- ckpt: \`${CKPT_PATH}\`"
  echo "- ft_recon_weight: \`${FT_RECON_WEIGHT}\`"
  echo "- val_ratio: \`${VAL_RATIO}\`"
  echo "- val_seed: \`${VAL_SEED}\`"
  echo "- test_vote_times: \`${TEST_VOTE_TIMES}\`"
  echo
  run_protocol_variant "objbg" "${OBJ_BG_CONFIG_PATH}" "${FT_MAX_EPOCH_OBJ_BG}" "testasval" "0" "29631"
  run_protocol_variant "objbg" "${OBJ_BG_CONFIG_PATH}" "${FT_MAX_EPOCH_OBJ_BG}" "strict" "1" "29632"
  run_protocol_variant "objonly" "${OBJ_ONLY_CONFIG_PATH}" "${FT_MAX_EPOCH_OBJ_ONLY}" "testasval" "0" "29633"
  run_protocol_variant "objonly" "${OBJ_ONLY_CONFIG_PATH}" "${FT_MAX_EPOCH_OBJ_ONLY}" "strict" "1" "29634"
  run_protocol_variant "hardest" "${HARDEST_CONFIG_PATH}" "${FT_MAX_EPOCH_HARDEST}" "testasval" "0" "29635"
  run_protocol_variant "hardest" "${HARDEST_CONFIG_PATH}" "${FT_MAX_EPOCH_HARDEST}" "strict" "1" "29636"
} | tee "${SUMMARY_PATH}"

echo "[done] summary=${SUMMARY_PATH}"
