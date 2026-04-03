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

RUN_TAG="${RUN_TAG:-pointgpt_ft_recipe_matrix_2x2_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_ft_recipe_matrix_2x2}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

NEPA_CKPT_PATH="${NEPA_CKPT_PATH:-${POINTGPT_DIR}/experiments/pretrain_nepa_cosine_shapenet_cache_v0/PointGPT-B/pointgpt_nepa_cosine_shapenet_cache_v0_online_retry_20260306_234205/ckpt-last.pth}"
NEPA_CLS_ONLY_RUN_TAG="${NEPA_CLS_ONLY_RUN_TAG:-}"
NEPA_CLS_ONLY_LOG_ROOT="${NEPA_CLS_ONLY_LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_nepa_cosine_ft}"

CDL12_CONFIG_PATH="${CDL12_CONFIG_PATH:-cfgs/PointGPT-B/pretrain_cdl12_shapenet_cache_v0.yaml}"
CDL12_EXP_NAME="${CDL12_EXP_NAME:-pointgpt_cdl12_shapenet_cache_v0_online_${RUN_TAG}}"
CDL12_MAX_EPOCH="${CDL12_MAX_EPOCH:-300}"

OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-30}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-50}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-30}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-pointgpt-pretrain}"
PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-pointgpt_matrix_pretrain}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_matrix_ft}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"

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

wait_for_exp_completion() {
  local label="$1"
  local exp_name="$2"
  local exp_path="$3"
  if [[ "${exp_path}" == ./* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path#./}"
  elif [[ "${exp_path}" != /* ]]; then
    exp_path="${POINTGPT_DIR}/${exp_path}"
  fi
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

wait_for_existing_chain_summary() {
  local label="$1"
  local summary_path="$2"
  local run_tag="$3"

  echo "[wait] ${label}: summary=${summary_path}"
  while true; do
    if [[ -f "${summary_path}" ]]; then
      echo "[done] ${label}: summary=${summary_path}"
      return 0
    fi

    local alive=0
    if pgrep -af -- "${run_tag}" >/dev/null; then
      alive=1
    fi
    if pgrep -af -- "pointgpt_nepa_cosine_ft_pipeline.sh" >/dev/null; then
      alive=1
    fi

    if [[ "${alive}" == "0" ]]; then
      echo "[error] ${label}: summary missing and no matching process is alive"
      exit 1
    fi

    echo "[wait] ${label}: existing chain still running"
    sleep "${POLL_SEC}"
  done
}

run_pretrain_sync() {
  local config_path="$1"
  local exp_name="$2"
  local tags="$3"
  local master_port="$4"
  local extra_args="${5:-}"

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
  CONFIG_PATH="${config_path}" \
  EXP_NAME="${exp_name}" \
  EXTRA_ARGS="${extra_args}" \
  "${PRETRAIN_WRAPPER}"
}

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local ckpt_path="$3"
  local tags="$4"
  local master_port="$5"
  local ft_recon_weight="$6"
  local extra_args="${7:-}"
  local meta_path="${8:-}"

  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${FT_WANDB_PROJECT}" \
  WANDB_GROUP="${FT_WANDB_GROUP}" \
  WANDB_RUN_NAME="${exp_name}" \
  WANDB_TAGS="${tags}" \
  WANDB_LOG_EVERY="${WANDB_LOG_EVERY}" \
  WANDB_DIR="${WANDB_DIR}" \
  FT_RECON_WEIGHT="${ft_recon_weight}" \
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
  EXTRA_ARGS="${extra_args}" \
  "${FINETUNE_WRAPPER}"
}

ckpt_epoch_if_exists() {
  local ckpt_path="$1"
  if [[ -f "${ckpt_path}" ]]; then
    ckpt_epoch "${ckpt_path}"
  else
    printf '%s\n' "-1"
  fi
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
  local ft_recon_weight="$9"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"
  exp_path="$(resolve_existing_exp_path "${exp_name}" "${exp_path}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch
  epoch="$(ckpt_epoch_if_exists "${ckpt_path}")"

  if (( epoch >= expected_epoch )); then
    echo "[done] ${label} already complete: epoch=${epoch}" >&2
    return 0
  fi

  if pgrep -af -- "--exp_name ${exp_name}" >/dev/null; then
    wait_for_exp_completion "${label}_existing" "${exp_name}" "${exp_path}" "${expected_epoch}"
    return 0
  fi

  local extra_args=""
  if (( epoch >= 0 )); then
    extra_args="--resume"
    echo "[resume] ${label}: epoch=${epoch} exp_name=${exp_name}" >&2
  else
    echo "[launch] ${label}: exp_name=${exp_name}" >&2
  fi

  run_finetune_sync \
    "${config_path}" \
    "${exp_name}" \
    "${source_ckpt_path}" \
    "${tags}" \
    "${master_port}" \
    "${ft_recon_weight}" \
    "${extra_args}" \
    "${meta_path}"
  load_meta_for_exp "${exp_name}"
  exp_path="${EXPERIMENT_PATH}"
  wait_for_exp_completion "${label}" "${exp_name}" "${exp_path}" "${expected_epoch}"
}

ensure_cdl12_pretrain() {
  local exp_path
  exp_path="$(exp_path_from_cfg "${CDL12_CONFIG_PATH}" "${CDL12_EXP_NAME}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch=-1

  if [[ -f "${ckpt_path}" ]]; then
    epoch="$(ckpt_epoch "${ckpt_path}")"
  fi

  if (( epoch >= CDL12_MAX_EPOCH )); then
    echo "[done] cdl12_pretrain already complete: epoch=${epoch}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  if pgrep -af -- "--exp_name ${CDL12_EXP_NAME}" >/dev/null; then
    wait_for_exp_completion "cdl12_pretrain_existing" "${CDL12_EXP_NAME}" "${exp_path}" "${CDL12_MAX_EPOCH}" >&2
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  local extra_args=""
  if [[ -f "${ckpt_path}" ]]; then
    extra_args="--resume"
    echo "[resume] cdl12_pretrain: epoch=${epoch} exp_name=${CDL12_EXP_NAME}" >&2
  else
    echo "[launch] cdl12_pretrain: exp_name=${CDL12_EXP_NAME}" >&2
  fi

  run_pretrain_sync \
    "${CDL12_CONFIG_PATH}" \
    "${CDL12_EXP_NAME}" \
    "pointgpt,pretrain,cdl12,ddp2,full,online" \
    "29557" \
    "${extra_args}"
  wait_for_exp_completion "cdl12_pretrain" "${CDL12_EXP_NAME}" "${exp_path}" "${CDL12_MAX_EPOCH}" >&2
  printf '%s\n' "${ckpt_path}"
}

run_ft_chain() {
  local source_label="$1"
  local ft_label="$2"
  local ckpt_path="$3"
  local ft_recon_weight="$4"

  local obj_bg_exp_name="pointgpt_ft_objbg_from_${source_label}_${ft_label}_${RUN_TAG}"
  local obj_only_exp_name="pointgpt_ft_objonly_from_${source_label}_${ft_label}_${RUN_TAG}"
  local hardest_exp_name="pointgpt_ft_hardest_from_${source_label}_${ft_label}_${RUN_TAG}"

  local obj_bg_exp_path
  local obj_only_exp_path
  local hardest_exp_path
  obj_bg_exp_path="$(exp_path_from_cfg "${OBJ_BG_CONFIG_PATH}" "${obj_bg_exp_name}")"
  obj_only_exp_path="$(exp_path_from_cfg "${OBJ_ONLY_CONFIG_PATH}" "${obj_only_exp_name}")"
  hardest_exp_path="$(exp_path_from_cfg "${HARDEST_CONFIG_PATH}" "${hardest_exp_name}")"

  ensure_finetune_stage \
    "${source_label}_${ft_label}_objbg" \
    "${OBJ_BG_CONFIG_PATH}" \
    "${obj_bg_exp_name}" \
    "${obj_bg_exp_path}" \
    "${FT_MAX_EPOCH_OBJ_BG}" \
    "${ckpt_path}" \
    "pointgpt,finetune,objbg,ddp2,${source_label},${ft_label}" \
    "29551" \
    "${ft_recon_weight}"

  ensure_finetune_stage \
    "${source_label}_${ft_label}_objonly" \
    "${OBJ_ONLY_CONFIG_PATH}" \
    "${obj_only_exp_name}" \
    "${obj_only_exp_path}" \
    "${FT_MAX_EPOCH_OBJ_ONLY}" \
    "${ckpt_path}" \
    "pointgpt,finetune,objonly,ddp2,${source_label},${ft_label}" \
    "29553" \
    "${ft_recon_weight}"

  ensure_finetune_stage \
    "${source_label}_${ft_label}_hardest" \
    "${HARDEST_CONFIG_PATH}" \
    "${hardest_exp_name}" \
    "${hardest_exp_path}" \
    "${FT_MAX_EPOCH_HARDEST}" \
    "${ckpt_path}" \
    "pointgpt,finetune,hardest,ddp2,${source_label},${ft_label}" \
    "29555" \
    "${ft_recon_weight}"

  local arm_summary_path="${LOG_ROOT}/${source_label}_${ft_label}_${RUN_TAG}_summary.md"
  cat > "${arm_summary_path}" <<EOF
# PointGPT FT Arm Summary

- run_tag: \`${RUN_TAG}\`
- source_label: \`${source_label}\`
- ft_label: \`${ft_label}\`
- source_ckpt: \`${ckpt_path}\`
- ft_recon_weight: \`${ft_recon_weight}\`

## Results

- obj_bg:
  - exp: \`${obj_bg_exp_name}\`
  - path: \`${obj_bg_exp_path}\`
  - best_acc: \`$(best_acc "${obj_bg_exp_path}/ckpt-last.pth")\`
- objonly:
  - exp: \`${obj_only_exp_name}\`
  - path: \`${obj_only_exp_path}\`
  - best_acc: \`$(best_acc "${obj_only_exp_path}/ckpt-last.pth")\`
- hardest:
  - exp: \`${hardest_exp_name}\`
  - path: \`${hardest_exp_path}\`
  - best_acc: \`$(best_acc "${hardest_exp_path}/ckpt-last.pth")\`
EOF

  echo "[done] arm summary written to ${arm_summary_path}"
}

existing_nepa_cls_only_results() {
  local run_tag="$1"
  local summary_path="${NEPA_CLS_ONLY_LOG_ROOT}/${run_tag}_summary.md"
  local obj_bg_exp_name="pointgpt_ft_objbg_from_nepa_cosine_${run_tag}"
  local obj_only_exp_name="pointgpt_ft_objonly_from_nepa_cosine_${run_tag}"
  local hardest_exp_name="pointgpt_ft_hardest_from_nepa_cosine_${run_tag}"

  local obj_bg_exp_path
  local obj_only_exp_path
  local hardest_exp_path
  obj_bg_exp_path="$(exp_path_from_cfg "${OBJ_BG_CONFIG_PATH}" "${obj_bg_exp_name}")"
  obj_only_exp_path="$(exp_path_from_cfg "${OBJ_ONLY_CONFIG_PATH}" "${obj_only_exp_name}")"
  hardest_exp_path="$(exp_path_from_cfg "${HARDEST_CONFIG_PATH}" "${hardest_exp_name}")"

  wait_for_existing_chain_summary "nepa_cls_only_existing" "${summary_path}" "${run_tag}"

  local arm_summary_path="${LOG_ROOT}/nepa_cosine_clsonly_${RUN_TAG}_summary.md"
  cat > "${arm_summary_path}" <<EOF
# PointGPT FT Arm Summary

- run_tag: \`${RUN_TAG}\`
- source_label: \`nepa_cosine\`
- ft_label: \`clsonly\`
- source_ckpt: \`${NEPA_CKPT_PATH}\`
- reused_run_tag: \`${run_tag}\`

## Results

- obj_bg:
  - exp: \`${obj_bg_exp_name}\`
  - path: \`${obj_bg_exp_path}\`
  - best_acc: \`$(best_acc "${obj_bg_exp_path}/ckpt-last.pth")\`
- objonly:
  - exp: \`${obj_only_exp_name}\`
  - path: \`${obj_only_exp_path}\`
  - best_acc: \`$(best_acc "${obj_only_exp_path}/ckpt-last.pth")\`
- hardest:
  - exp: \`${hardest_exp_name}\`
  - path: \`${hardest_exp_path}\`
  - best_acc: \`$(best_acc "${hardest_exp_path}/ckpt-last.pth")\`
EOF

  echo "[done] reused arm summary written to ${arm_summary_path}"
}

echo "=== POINTGPT FT RECIPE MATRIX 2x2 ==="
echo "date=$(date -Is)"
echo "run_tag=${RUN_TAG}"
echo "nepa_ckpt_path=${NEPA_CKPT_PATH}"
echo "nepa_cls_only_run_tag=${NEPA_CLS_ONLY_RUN_TAG:-<none>}"
echo "cdl12_exp_name=${CDL12_EXP_NAME}"
echo "save_last_every_epoch=${SAVE_LAST_EVERY_EPOCH}"
echo

if [[ -n "${NEPA_CLS_ONLY_RUN_TAG}" ]]; then
  existing_nepa_cls_only_results "${NEPA_CLS_ONLY_RUN_TAG}"
else
  run_ft_chain "nepa_cosine" "clsonly" "${NEPA_CKPT_PATH}" "0"
fi

run_ft_chain "nepa_cosine" "pointgptft" "${NEPA_CKPT_PATH}" "3"

CDL12_CKPT_PATH="$(ensure_cdl12_pretrain)"
run_ft_chain "cdl12" "clsonly" "${CDL12_CKPT_PATH}" "0"
run_ft_chain "cdl12" "pointgptft" "${CDL12_CKPT_PATH}" "3"

MATRIX_SUMMARY_PATH="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${MATRIX_SUMMARY_PATH}" <<EOF
# PointGPT FT Recipe Matrix 2x2 Summary

- run_tag: \`${RUN_TAG}\`
- date: \`$(date -Is)\`
- nepa_ckpt_path: \`${NEPA_CKPT_PATH}\`
- cdl12_ckpt_path: \`${CDL12_CKPT_PATH}\`
- reused_nepa_cls_only_run_tag: \`${NEPA_CLS_ONLY_RUN_TAG:-}\`

## Arms

- nepa_cosine x cls_only:
  - summary: \`${LOG_ROOT}/nepa_cosine_clsonly_${RUN_TAG}_summary.md\`
- nepa_cosine x pointgpt_ft:
  - summary: \`${LOG_ROOT}/nepa_cosine_pointgptft_${RUN_TAG}_summary.md\`
- cdl12 x cls_only:
  - summary: \`${LOG_ROOT}/cdl12_clsonly_${RUN_TAG}_summary.md\`
- cdl12 x pointgpt_ft:
  - summary: \`${LOG_ROOT}/cdl12_pointgptft_${RUN_TAG}_summary.md\`
EOF

DONE_PATH="${MATRIX_SUMMARY_PATH}.done"
cat > "${DONE_PATH}" <<EOF
run_tag=${RUN_TAG}
date=$(date -Is)
summary=${MATRIX_SUMMARY_PATH}
EOF

echo "[done] matrix summary written to ${MATRIX_SUMMARY_PATH}"
