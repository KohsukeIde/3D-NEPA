#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
POST_WRAPPER="${POST_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_post_pretrain_local_ddp.sh}"
FINETUNE_WRAPPER="${FINETUNE_WRAPPER:-${WORKDIR}/scripts/local/pointgpt_finetune_local_ddp.sh}"
LHY_CHECKER="${LHY_CHECKER:-${WORKDIR}/scripts/local/pointgpt_labeledhybrid_status.sh}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

RUN_TAG="${RUN_TAG:-pointgpt_postpretrain_effect_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/local/pointgpt_postpretrain_effect}"
POLL_SEC="${POLL_SEC:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

HYBRID_ROOT="${HYBRID_ROOT:-${POINTGPT_DIR}/data/HybridDatasets}"
DATA_ROOT="${DATA_ROOT:-${HYBRID_ROOT}/post_pretrain}"
PC_PATH="${PC_PATH:-${HYBRID_ROOT}}"

NEPA_CKPT_PATH="${NEPA_CKPT_PATH:-${POINTGPT_DIR}/experiments/pretrain_nepa_cosine_shapenet_cache_v0/PointGPT-B/pointgpt_nepa_cosine_shapenet_cache_v0_online_retry_20260306_234205/ckpt-last.pth}"
CDL12_CKPT_PATH="${CDL12_CKPT_PATH:-${POINTGPT_DIR}/experiments/pretrain_cdl12_shapenet_cache_v0/PointGPT-B/pointgpt_cdl12_shapenet_cache_v0_online_pointgpt_ft_recipe_matrix_2x2_20260311_153835/ckpt-last.pth}"
POST_CONFIG_PATH="${POST_CONFIG_PATH:-cfgs/PointGPT-B/post_pretrain.yaml}"
POST_MAX_EPOCH="${POST_MAX_EPOCH:-100}"

OBJ_BG_CONFIG_PATH="${OBJ_BG_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objbg.yaml}"
OBJ_ONLY_CONFIG_PATH="${OBJ_ONLY_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_objonly.yaml}"
HARDEST_CONFIG_PATH="${HARDEST_CONFIG_PATH:-cfgs/PointGPT-B/finetune_scan_hardest.yaml}"
FT_MAX_EPOCH_OBJ_BG="${FT_MAX_EPOCH_OBJ_BG:-30}"
FT_MAX_EPOCH_OBJ_ONLY="${FT_MAX_EPOCH_OBJ_ONLY:-50}"
FT_MAX_EPOCH_HARDEST="${FT_MAX_EPOCH_HARDEST:-30}"
FT_VAL_FREQ="${FT_VAL_FREQ:-1}"
FT_RECON_WEIGHT="${FT_RECON_WEIGHT:-3}"
SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
NO_TEST_AS_VAL="${NO_TEST_AS_VAL:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
POST_WANDB_PROJECT="${POST_WANDB_PROJECT:-pointgpt-postpretrain}"
POST_WANDB_GROUP="${POST_WANDB_GROUP:-pointgpt_postpretrain_effect}"
FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-pointgpt-transfer}"
FT_WANDB_GROUP="${FT_WANDB_GROUP:-pointgpt_postpretrain_effect}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-10}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

mkdir -p "${LOG_ROOT}"

if [[ ! -x "${POST_WRAPPER}" ]]; then
  echo "[error] post wrapper missing or not executable: ${POST_WRAPPER}"
  exit 2
fi
if [[ ! -x "${FINETUNE_WRAPPER}" ]]; then
  echo "[error] finetune wrapper missing or not executable: ${FINETUNE_WRAPPER}"
  exit 2
fi
if [[ ! -x "${LHY_CHECKER}" ]]; then
  echo "[error] labeledhybrid checker missing or not executable: ${LHY_CHECKER}"
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
if [[ ! -f "${CDL12_CKPT_PATH}" ]]; then
  echo "[error] cdl12 checkpoint missing: ${CDL12_CKPT_PATH}"
  exit 2
fi

POINTGPT_DIR="${POINTGPT_DIR}" \
HYBRID_ROOT="${HYBRID_ROOT}" \
DATA_ROOT="${DATA_ROOT}" \
PC_PATH="${PC_PATH}" \
"${LHY_CHECKER}" >/dev/null

exp_path_from_cfg() {
  local cfg_path="$1"
  local exp_name="$2"
  local cfg_stem
  local cfg_parent
  cfg_stem="$(basename "${cfg_path%.*}")"
  cfg_parent="$(basename "$(dirname "${cfg_path}")")"
  printf '%s/experiments/%s/%s/%s\n' "${POINTGPT_DIR}" "${cfg_stem}" "${cfg_parent}" "${exp_name}"
}

normalize_exp_path() {
  local path="$1"
  if [[ "${path}" == ./* ]]; then
    printf '%s/%s\n' "${POINTGPT_DIR}" "${path#./}"
  elif [[ "${path}" != /* ]]; then
    printf '%s/%s\n' "${POINTGPT_DIR}" "${path}"
  else
    printf '%s\n' "${path}"
  fi
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
    normalize_exp_path "${EXPERIMENT_PATH}"
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

  normalize_exp_path "${fallback}"
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

ckpt_epoch_if_exists() {
  local ckpt_path="$1"
  if [[ -f "${ckpt_path}" ]]; then
    ckpt_epoch "${ckpt_path}"
  else
    printf '%s\n' "-1"
  fi
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
  local expected_epoch="$4"
  exp_path="$(normalize_exp_path "${exp_path}")"
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

run_post_sync() {
  local config_path="$1"
  local exp_name="$2"
  local source_ckpt="$3"
  local tags="$4"
  local master_port="$5"
  local extra_args="${6:-}"
  local meta_path="${7:-}"

  HYBRID_ROOT="${HYBRID_ROOT}" \
  DATA_ROOT="${DATA_ROOT}" \
  PC_PATH="${PC_PATH}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  WANDB_MODE="${WANDB_MODE}" \
  WANDB_PROJECT="${POST_WANDB_PROJECT}" \
  WANDB_GROUP="${POST_WANDB_GROUP}" \
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
  CKPT_PATH="${source_ckpt}" \
  RUNTIME_META_PATH="${meta_path}" \
  EXTRA_ARGS="${extra_args}" \
  "${POST_WRAPPER}"
}

run_finetune_sync() {
  local config_path="$1"
  local exp_name="$2"
  local ckpt_path="$3"
  local tags="$4"
  local master_port="$5"
  local meta_path="${6:-}"

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
  NO_TEST_AS_VAL="${NO_TEST_AS_VAL}" \
  VAL_RATIO="${VAL_RATIO}" \
  VAL_SEED="${VAL_SEED}" \
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

ensure_post_stage() {
  local label="$1"
  local source_label="$2"
  local source_ckpt="$3"
  local exp_name="pointgpt_postpretrain_from_${source_label}_${RUN_TAG}"
  local exp_path
  exp_path="$(exp_path_from_cfg "${POST_CONFIG_PATH}" "${exp_name}")"
  exp_path="$(resolve_existing_exp_path "${exp_name}" "${exp_path}")"
  local ckpt_path="${exp_path}/ckpt-last.pth"
  local epoch
  epoch="$(ckpt_epoch_if_exists "${ckpt_path}")"
  local meta_path
  meta_path="$(meta_path_for_exp "${exp_name}")"

  if (( epoch >= POST_MAX_EPOCH )); then
    echo "[done] ${label} already complete: epoch=${epoch}"
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  if pgrep -af -- "--exp_name ${exp_name}" >/dev/null; then
    wait_for_exp_completion "${label}_existing" "${exp_name}" "${exp_path}" "${POST_MAX_EPOCH}"
    printf '%s\n' "${ckpt_path}"
    return 0
  fi

  local extra_args=""
  if (( epoch >= 0 )); then
    extra_args="--resume"
    echo "[resume] ${label}: epoch=${epoch} exp_name=${exp_name}"
  else
    echo "[launch] ${label}: exp_name=${exp_name}"
  fi

  run_post_sync \
    "${POST_CONFIG_PATH}" \
    "${exp_name}" \
    "${source_ckpt}" \
    "pointgpt,postpretrain,${source_label},ddp2" \
    "29571" \
    "${extra_args}" \
    "${meta_path}"
  load_meta_for_exp "${exp_name}"
  wait_for_exp_completion "${label}" "${exp_name}" "${EXPERIMENT_PATH}" "${POST_MAX_EPOCH}"
  printf '%s\n' "$(normalize_exp_path "${EXPERIMENT_PATH}")/ckpt-last.pth"
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
  local epoch
  epoch="$(ckpt_epoch_if_exists "${ckpt_path}")"

  if (( epoch >= expected_epoch )); then
    echo "[done] ${label} already complete: epoch=${epoch}"
    return 0
  fi

  if pgrep -af -- "--exp_name ${exp_name}" >/dev/null; then
    wait_for_exp_completion "${label}_existing" "${exp_name}" "${exp_path}" "${expected_epoch}"
    return 0
  fi

  local extra_args=""
  if (( epoch >= 0 )); then
    extra_args="--resume"
    echo "[resume] ${label}: epoch=${epoch} exp_name=${exp_name}"
  else
    echo "[launch] ${label}: exp_name=${exp_name}"
  fi

  run_finetune_sync \
    "${config_path}" \
    "${exp_name}" \
    "${source_ckpt_path}" \
    "${tags}" \
    "${master_port}" \
    "${meta_path}"
  load_meta_for_exp "${exp_name}"
  wait_for_exp_completion "${label}" "${exp_name}" "${EXPERIMENT_PATH}" "${expected_epoch}"
}

run_ft_chain() {
  local source_label="$1"
  local source_ckpt="$2"

  local obj_bg_exp_name="pointgpt_ft_objbg_from_${source_label}_postpretrain_${RUN_TAG}"
  local obj_only_exp_name="pointgpt_ft_objonly_from_${source_label}_postpretrain_${RUN_TAG}"
  local hardest_exp_name="pointgpt_ft_hardest_from_${source_label}_postpretrain_${RUN_TAG}"

  local obj_bg_exp_path
  local obj_only_exp_path
  local hardest_exp_path
  obj_bg_exp_path="$(exp_path_from_cfg "${OBJ_BG_CONFIG_PATH}" "${obj_bg_exp_name}")"
  obj_only_exp_path="$(exp_path_from_cfg "${OBJ_ONLY_CONFIG_PATH}" "${obj_only_exp_name}")"
  hardest_exp_path="$(exp_path_from_cfg "${HARDEST_CONFIG_PATH}" "${hardest_exp_name}")"

  ensure_finetune_stage \
    "${source_label}_objbg" \
    "${OBJ_BG_CONFIG_PATH}" \
    "${obj_bg_exp_name}" \
    "${obj_bg_exp_path}" \
    "${FT_MAX_EPOCH_OBJ_BG}" \
    "${source_ckpt}" \
    "pointgpt,finetune,objbg,postpretrain,${source_label},ddp2" \
    "29573"

  ensure_finetune_stage \
    "${source_label}_objonly" \
    "${OBJ_ONLY_CONFIG_PATH}" \
    "${obj_only_exp_name}" \
    "${obj_only_exp_path}" \
    "${FT_MAX_EPOCH_OBJ_ONLY}" \
    "${source_ckpt}" \
    "pointgpt,finetune,objonly,postpretrain,${source_label},ddp2" \
    "29575"

  ensure_finetune_stage \
    "${source_label}_hardest" \
    "${HARDEST_CONFIG_PATH}" \
    "${hardest_exp_name}" \
    "${hardest_exp_path}" \
    "${FT_MAX_EPOCH_HARDEST}" \
    "${source_ckpt}" \
    "pointgpt,finetune,hardest,postpretrain,${source_label},ddp2" \
    "29577"

  local arm_summary_path="${LOG_ROOT}/${source_label}_postpretrain_${RUN_TAG}_summary.md"
  cat > "${arm_summary_path}" <<EOF
# PointGPT Post-Pretrain Effect Arm Summary

- run_tag: \`${RUN_TAG}\`
- source_label: \`${source_label}\`
- source_ckpt: \`${source_ckpt}\`
- ft_recon_weight: \`${FT_RECON_WEIGHT}\`
- no_test_as_val: \`${NO_TEST_AS_VAL}\`

## Results

- obj_bg:
  - exp: \`${obj_bg_exp_name}\`
  - path: \`$(resolve_existing_exp_path "${obj_bg_exp_name}" "${obj_bg_exp_path}")\`
  - best_acc: \`$(best_acc "$(resolve_existing_exp_path "${obj_bg_exp_name}" "${obj_bg_exp_path}")/ckpt-last.pth")\`
- objonly:
  - exp: \`${obj_only_exp_name}\`
  - path: \`$(resolve_existing_exp_path "${obj_only_exp_name}" "${obj_only_exp_path}")\`
  - best_acc: \`$(best_acc "$(resolve_existing_exp_path "${obj_only_exp_name}" "${obj_only_exp_path}")/ckpt-last.pth")\`
- hardest:
  - exp: \`${hardest_exp_name}\`
  - path: \`$(resolve_existing_exp_path "${hardest_exp_name}" "${hardest_exp_path}")\`
  - best_acc: \`$(best_acc "$(resolve_existing_exp_path "${hardest_exp_name}" "${hardest_exp_path}")/ckpt-last.pth")\`
EOF

  echo "[done] arm summary written to ${arm_summary_path}"
}

echo "=== POINTGPT POST-PRETRAIN EFFECT PIPELINE ==="
echo "date=$(date -Is)"
echo "run_tag=${RUN_TAG}"
echo "hybrid_root=${HYBRID_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "pc_path=${PC_PATH}"
echo "no_test_as_val=${NO_TEST_AS_VAL}"
echo

cdl12_post_ckpt="$(ensure_post_stage "cdl12_postpretrain" "cdl12" "${CDL12_CKPT_PATH}")"
run_ft_chain "cdl12" "${cdl12_post_ckpt}"

nepa_post_ckpt="$(ensure_post_stage "nepa_postpretrain" "nepa_cosine" "${NEPA_CKPT_PATH}")"
run_ft_chain "nepa_cosine" "${nepa_post_ckpt}"

summary_path="${LOG_ROOT}/${RUN_TAG}_summary.md"
cat > "${summary_path}" <<EOF
# PointGPT Post-Pretrain Effect Summary

- run_tag: \`${RUN_TAG}\`
- date: \`$(date -Is)\`
- hybrid_root: \`${HYBRID_ROOT}\`
- data_root: \`${DATA_ROOT}\`
- pc_path: \`${PC_PATH}\`
- ft_recon_weight: \`${FT_RECON_WEIGHT}\`
- no_test_as_val: \`${NO_TEST_AS_VAL}\`

## Arms

- cdl12 + post-pretrain:
  - summary: \`${LOG_ROOT}/cdl12_postpretrain_${RUN_TAG}_summary.md\`
- nepa_cosine + post-pretrain:
  - summary: \`${LOG_ROOT}/nepa_cosine_postpretrain_${RUN_TAG}_summary.md\`
EOF

echo "[done] summary written to ${summary_path}"
