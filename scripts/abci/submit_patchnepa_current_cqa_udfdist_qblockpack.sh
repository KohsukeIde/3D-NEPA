#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_SET="${BASE_RUN_SET:-patchnepa_cqa_udfdist_qblockpack_${STAMP}}"
GRID_RES="${GRID_RES:-16}"
MAX_SHAPES="${MAX_SHAPES:-64}"

submit_and_capture_job() {
  local cmd="$1"
  local out
  out="$(eval "${cmd}")"
  printf '%s\n' "${out}" >&2
  printf '%s\n' "${out}" | awk '/^\[submitted\] / {print $2}' | tail -n 1
}

submit_mode() {
  local mode="$1"
  local train_script
  local train_tag
  if [[ "${mode}" == "selfq" ]]; then
    train_script="scripts/abci/submit_patchnepa_current_cqa_udfdist_curve_selfq.sh"
    train_tag="cqa_udfdist_worldv3_independent_selfq_g2_s10000"
  elif [[ "${mode}" == "noq" ]]; then
    train_script="scripts/abci/submit_patchnepa_current_cqa_udfdist_curve_noq.sh"
    train_tag="cqa_udfdist_worldv3_independent_noq_g2_s10000"
  else
    echo "[error] unknown mode=${mode}" >&2
    exit 2
  fi

  local train_run_set="${BASE_RUN_SET}_${mode}"
  local train_cmd
  train_cmd=$(
    cat <<EOF
env RUN_SET="${train_run_set}" RUN_TAG="${train_tag}" SEED="${SEED:-0}" \
WANDB_PROJECT="${WANDB_PROJECT_CURVE:-patchnepa-cqa-qblock}" \
WANDB_GROUP="${WANDB_GROUP_CURVE:-${BASE_RUN_SET}}" \
WANDB_RUN_NAME="${train_tag}" \
bash "${train_script}"
EOF
  )
  local train_job
  train_job="$(submit_and_capture_job "${train_cmd}")"
  if [[ -z "${train_job}" ]]; then
    echo "[error] failed to capture train job id for mode=${mode}" >&2
    exit 1
  fi

  local ckpt_path="runs/cqa/${train_run_set}/${train_tag}/ckpt_final.pt"

  local offdiag_cmd
  offdiag_cmd=$(
    cat <<EOF
env RUN_SET="${BASE_RUN_SET}_${mode}_offdiag" RUN_TAG="cqa_udfdist_offdiag_eval_${mode}" \
CKPT="${ckpt_path}" QSUB_DEPEND="afterok:${train_job}" SEED="${SEED:-0}" \
WANDB_PROJECT="${WANDB_PROJECT_OFFDIAG:-patchnepa-cqa-qblock}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_offdiag.sh
EOF
  )
  local offdiag_job
  offdiag_job="$(submit_and_capture_job "${offdiag_cmd}")"
  if [[ -z "${offdiag_job}" ]]; then
    echo "[error] failed to capture offdiag job id for mode=${mode}" >&2
    exit 1
  fi

  local same_cmp_cmd
  same_cmp_cmd=$(
    cat <<EOF
env RUN_SET="${BASE_RUN_SET}_${mode}_same_completion" RUN_TAG="cqa_udfdist_same_translation_g${GRID_RES}_s${MAX_SHAPES}_${mode}_assets" \
MODE="same" CKPT="${ckpt_path}" QSUB_DEPEND="afterok:${train_job}" \
GRID_RES="${GRID_RES}" MAX_SHAPES="${MAX_SHAPES}" SEED="${SEED:-0}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh
EOF
  )
  local same_cmp_job
  same_cmp_job="$(submit_and_capture_job "${same_cmp_cmd}")"
  if [[ -z "${same_cmp_job}" ]]; then
    echo "[error] failed to capture same completion job id for mode=${mode}" >&2
    exit 1
  fi

  local off_cmp_cmd
  off_cmp_cmd=$(
    cat <<EOF
env RUN_SET="${BASE_RUN_SET}_${mode}_offdiag_completion" RUN_TAG="cqa_udfdist_offdiag_translation_g${GRID_RES}_s${MAX_SHAPES}_${mode}_assets" \
MODE="offdiag" CKPT="${ckpt_path}" QSUB_DEPEND="afterok:${train_job}" \
GRID_RES="${GRID_RES}" MAX_SHAPES="${MAX_SHAPES}" SEED="${SEED:-0}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_translation.sh
EOF
  )
  local off_cmp_job
  off_cmp_job="$(submit_and_capture_job "${off_cmp_cmd}")"
  if [[ -z "${off_cmp_job}" ]]; then
    echo "[error] failed to capture offdiag completion job id for mode=${mode}" >&2
    exit 1
  fi

  echo "[mode=${mode}] train_job=${train_job} offdiag_job=${offdiag_job} same_completion_job=${same_cmp_job} offdiag_completion_job=${off_cmp_job} ckpt=${ckpt_path}"
}

submit_mode "selfq"
submit_mode "noq"
