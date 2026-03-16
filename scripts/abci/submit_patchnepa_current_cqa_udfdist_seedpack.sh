#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SEEDS_CSV="${SEEDS_CSV:-1,2}"
BASE_RUN_SET="${BASE_RUN_SET:-patchnepa_cqa_udfdist_seedpack_${STAMP}}"

parse_csv() {
  local text="$1"
  local out=()
  IFS=',' read -r -a out <<< "${text}"
  for x in "${out[@]}"; do
    x="$(echo "${x}" | xargs)"
    [[ -n "${x}" ]] && echo "${x}"
  done
}

submit_and_capture_job() {
  local cmd="$1"
  local out
  out="$(eval "${cmd}")"
  printf '%s\n' "${out}" >&2
  printf '%s\n' "${out}" | awk '/^\[submitted\] / {print $2}' | tail -n 1
}

for seed in $(parse_csv "${SEEDS_CSV}"); do
  curve_run_set="${BASE_RUN_SET}_curve_seed${seed}"
  curve_run_tag="cqa_udfdist_worldv3_g2_s10000_seed${seed}"
  curve_cmd=$(
    cat <<EOF
env RUN_SET="${curve_run_set}" RUN_TAG="${curve_run_tag}" SEED="${seed}" \
WANDB_PROJECT="${WANDB_PROJECT_CURVE:-patchnepa-cqa-curve-seeds}" \
WANDB_GROUP="${WANDB_GROUP_CURVE:-${BASE_RUN_SET}}" \
WANDB_RUN_NAME="${curve_run_tag}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_curve.sh
EOF
  )
  curve_job="$(submit_and_capture_job "${curve_cmd}")"
  if [[ -z "${curve_job}" ]]; then
    echo "[error] failed to capture curve job id for seed=${seed}" >&2
    exit 1
  fi

  ckpt_path="runs/cqa/${curve_run_set}/${curve_run_tag}/ckpt_final.pt"
  offdiag_run_set="${BASE_RUN_SET}_offdiag_seed${seed}"
  offdiag_run_tag="cqa_udfdist_offdiag_eval_seed${seed}"
  offdiag_cmd=$(
    cat <<EOF
env RUN_SET="${offdiag_run_set}" RUN_TAG="${offdiag_run_tag}" SEED="${seed}" \
CKPT="${ckpt_path}" QSUB_DEPEND="afterok:${curve_job}" \
WANDB_PROJECT="${WANDB_PROJECT_OFFDIAG:-patchnepa-cqa-offdiag-seeds}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_offdiag.sh
EOF
  )
  offdiag_job="$(submit_and_capture_job "${offdiag_cmd}")"
  if [[ -z "${offdiag_job}" ]]; then
    echo "[error] failed to capture offdiag job id for seed=${seed}" >&2
    exit 1
  fi

  echo "[seed=${seed}] curve_job=${curve_job} offdiag_job=${offdiag_job} ckpt=${ckpt_path}"
done
