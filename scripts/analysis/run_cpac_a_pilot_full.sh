#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
CKPT="${CKPT:-runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt}"

SPLIT="${SPLIT:-eval}"
CONTEXT_BACKEND="${CONTEXT_BACKEND:-pointcloud_noray}"
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-udfgrid}"

N_CONTEXT="${N_CONTEXT:-256}"
N_QUERY="${N_QUERY:-256}"
MAX_SHAPES="${MAX_SHAPES:-800}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-4000}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
TAU="${TAU:-0.03}"
DISJOINT_CONTEXT_QUERY="${DISJOINT_CONTEXT_QUERY:-1}"
REP_SOURCE="${REP_SOURCE:-h}"

# Space separated, e.g. "0 1 2"
EVAL_SEEDS="${EVAL_SEEDS:-0}"
OUT_DIR="${OUT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs/analysis/cpac_a_pilot_full}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

run_one() {
  local seed="$1"
  local tag="$2"
  shift 2
  local out_json="${OUT_DIR}/cpac_nepa_qa_dualmask_s0_pc2udf_800_${tag}_seed${seed}.json"
  local log_path="${LOG_DIR}/${tag}_seed${seed}.log"

  if [ -f "${out_json}" ]; then
    echo "[skip] exists: ${out_json}"
    return 0
  fi

  echo "[run] seed=${seed} tag=${tag}"
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
      --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
      --ckpt "${CKPT}" \
      --context_backend "${CONTEXT_BACKEND}" \
      --head_train_split "${HEAD_TRAIN_SPLIT}" \
      --head_train_backend "${HEAD_TRAIN_BACKEND}" \
      --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
      --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
      --disjoint_context_query "${DISJOINT_CONTEXT_QUERY}" \
      --context_mode_train normal --context_mode_test normal \
      --rep_source "${REP_SOURCE}" \
      --baseline nn_copy \
      --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
      --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" \
      --eval_seed "${seed}" \
      "$@" \
      --out_json "${out_json}" | tee "${log_path}"
}

for seed in ${EVAL_SEEDS}; do
  # A-baselines
  run_one "${seed}" "pool_uniform" \
    --query_source pool \
    --grid_sample_mode uniform

  run_one "${seed}" "grid_uniform" \
    --query_source grid \
    --grid_sample_mode uniform

  run_one "${seed}" "grid_near08" \
    --query_source grid \
    --grid_sample_mode near_surface \
    --grid_near_tau 0.05 --grid_near_frac 0.8

  run_one "${seed}" "hybrid50_uniform" \
    --query_source hybrid --query_pool_frac 0.5 \
    --grid_sample_mode uniform

  run_one "${seed}" "hybrid50_near08" \
    --query_source hybrid --query_pool_frac 0.5 \
    --grid_sample_mode near_surface \
    --grid_near_tau 0.05 --grid_near_frac 0.8

  # A-3 (aux): truncation target transform
  run_one "${seed}" "hybrid50_near08_trunc01" \
    --query_source hybrid --query_pool_frac 0.5 \
    --grid_sample_mode near_surface \
    --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --target_transform trunc --target_trunc_max 0.1
done

echo "[done] cpac A-pilot full matrix"
