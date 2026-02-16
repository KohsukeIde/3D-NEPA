#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SPLIT="${SPLIT:-eval}"
EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_GALLERY="${EVAL_SEED_GALLERY:-999}"
MAX_FILES="${MAX_FILES:-1000}"
MAX_SHAPES="${MAX_SHAPES:-800}"
N_CONTEXT="${N_CONTEXT:-256}"
N_QUERY="${N_QUERY:-256}"
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-udfgrid}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-4000}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
TAU="${TAU:-0.03}"
WAIT_SEC="${WAIT_SEC:-120}"
RESULT_DIR="${RESULT_DIR:-results}"
TIE_BREAK_EPS="${TIE_BREAK_EPS:-1e-6}"

MODEL1_TAG="${MODEL1_TAG:-kplane_sum_s0_e50}"
MODEL1_CKPT="${MODEL1_CKPT:-runs/eccv_kplane_sum_s0/ckpt_ep049.pt}"
MODEL1_GPU="${MODEL1_GPU:-0}"

MODEL2_TAG="${MODEL2_TAG:-kplane_sum_large_s0_e50}"
MODEL2_CKPT="${MODEL2_CKPT:-runs/eccv_kplane_sum_large_s0/ckpt_ep049.pt}"
MODEL2_GPU="${MODEL2_GPU:-1}"

mkdir -p "${RESULT_DIR}"

ts() { date +"%F %T"; }

wait_for_ckpt() {
  local ckpt="$1"
  local tag="$2"
  while [ ! -f "${ckpt}" ]; do
    echo "[$(ts)] waiting ${tag}: ${ckpt}"
    sleep "${WAIT_SEC}"
  done
  echo "[$(ts)] found ${tag}: ${ckpt}"
}

run_model_eval() {
  local tag="$1"
  local ckpt="$2"
  local gpu="$3"

  echo "[$(ts)] [${tag}] start evaluations on GPU${gpu}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --query_backend mesh --gallery_backend udfgrid \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --eval_seed "${EVAL_SEED}" --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
    --max_files "${MAX_FILES}" --pooling mean_query \
    --tie_break_eps "${TIE_BREAK_EPS}" \
    --out_json "${RESULT_DIR}/ucpr_${tag}_mesh2udf_1k_mean_query_tiefix.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --eval_seed "${EVAL_SEED}" --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
    --max_files "${MAX_FILES}" --pooling mean_query \
    --tie_break_eps "${TIE_BREAK_EPS}" \
    --out_json "${RESULT_DIR}/ucpr_${tag}_mesh2pc_1k_mean_query_tiefix.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --query_source pool --baseline nn_copy \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "${RESULT_DIR}/cpac_${tag}_pc2udf_800_pool_htrain4k_with_nncopy.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test none \
    --query_source pool --baseline none \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "${RESULT_DIR}/cpac_${tag}_pc2udf_800_testnone_h_htrain4k.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test mismatch \
    --mismatch_shift 1 --query_source pool --baseline none \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "${RESULT_DIR}/cpac_${tag}_pc2udf_800_testmismatch_h_htrain4k.json"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_kplane \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_CONTEXT}" --n_query "${N_QUERY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --query_source grid --baseline nn_copy \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "${RESULT_DIR}/cpac_${tag}_pc2udf_800_grid_h_htrain4k_with_nncopy.json"

  echo "[$(ts)] [${tag}] evaluations finished"
}

echo "[$(ts)] wait start"
wait_for_ckpt "${MODEL1_CKPT}" "${MODEL1_TAG}"
wait_for_ckpt "${MODEL2_CKPT}" "${MODEL2_TAG}"

run_model_eval "${MODEL1_TAG}" "${MODEL1_CKPT}" "${MODEL1_GPU}" &
pid0=$!
run_model_eval "${MODEL2_TAG}" "${MODEL2_CKPT}" "${MODEL2_GPU}" &
pid1=$!

wait "${pid0}" "${pid1}"
echo "[$(ts)] all evaluations finished"
