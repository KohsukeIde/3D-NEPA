#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SPLIT="${SPLIT:-eval}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix.yaml}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-200000}"
MIX_SEED="${MIX_SEED:-0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-96}"
NUM_WORKERS="${NUM_WORKERS:-6}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
ADD_EOS="${ADD_EOS:-1}"
QA_TOKENS="${QA_TOKENS:-1}"
DUAL_MASK_NEAR="${DUAL_MASK_NEAR:-0.4}"
DUAL_MASK_FAR="${DUAL_MASK_FAR:-0.1}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-32}"
DUAL_MASK_WARMUP_FRAC="${DUAL_MASK_WARMUP_FRAC:-0.05}"
INCLUDE_PT_GRAD="${INCLUDE_PT_GRAD:-0}"
PT_GRAD_MODE="${PT_GRAD_MODE:-raw}"
PT_GRAD_EPS="${PT_GRAD_EPS:-1e-3}"
PT_GRAD_CLIP="${PT_GRAD_CLIP:-10.0}"
PT_GRAD_ORIENT="${PT_GRAD_ORIENT:-none}"
INCLUDE_RAY_UNC="${INCLUDE_RAY_UNC:-0}"
RAY_UNC_K="${RAY_UNC_K:-8}"
RAY_UNC_MODE="${RAY_UNC_MODE:-normal_var}"

EVAL_SEED="${EVAL_SEED:-0}"
EVAL_SEED_GALLERY="${EVAL_SEED_GALLERY:-999}"
MAX_FILES="${MAX_FILES:-1000}"
MAX_SHAPES="${MAX_SHAPES:-800}"
HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
HEAD_TRAIN_BACKEND="${HEAD_TRAIN_BACKEND:-udfgrid}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-4000}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-3}"
TAU="${TAU:-0.03}"
TIE_BREAK_EPS="${TIE_BREAK_EPS:-1e-6}"

MODEL1_TAG="${MODEL1_TAG:-nepa_impl_causal_s0_e50}"
MODEL1_GPU="${MODEL1_GPU:-0}"
MODEL1_ARCH="${MODEL1_ARCH:-causal}"
MODEL1_QA_LAYOUT="${MODEL1_QA_LAYOUT:-interleave}"
MODEL1_TOPO_K="${MODEL1_TOPO_K:-0}"
MODEL1_TOPO_RAY_COORD="${MODEL1_TOPO_RAY_COORD:-origin}"
MODEL1_TOPO_RAY_BBOX="${MODEL1_TOPO_RAY_BBOX:-0.5}"
MODEL1_SAVE_DIR="${MODEL1_SAVE_DIR:-runs/eccv_upmix_nepa_impl_causal_s0}"

MODEL2_TAG="${MODEL2_TAG:-nepa_impl_encdec_s0_e50}"
MODEL2_GPU="${MODEL2_GPU:-1}"
MODEL2_ARCH="${MODEL2_ARCH:-encdec}"
MODEL2_QA_LAYOUT="${MODEL2_QA_LAYOUT:-split}"
MODEL2_TOPO_K="${MODEL2_TOPO_K:-16}"
MODEL2_TOPO_RAY_COORD="${MODEL2_TOPO_RAY_COORD:-origin}"
MODEL2_TOPO_RAY_BBOX="${MODEL2_TOPO_RAY_BBOX:-0.5}"
MODEL2_SAVE_DIR="${MODEL2_SAVE_DIR:-runs/eccv_upmix_nepa_impl_encdec_s0}"

mkdir -p logs/pretrain logs/analysis results

ts() { date +"%F %T"; }

run_model_chain() {
  local tag="$1"
  local gpu="$2"
  local arch="$3"
  local qa_layout="$4"
  local topo_k="$5"
  local topo_ray_coord="$6"
  local topo_ray_bbox="$7"
  local save_dir="$8"
  local ep_last
  ep_last=$(printf "%03d" "$((EPOCHS - 1))")
  local ckpt="${save_dir}/ckpt_ep${ep_last}.pt"

  echo "[$(ts)] [${tag}] chain start on GPU${gpu}"

  if [ ! -f "${ckpt}" ]; then
    echo "[$(ts)] [${tag}] start pretrain -> ${save_dir}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
      --mix_config "${MIX_CONFIG}" \
      --mix_num_samples "${MIX_NUM_SAMPLES}" --mix_seed "${MIX_SEED}" \
      --objective nepa \
      --qa_tokens "${QA_TOKENS}" \
      --qa_layout "${qa_layout}" \
      --arch "${arch}" \
      --topo_k "${topo_k}" \
      --topo_ray_coord "${topo_ray_coord}" \
      --topo_ray_bbox "${topo_ray_bbox}" \
      --dual_mask_near "${DUAL_MASK_NEAR}" \
      --dual_mask_far "${DUAL_MASK_FAR}" \
      --dual_mask_window "${DUAL_MASK_WINDOW}" \
      --dual_mask_warmup_frac "${DUAL_MASK_WARMUP_FRAC}" \
      --include_pt_grad "${INCLUDE_PT_GRAD}" \
      --pt_grad_mode "${PT_GRAD_MODE}" \
      --pt_grad_eps "${PT_GRAD_EPS}" \
      --pt_grad_clip "${PT_GRAD_CLIP}" \
      --pt_grad_orient "${PT_GRAD_ORIENT}" \
      --include_ray_unc "${INCLUDE_RAY_UNC}" \
      --ray_unc_k "${RAY_UNC_K}" \
      --ray_unc_mode "${RAY_UNC_MODE}" \
      --epochs "${EPOCHS}" --batch "${BATCH}" \
      --n_point "${N_POINT}" --n_ray "${N_RAY}" \
      --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
      --num_workers "${NUM_WORKERS}" \
      --add_eos "${ADD_EOS}" \
      --save_every 1 --save_last 1 \
      --save_dir "${save_dir}" --seed "${SEED}" \
      2>&1 | tee "logs/pretrain/${tag}_bs${BATCH}_e${EPOCHS}.log"
  else
    echo "[$(ts)] [${tag}] skip pretrain (ckpt exists): ${ckpt}"
  fi

  if [ ! -f "${ckpt}" ]; then
    echo "[$(ts)] [${tag}] missing ckpt after pretrain: ${ckpt}"
    return 1
  fi

  echo "[$(ts)] [${tag}] start UCPR/CPAC eval"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --query_backend mesh --gallery_backend udfgrid \
    --n_point "${N_POINT}" --n_ray "${N_RAY}" \
    --eval_seed "${EVAL_SEED}" --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
    --max_files "${MAX_FILES}" --pooling mean_a \
    --tie_break_eps "${TIE_BREAK_EPS}" \
    --out_json "results/ucpr_${tag}_mesh2udf_1k_indep_mean_a.json" \
    2>&1 | tee "logs/analysis/ucpr_${tag}_mesh2udf_1k_indep_mean_a.log"

  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_ucpr \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --query_backend mesh --gallery_backend pointcloud_noray \
    --n_point "${N_POINT}" --n_ray "${N_RAY}" \
    --eval_seed "${EVAL_SEED}" --eval_seed_gallery "${EVAL_SEED_GALLERY}" \
    --max_files "${MAX_FILES}" --pooling mean_a \
    --tie_break_eps "${TIE_BREAK_EPS}" \
    --out_json "results/ucpr_${tag}_mesh2pc_1k_indep_mean_a.json" \
    2>&1 | tee "logs/analysis/ucpr_${tag}_mesh2pc_1k_indep_mean_a.log"

  CUDA_VISIBLE_DEVICES="${gpu}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_POINT}" --n_query "${N_RAY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source pool --baseline nn_copy \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${tag}_pc2udf_800_pool_h_htrain4k_with_nncopy.json" \
    2>&1 | tee "logs/analysis/cpac_${tag}_pc2udf_800_pool_h_htrain4k_with_nncopy.log"

  CUDA_VISIBLE_DEVICES="${gpu}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_POINT}" --n_query "${N_RAY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test none \
    --rep_source h --query_source pool \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${tag}_pc2udf_800_pool_testnone_h_htrain4k.json" \
    2>&1 | tee "logs/analysis/cpac_${tag}_pc2udf_800_pool_testnone_h_htrain4k.log"

  CUDA_VISIBLE_DEVICES="${gpu}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_POINT}" --n_query "${N_RAY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test mismatch --mismatch_shift 1 \
    --rep_source h --query_source pool \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${tag}_pc2udf_800_pool_testmismatch_h_htrain4k.json" \
    2>&1 | tee "logs/analysis/cpac_${tag}_pc2udf_800_pool_testmismatch_h_htrain4k.log"

  CUDA_VISIBLE_DEVICES="${gpu}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context "${N_POINT}" --n_query "${N_RAY}" \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid --grid_sample_mode near_surface \
    --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --baseline nn_copy \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda "${RIDGE_LAMBDA}" --tau "${TAU}" --eval_seed "${EVAL_SEED}" \
    --out_json "results/cpac_${tag}_pc2udf_800_grid_near08_h_htrain4k_with_nncopy.json" \
    2>&1 | tee "logs/analysis/cpac_${tag}_pc2udf_800_grid_near08_h_htrain4k_with_nncopy.log"

  CUDA_VISIBLE_DEVICES="${gpu}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.qualitative_cpac_marching_cubes \
    --cache_root "${CACHE_ROOT}" --split "${SPLIT}" \
    --ckpt "${ckpt}" \
    --context_backend pointcloud_noray \
    --head_train_split "${HEAD_TRAIN_SPLIT}" --head_train_backend "${HEAD_TRAIN_BACKEND}" \
    --head_train_max_shapes 1000 \
    --n_context "${N_POINT}" --n_query_probe "${N_RAY}" \
    --grid_res 32 --mc_level 0.03 \
    --max_shapes 8 --shape_offset 0 \
    --mesh_metrics 1 --mesh_samples 20000 --fscore_taus 0.005,0.01,0.02 \
    --save_volumes 0 --save_png 0 \
    --out_dir "results/qual_mc_${tag}" \
    2>&1 | tee "logs/analysis/qual_mc_${tag}.log"

  echo "[$(ts)] [${tag}] chain finished"
}

run_model_chain "${MODEL1_TAG}" "${MODEL1_GPU}" "${MODEL1_ARCH}" "${MODEL1_QA_LAYOUT}" "${MODEL1_TOPO_K}" "${MODEL1_TOPO_RAY_COORD}" "${MODEL1_TOPO_RAY_BBOX}" "${MODEL1_SAVE_DIR}" &
pid0=$!
run_model_chain "${MODEL2_TAG}" "${MODEL2_GPU}" "${MODEL2_ARCH}" "${MODEL2_QA_LAYOUT}" "${MODEL2_TOPO_K}" "${MODEL2_TOPO_RAY_COORD}" "${MODEL2_TOPO_RAY_BBOX}" "${MODEL2_SAVE_DIR}" &
pid1=$!

wait "${pid0}" "${pid1}"
echo "[$(ts)] all model chains finished"
