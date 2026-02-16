#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
GPU="${GPU:-0}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix.yaml}"
BASE_CKPT="${BASE_CKPT:-runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt}"
B2C2_CKPT="${B2C2_CKPT:-runs/eccv_upmix_nepa_qa_dualmask_b2c2quick_s0/ckpt_ep050.pt}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"

mkdir -p logs/pretrain logs/analysis results

run_pretrain_if_missing() {
  local out_ckpt="$1"
  shift
  if [ -f "${out_ckpt}" ]; then
    echo "[skip] ckpt exists: ${out_ckpt}"
    return 0
  fi
  echo "[run] pretrain -> ${out_ckpt}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m nepa3d.train.pretrain "$@"
}

run_cpac_if_missing() {
  local out_json="$1"
  shift
  if [ -f "${out_json}" ]; then
    echo "[skip] json exists: ${out_json}"
    return 0
  fi
  echo "[run] cpac -> ${out_json}"
  TQDM_DISABLE=1 CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf "$@" --out_json "${out_json}"
}

# ------------------------------------------------------------------
# D-only quick (single-factor)
# ------------------------------------------------------------------
run_pretrain_if_missing \
  runs/eccv_upmix_nepa_qa_dualmask_dquick_s0/ckpt_ep050.pt \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume "${BASE_CKPT}" \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_dquick_s0 --seed 0 \
  --d_hard_weight 0.1 --d_hard_top_frac 0.25 --d_hard_min_tokens 32 \
  | tee logs/pretrain/eccv_completion_ae6_dquick_s0.log

run_cpac_if_missing \
  results/cpac_tmp_d_ep050_pc2udf_800_normal_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_dquick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_d_ep050_pc2udf_800_normal_h_htrain4k.log

run_cpac_if_missing \
  results/cpac_tmp_d_ep050_pc2udf_800_grid_near08_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_dquick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_d_ep050_pc2udf_800_grid_near08_h_htrain4k.log

# ------------------------------------------------------------------
# E-only quick (single-factor)
# ------------------------------------------------------------------
run_pretrain_if_missing \
  runs/eccv_upmix_nepa_qa_dualmask_equick_s0/ckpt_ep050.pt \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 51 --batch 96 --n_point 256 --n_ray 256 --num_workers 6 \
  --save_every 1 --save_last 1 \
  --resume "${BASE_CKPT}" \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_equick_s0 --seed 0 \
  --aux_e_weight 0.1 \
  | tee logs/pretrain/eccv_completion_ae6_equick_s0.log

run_cpac_if_missing \
  results/cpac_tmp_e_ep050_pc2udf_800_normal_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_equick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_e_ep050_pc2udf_800_normal_h_htrain4k.log

run_cpac_if_missing \
  results/cpac_tmp_e_ep050_pc2udf_800_grid_near08_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_equick_s0/ckpt_ep050.pt \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 256 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_e_ep050_pc2udf_800_grid_near08_h_htrain4k.log

# ------------------------------------------------------------------
# 6 quick scale pilot on top of B-2+C-2
# ------------------------------------------------------------------
run_pretrain_if_missing \
  runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples 8000 --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
  --epochs 52 --batch 64 --n_point 256 --n_ray 256 \
  --n_point_schedule '0:256,51:512' --n_ray_schedule '0:256' --max_len -1 \
  --num_workers 6 --save_every 1 --save_last 1 \
  --resume "${B2C2_CKPT}" \
  --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0 --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4 \
  | tee logs/pretrain/eccv_completion_ae6_b2c2_scalequick_s0.log

run_cpac_if_missing \
  results/cpac_tmp_b2c2_ep050_pc512q256_pool_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt "${B2C2_CKPT}" --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_b2c2_ep050_pc512q256_pool_h_htrain4k.log

run_cpac_if_missing \
  results/cpac_tmp_b2c2_ep050_pc512q256_grid_near08_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt "${B2C2_CKPT}" --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_b2c2_ep050_pc512q256_grid_near08_h_htrain4k.log

run_cpac_if_missing \
  results/cpac_tmp_b2c2_scale_ep051_pc512q256_pool_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source pool \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_b2c2_scale_ep051_pc512q256_pool_h_htrain4k.log

run_cpac_if_missing \
  results/cpac_tmp_b2c2_scale_ep051_pc512q256_grid_near08_h_htrain4k.json \
  --cache_root "${CACHE_ROOT}" --split eval \
  --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt --max_len 1538 \
  --context_backend pointcloud_noray \
  --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
  --n_context 512 --n_query 256 --disjoint_context_query 1 \
  --context_mode_train normal --context_mode_test normal \
  --rep_source h --query_source grid \
  --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
  --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
  | tee logs/analysis/cpac_tmp_b2c2_scale_ep051_pc512q256_grid_near08_h_htrain4k.log

echo "[done] D/E/6 quick chain complete"
