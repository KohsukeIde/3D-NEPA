#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

BBOX_PID="${BBOX_PID:?BBOX_PID is required}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

echo "[$(date +"%F %T")] wait-start bbox_pid=${BBOX_PID}"
while kill -0 "${BBOX_PID}" 2>/dev/null; do
  sleep 120
done
echo "[$(date +"%F %T")] bbox chain finished; launching scale_longfix"

CKPT_OUT="runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_longfix_s0/ckpt_ep054.pt"
if [ ! -f "${CKPT_OUT}" ]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
    --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
    --mix_num_samples 12000 --mix_seed 0 \
    --objective nepa --qa_tokens 1 \
    --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window 32 --dual_mask_warmup_frac 0.05 \
    --epochs 55 --batch 16 --lr 3e-4 \
    --n_point 256 --n_ray 256 \
    --n_point_schedule "0:256,51:512,53:1024" --n_ray_schedule "0:256" \
    --max_len 2562 \
    --num_workers 6 --save_every 1 --save_last 1 \
    --resume runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt \
    --resume_optimizer 1 --resume_optimizer_partial 1 \
    --save_dir runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_longfix_s0 --seed 0 \
    --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
    --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
    --cycle_weight 0.1 --cycle_answer_drop_prob 0.4 \
    2>&1 | tee logs/pretrain/eccv_upmix_nepa_qa_dualmask_b2c2_scale_longfix_s0_bs16_e55.log
else
  echo "[$(date +"%F %T")] skip pretrain; checkpoint exists: ${CKPT_OUT}"
fi

for NC in 512 1024; do
  # pool
  CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_longfix_s0/ckpt_ep054.pt --max_len 2562 \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
    --n_context "${NC}" --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source pool \
    --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --out_json "results/cpac_tmp_b2c2_scale_longfix_ep054_pc${NC}q256_pool_h_htrain4k.json" \
    2>&1 | tee "logs/analysis/cpac_tmp_b2c2_scale_longfix_ep054_pc${NC}q256_pool_h_htrain4k.log"

  # grid near08
  CUDA_VISIBLE_DEVICES=0 TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt runs/eccv_upmix_nepa_qa_dualmask_b2c2_scale_longfix_s0/ckpt_ep054.pt --max_len 2562 \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
    --n_context "${NC}" --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid \
    --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --out_json "results/cpac_tmp_b2c2_scale_longfix_ep054_pc${NC}q256_grid_near08_h_htrain4k.json" \
    2>&1 | tee "logs/analysis/cpac_tmp_b2c2_scale_longfix_ep054_pc${NC}q256_grid_near08_h_htrain4k.log"
done

echo "[$(date +"%F %T")] scale_longfix pipeline finished"

