#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CUDA_ID="${CUDA_ID:-0}"
RUN_TAG="${RUN_TAG:-eccv_upmix_nepa_qa_dualmask_b2c2_scale_retry_tmp_s0}"
LR="${LR:-2e-4}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-64}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-12000}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-55}"
RESUME_CKPT="${RESUME_CKPT:-runs/eccv_upmix_nepa_qa_dualmask_b2c2_scalequick_s0/ckpt_ep051.pt}"
SAVE_DIR="${SAVE_DIR:-runs/${RUN_TAG}}"
MAX_LEN="${MAX_LEN:-2562}"

PIPE_LOG="logs/analysis/${RUN_TAG}_chain.log"
PRETRAIN_LOG="logs/pretrain/${RUN_TAG}_bs${BATCH}_e${EPOCHS}.log"
TARGET_EP=$((EPOCHS - 1))
TARGET_EP_STR="$(printf '%03d' "${TARGET_EP}")"
CKPT_OUT="${SAVE_DIR}/ckpt_ep${TARGET_EP_STR}.pt"

mkdir -p logs/analysis logs/pretrain "${SAVE_DIR}" results
exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "[$(date +"%F %T")] [${RUN_TAG}] start (gpu=${CUDA_ID}, lr=${LR}, dual_mask_window=${DUAL_MASK_WINDOW})"
echo "[$(date +"%F %T")] [${RUN_TAG}] cfg epochs=${EPOCHS} batch=${BATCH} mix_num_samples=${MIX_NUM_SAMPLES} max_len=${MAX_LEN} target_ep=${TARGET_EP_STR}"
echo "[$(date +"%F %T")] [${RUN_TAG}] resume_ckpt=${RESUME_CKPT}"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
  --mix_config nepa3d/configs/shapenet_unpaired_mix.yaml \
  --mix_num_samples "${MIX_NUM_SAMPLES}" --mix_seed 0 \
  --objective nepa --qa_tokens 1 \
  --dual_mask_near 0.4 --dual_mask_far 0.1 --dual_mask_window "${DUAL_MASK_WINDOW}" --dual_mask_warmup_frac 0.05 \
  --epochs "${EPOCHS}" --batch "${BATCH}" --lr "${LR}" \
  --n_point 256 --n_ray 256 \
  --n_point_schedule "0:256,51:512,53:1024" --n_ray_schedule "0:256" \
  --max_len "${MAX_LEN}" \
  --num_workers 6 --save_every 1 --save_last 1 \
  --resume "${RESUME_CKPT}" \
  --resume_optimizer 1 --resume_optimizer_partial 1 \
  --save_dir "${SAVE_DIR}" --seed 0 \
  --aux_b2_weight 0.1 --aux_b2_hit_weight 1.0 --aux_b2_t_weight 1.0 \
  --aux_b2_rank_weight 1.0 --aux_b2_rank_pairs 128 --aux_b2_rank_margin 0.0 \
  --cycle_weight 0.1 --cycle_answer_drop_prob 0.4 \
  2>&1 | tee "${PRETRAIN_LOG}"

if [ ! -f "${CKPT_OUT}" ]; then
  echo "[$(date +"%F %T")] [${RUN_TAG}] ERROR: missing checkpoint ${CKPT_OUT}"
  exit 1
fi

for NC in 512 1024; do
  # pool normal + nn_copy baseline
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT_OUT}" --max_len "${MAX_LEN}" \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
    --n_context "${NC}" --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source pool --baseline nn_copy \
    --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --out_json "results/cpac_tmp_${RUN_TAG}_pc${NC}q256_pool_h_htrain4k.json"

  # grid near-surface + nn_copy baseline
  CUDA_VISIBLE_DEVICES="${CUDA_ID}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root data/shapenet_unpaired_cache_v1 --split eval \
    --ckpt "${CKPT_OUT}" --max_len "${MAX_LEN}" \
    --context_backend pointcloud_noray \
    --head_train_split train_udf --head_train_backend udfgrid --head_train_max_shapes 4000 \
    --n_context "${NC}" --n_query 256 --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid --grid_sample_mode near_surface --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --baseline nn_copy \
    --max_shapes 800 --ridge_lambda 1e-3 --tau 0.03 --eval_seed 0 \
    --out_json "results/cpac_tmp_${RUN_TAG}_pc${NC}q256_grid_near08_h_htrain4k.json"
done

# UCPR hard pairs guardrail
CUDA_VISIBLE_DEVICES="${CUDA_ID}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt "${CKPT_OUT}" --max_len "${MAX_LEN}" \
  --query_backend mesh --gallery_backend udfgrid \
  --n_point 256 --n_ray 256 \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --pooling mean_a \
  --out_json "results/ucpr_tmp_${RUN_TAG}_mesh2udf_1k_indep_mean_a.json"

CUDA_VISIBLE_DEVICES="${CUDA_ID}" TQDM_DISABLE=1 "${PYTHON_BIN}" -u -m nepa3d.analysis.retrieval_ucpr \
  --cache_root data/shapenet_unpaired_cache_v1 --split eval \
  --ckpt "${CKPT_OUT}" --max_len "${MAX_LEN}" \
  --query_backend mesh --gallery_backend pointcloud_noray \
  --n_point 256 --n_ray 256 \
  --eval_seed 0 --eval_seed_gallery 999 --max_files 1000 \
  --pooling mean_a \
  --out_json "results/ucpr_tmp_${RUN_TAG}_mesh2pc_1k_indep_mean_a.json"

echo "[$(date +"%F %T")] [${RUN_TAG}] finished"
