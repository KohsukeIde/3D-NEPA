#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=12:00:00
#PBS -P gag51403
#PBS -N nepa3d_ft_scanobj
#PBS -o nepa3d_ft_scanobj.out
#PBS -e nepa3d_ft_scanobj.err

set -eu

. /etc/profile.d/modules.sh
module load cuda/12.6

cd /groups/gag51403/ide/3D-NEPA
if [ -f ".venv/bin/activate" ]; then
  . .venv/bin/activate
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
BACKEND="${BACKEND:-pointcloud_noray}"
# Use n_types=5 ckpt by default (compatible with pointcloud_noray / TYPE_MISSING_RAY).
CKPT="${CKPT:-runs/querynepa3d_ab_meshpre_p0/ckpt_ep049.pt}"
BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SEED="${EVAL_SEED:-0}"
MC_EVAL_K="${MC_EVAL_K:-4}"
SEED="${SEED:-0}"
FEWSHOT_K="${FEWSHOT_K:-0}"
FEWSHOT_SEED="${FEWSHOT_SEED:-0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"
ALLOW_SCAN_MAIN_SPLIT_V2="${ALLOW_SCAN_MAIN_SPLIT_V2:-0}"
ADD_EOS="${ADD_EOS:--1}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"
PT_FPS_KEY="${PT_FPS_KEY:-pt_fps_order}"
PT_RFPS_M="${PT_RFPS_M:-4096}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
AUG_PRESET="${AUG_PRESET:-scanobjectnn}"
AUG_EVAL="${AUG_EVAL:-0}"
LLRD="${LLRD:-1.0}"
DROP_PATH="${DROP_PATH:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"

AUG_EVAL_FLAG=""
if [ "${AUG_EVAL}" = "1" ]; then
  AUG_EVAL_FLAG="--aug_eval"
fi

"${PYTHON_BIN}" -m nepa3d.train.finetune_cls \
  --cache_root "${CACHE_ROOT}" \
  --backend "${BACKEND}" \
  --ckpt "${CKPT}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --n_point "${N_POINT}" \
  --n_ray "${N_RAY}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --add_eos "${ADD_EOS}" \
  --eval_seed "${EVAL_SEED}" \
  --mc_eval_k "${MC_EVAL_K}" \
  --val_ratio "${VAL_RATIO}" \
  --val_seed "${VAL_SEED}" \
  --val_split_mode "${VAL_SPLIT_MODE}" \
  --allow_scan_uniscale_v2 "${ALLOW_SCAN_UNISCALE_V2}" \
  --allow_scan_main_split_v2 "${ALLOW_SCAN_MAIN_SPLIT_V2}" \
  --fewshot_k "${FEWSHOT_K}" \
  --fewshot_seed "${FEWSHOT_SEED}" \
  --pt_sample_mode_train "${PT_SAMPLE_MODE_TRAIN}" \
  --pt_sample_mode_eval "${PT_SAMPLE_MODE_EVAL}" \
  --pt_fps_key "${PT_FPS_KEY}" \
  --pt_rfps_m "${PT_RFPS_M}" \
  --point_order_mode "${POINT_ORDER_MODE}" \
  --aug_preset "${AUG_PRESET}" \
  ${AUG_EVAL_FLAG} \
  --llrd "${LLRD}" \
  --drop_path "${DROP_PATH}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --max_grad_norm "${MAX_GRAD_NORM}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --warmup_start_factor "${WARMUP_START_FACTOR}" \
  --min_lr "${MIN_LR}"
