#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1:ncpus=16:ngpus=1
#PBS -l walltime=24:00:00
#PBS -P gag51403
#PBS -N nepa3d_kplane_pretrain
#PBS -o nepa3d_kplane_pretrain.out
#PBS -e nepa3d_kplane_pretrain.err

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
MIX_CONFIG="${MIX_CONFIG:?set MIX_CONFIG=...}"
MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-0}"
MIX_SEED="${MIX_SEED:-0}"
BATCH="${BATCH:-96}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
# 2D NEPA style linear LR scaling:
#   LEARNING_RATE = BASE_LEARNING_RATE * TOTAL_BATCH_SIZE / 256
# Backward-compatible default keeps LR at current default when TOTAL_BATCH_SIZE=96.
LR_SCALE_ENABLE="${LR_SCALE_ENABLE:-1}"          # 1: enable linear scaling, 0: disable
LR_SCALE_REF_BATCH="${LR_SCALE_REF_BATCH:-256}"  # denominator in scaling rule
LR_BASE_TOTAL_BATCH="${LR_BASE_TOTAL_BATCH:-96}" # baseline total batch for deriving BASE_LEARNING_RATE
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-}"     # if empty, derived from LR and LR_BASE_TOTAL_BATCH
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-6}"
N_CONTEXT="${N_CONTEXT:-256}"
N_QUERY="${N_QUERY:-256}"
QUERY_SOURCE="${QUERY_SOURCE:-pool}"
TARGET_MODE="${TARGET_MODE:-backend}"
DISJOINT_CONTEXT_QUERY="${DISJOINT_CONTEXT_QUERY:-1}"
PLANE_TYPE="${PLANE_TYPE:-kplane}"            # triplane / kplane
FUSION="${FUSION:-auto}"                       # auto / sum / product / rg_product
PRODUCT_RANK_GROUPS="${PRODUCT_RANK_GROUPS:-0}" # only for rg_product
PRODUCT_GROUP_REDUCE="${PRODUCT_GROUP_REDUCE:-sum}" # sum / mean (only for rg_product)
PLANE_RESOLUTIONS="${PLANE_RESOLUTIONS:-64}"   # e.g. 64 or 32,64,128
PLANE_CHANNELS="${PLANE_CHANNELS:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
VOXEL_GRID="${VOXEL_GRID:-64}"
VOXEL_DILATE="${VOXEL_DILATE:-1}"
VOXEL_MAX_STEPS="${VOXEL_MAX_STEPS:-0}"
SAVE_DIR="${SAVE_DIR:-runs/eccv_kplane_baseline}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME="${RESUME:-}"
SEED="${SEED:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}" # auto/no/fp16/bf16
NUM_PROCESSES="${NUM_PROCESSES:-1}"       # >1 enables Accelerate DDP
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
ACCELERATE_PYTHON="${ACCELERATE_PYTHON:-${PYTHON_BIN}}"
ACCELERATE_LAUNCH_MODULE="${ACCELERATE_LAUNCH_MODULE:-accelerate.commands.launch}"

TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((BATCH * NUM_PROCESSES))}"
if [ "${LR_SCALE_ENABLE}" = "1" ]; then
  if [ -z "${BASE_LEARNING_RATE}" ]; then
    BASE_LEARNING_RATE="$("${PYTHON_BIN}" -c "print(float('${LR}') * float('${LR_BASE_TOTAL_BATCH}') / 256.0)")"
  fi
  LR="$("${PYTHON_BIN}" -c "print(float('${BASE_LEARNING_RATE}') * float('${TOTAL_BATCH_SIZE}') / float('${LR_SCALE_REF_BATCH}'))")"
  echo "[lr-scale] enabled: base_lr=${BASE_LEARNING_RATE} total_batch=${TOTAL_BATCH_SIZE} ref_batch=${LR_SCALE_REF_BATCH} lr=${LR}"
else
  echo "[lr-scale] disabled: lr=${LR}"
fi

if [ "${NUM_PROCESSES}" -gt 1 ]; then
  "${ACCELERATE_PYTHON}" -m "${ACCELERATE_LAUNCH_MODULE}" \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${MIXED_PRECISION}" \
    -m nepa3d.train.pretrain_kplane \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples "${MIX_NUM_SAMPLES}" \
  --mix_seed "${MIX_SEED}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --n_context "${N_CONTEXT}" \
  --n_query "${N_QUERY}" \
  --query_source "${QUERY_SOURCE}" \
  --target_mode "${TARGET_MODE}" \
  --disjoint_context_query "${DISJOINT_CONTEXT_QUERY}" \
  --plane_type "${PLANE_TYPE}" \
  --fusion "${FUSION}" \
  --product_rank_groups "${PRODUCT_RANK_GROUPS}" \
  --product_group_reduce "${PRODUCT_GROUP_REDUCE}" \
  --plane_resolutions "${PLANE_RESOLUTIONS}" \
  --plane_channels "${PLANE_CHANNELS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --save_dir "${SAVE_DIR}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --seed "${SEED}"
else
  "${PYTHON_BIN}" -m nepa3d.train.pretrain_kplane \
  --mix_config "${MIX_CONFIG}" \
  --mix_num_samples "${MIX_NUM_SAMPLES}" \
  --mix_seed "${MIX_SEED}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --n_context "${N_CONTEXT}" \
  --n_query "${N_QUERY}" \
  --query_source "${QUERY_SOURCE}" \
  --target_mode "${TARGET_MODE}" \
  --disjoint_context_query "${DISJOINT_CONTEXT_QUERY}" \
  --plane_type "${PLANE_TYPE}" \
  --fusion "${FUSION}" \
  --product_rank_groups "${PRODUCT_RANK_GROUPS}" \
  --product_group_reduce "${PRODUCT_GROUP_REDUCE}" \
  --plane_resolutions "${PLANE_RESOLUTIONS}" \
  --plane_channels "${PLANE_CHANNELS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --voxel_grid "${VOXEL_GRID}" \
  --voxel_dilate "${VOXEL_DILATE}" \
  --voxel_max_steps "${VOXEL_MAX_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --save_dir "${SAVE_DIR}" \
  --save_every "${SAVE_EVERY}" \
  --save_last "${SAVE_LAST}" \
  --auto_resume "${AUTO_RESUME}" \
  --resume "${RESUME}" \
  --seed "${SEED}"
fi
