#!/usr/bin/env bash
set -euo pipefail

if [ -n "${WORKDIR:-}" ]; then
  ROOT_DIR="${WORKDIR}"
elif [ -n "${PBS_O_WORKDIR:-}" ]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
ROOT="${ROOT:-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
RUN_NAME="${RUN_NAME:-patchnepa_shapenetpart_ft}"
SAVE_DIR="${SAVE_DIR:-runs/patchpart_itachi}"
CKPT="${CKPT:-}"
CKPT_USE_EMA="${CKPT_USE_EMA:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-geo-teacher-shapenetpart}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WANDB_GROUP="${WANDB_GROUP:-patchnepa-shapenetpart-ft}"
WANDB_TAGS="${WANDB_TAGS:-patchnepa,shapenetpart,partseg}"
WANDB_MODE="${WANDB_MODE:-online}"

EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
BATCH_MODE="${BATCH_MODE:-global}"
LR="${LR:-2e-4}"
WD="${WD:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
GRAD_CLIP="${GRAD_CLIP:-10}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
N_POINT="${N_POINT:-2048}"
USE_NORMALS="${USE_NORMALS:-0}"
DETERMINISTIC_EVAL_SAMPLING="${DETERMINISTIC_EVAL_SAMPLING:-1}"

PATCHNEPA_FT_MODE="${PATCHNEPA_FT_MODE:-q_only}"
PATCHNEPA_FREEZE_PATCH_EMBED="${PATCHNEPA_FREEZE_PATCH_EMBED:-1}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.5}"
LABEL_DIM="${LABEL_DIM:-64}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29588}"

if [ ! -d "${ROOT}" ]; then
  echo "[error] missing ShapeNetPart root: ${ROOT}"
  exit 1
fi

ARGS=(
  -m nepa3d.train.finetune_patch_partseg
  --root "${ROOT}"
  --run_name "${RUN_NAME}"
  --save_dir "${SAVE_DIR}"
  --ckpt "${CKPT}"
  --ckpt_use_ema "${CKPT_USE_EMA}"
  --use_wandb "${USE_WANDB}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_entity "${WANDB_ENTITY}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_group "${WANDB_GROUP}"
  --wandb_tags "${WANDB_TAGS}"
  --wandb_mode "${WANDB_MODE}"
  --epochs "${EPOCHS}"
  --batch "${BATCH}"
  --batch_mode "${BATCH_MODE}"
  --lr "${LR}"
  --weight_decay "${WD}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --grad_clip "${GRAD_CLIP}"
  --seed "${SEED}"
  --num_workers "${NUM_WORKERS}"
  --n_point "${N_POINT}"
  --use_normals "${USE_NORMALS}"
  --deterministic_eval_sampling "${DETERMINISTIC_EVAL_SAMPLING}"
  --patchnepa_ft_mode "${PATCHNEPA_FT_MODE}"
  --patchnepa_freeze_patch_embed "${PATCHNEPA_FREEZE_PATCH_EMBED}"
  --head_dropout "${HEAD_DROPOUT}"
  --label_dim "${LABEL_DIM}"
)

echo "[patchpart] run_name=${RUN_NAME} root=${ROOT} nproc_per_node=${NPROC_PER_NODE} batch=${BATCH} batch_mode=${BATCH_MODE}"
echo "[patchpart] wandb use=${USE_WANDB} project=${WANDB_PROJECT} run=${WANDB_RUN_NAME} group=${WANDB_GROUP} mode=${WANDB_MODE}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${ARGS[@]}"
fi
