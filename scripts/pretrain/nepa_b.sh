#!/usr/bin/env bash
set -euo pipefail

# ========================
export NCCL_DEBUG=INFO
export NCCL_IB_TC=106
export NCCL_IB_GID_INDEX=3
export NCCL_CROSS_NIC=0
export TORCH_DISTRIBUTED_TIMEOUT=1800

: "${WORLD_SIZE:=1}"
: "${RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${PREFERRED_IFNAME:=}"

# Pick a valid socket interface on the compute node.
# If NCCL_SOCKET_IFNAME is already set by caller and exists, keep it.
if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && [[ -d "/sys/class/net/${NCCL_SOCKET_IFNAME}" ]]; then
  :
else
  ifname=""
  if [[ -n "${PREFERRED_IFNAME}" ]] && [[ -d "/sys/class/net/${PREFERRED_IFNAME}" ]]; then
    ifname="${PREFERRED_IFNAME}"
  else
    ifname="$(ls /sys/class/net 2>/dev/null | grep -E '^(ibp|ib|en|eth)' | head -n 1 || true)"
  fi
  if [[ -n "${ifname}" ]]; then
    export NCCL_SOCKET_IFNAME="${ifname}"
    export GLOO_SOCKET_IFNAME="${ifname}"
  else
    unset NCCL_SOCKET_IFNAME || true
    unset GLOO_SOCKET_IFNAME || true
  fi
fi
echo "[nccl] socket_ifname=${NCCL_SOCKET_IFNAME:-auto}"
echo "[nccl] gloo_ifname=${GLOO_SOCKET_IFNAME:-auto}"

# ========================
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

: "${EXPERIMENT_NAME:=nepa-base-patch14-224}"
: "${WANDB_PROJECT:=Nepa-Pretrain}"

: "${CONFIG_NAME:=configs/pretrain/nepa-base-patch14-224}"
: "${DATASET_PATH:=data/imagenet-1k-hf}"
: "${TRAIN_DIR:=}"
: "${VALIDATION_DIR:=}"
: "${OUTPUT_DIR:=outputs/${EXPERIMENT_NAME}}"
: "${LOAD_FROM_DISK:=True}"
: "${IMAGE_COLUMN_NAME:=image}"

: "${TOTAL_BATCH_SIZE:=4096}"
: "${PER_DEVICE_BATCH_SIZE:=256}"
GRAD_ACCUM_STEPS=$(( TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU * WORLD_SIZE) ))
: "${NUM_EPOCHS:=1600}"

: "${BASE_LEARNING_RATE:=3e-4}"
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=$((4 * NGPU))
: "${DIAG_COPY:=1}"
: "${DIAG_EVERY:=50}"
: "${DIAG_K:=1}"

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# dataset source: HF dataset on disk vs imagefolder directories.
DATASET_ARGS=""
if [[ -n "${TRAIN_DIR}" || -n "${VALIDATION_DIR}" ]]; then
    if [[ -z "${TRAIN_DIR}" || -z "${VALIDATION_DIR}" ]]; then
        echo "ERROR: TRAIN_DIR and VALIDATION_DIR must be set together."
        exit 2
    fi
    DATASET_ARGS="--train_dir ${TRAIN_DIR} --validation_dir ${VALIDATION_DIR}"
else
    DATASET_ARGS="--dataset_name ${DATASET_PATH} --load_from_disk ${LOAD_FROM_DISK}"
fi

# ========================
torchrun \
    --nnodes=$WORLD_SIZE  \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node $NGPU run_nepa.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --config_name $CONFIG_NAME \
    --image_processor_name  $CONFIG_NAME \
    $DATASET_ARGS \
    --image_column_name $IMAGE_COLUMN_NAME \
    --dataloader_drop_last True \
    \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.025 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 50000 \
    \
    --seed 1337 \
    --bf16 True \
    \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \
    \
    --diag_copy "${DIAG_COPY}" \
    --diag_every "${DIAG_EVERY}" \
    --diag_k "${DIAG_K}" \
    \
    --report_to wandb \
    --run_name $EXPERIMENT_NAME
