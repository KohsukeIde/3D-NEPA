#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-S/finetune_scan_objbg.yaml}"
EXP_NAME="${EXP_NAME:-pgpt_s_nomask_ordrand_objbg_e300}"
CKPT_PATH="${CKPT_PATH:-${WORKDIR}/PointGPT/experiments/pretrain_nomask_orderrandom/PointGPT-S/pgpt_s_nomask_ordrand_e300/ckpt-last.pth}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
USE_WANDB="${USE_WANDB:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export CONFIG_PATH
export EXP_NAME
export CKPT_PATH
export NPROC_PER_NODE
export NUM_WORKERS
export USE_WANDB
export CUDA_VISIBLE_DEVICES

exec bash "${SCRIPT_DIR}/pointgpt_finetune_local_ddp.sh"
