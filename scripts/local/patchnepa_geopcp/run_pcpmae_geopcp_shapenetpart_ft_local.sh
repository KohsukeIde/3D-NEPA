#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

CKPT="${CKPT:?set CKPT to a PCP-MAE / Geo-PCP pretrain checkpoint}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
ROOT_DATA="${ROOT_DATA:-${ROOT_DIR}/data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
LOG_DIR="${LOG_DIR:-geopcp_shapenetpart_ft}"

export USE_WANDB="${GEOPCP_USE_WANDB:-1}"
export WANDB_PROJECT="${GEOPCP_PARTSEG_WANDB_PROJECT:-patchnepa-geopcp-shapenetpart}"
export WANDB_ENTITY="${GEOPCP_WANDB_ENTITY:-}"
export WANDB_GROUP="${GEOPCP_WANDB_GROUP:-routea_geopcp}"
export WANDB_RUN_NAME="${GEOPCP_WANDB_RUN_NAME:-${LOG_DIR}}"
export WANDB_MODE="${GEOPCP_WANDB_MODE:-online}"
export WANDB_JOB_TYPE="${GEOPCP_WANDB_JOB_TYPE:-shapenetpart_finetune}"
export WANDB_DIR="${GEOPCP_WANDB_DIR:-${PCP_ROOT}/wandb}"

geopcp_require_python
geopcp_require_gpu
geopcp_require_compiled_backends

cd "${PCP_ROOT}/segmentation"
"${PYTHON_BIN}" main.py \
  --gpu "${GPU}" \
  --ckpts "${CKPT}" \
  --log_dir "${LOG_DIR}" \
  --learning_rate "${LEARNING_RATE:-0.0002}" \
  --epoch "${EPOCHS:-300}" \
  --batch_size "${BATCH_SIZE:-16}" \
  --root "${ROOT_DATA}" \
  --seed "${SEED}"
