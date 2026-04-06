#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

CKPT="${CKPT:?set CKPT to a PCP-MAE / Geo-PCP pretrain checkpoint}"
VARIANT="${VARIANT:-obj_bg}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29751}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_WORKERS="${FT_NUM_WORKERS:-8}"

case "${VARIANT}" in
  obj_bg)
    CONFIG="cfgs/itachi/finetune_scan_objbg_itachi.yaml"
    ;;
  obj_only)
    CONFIG="cfgs/itachi/finetune_scan_objonly_itachi.yaml"
    ;;
  hardest|pb_t50_rs)
    CONFIG="cfgs/itachi/finetune_scan_hardest_itachi.yaml"
    VARIANT="hardest"
    ;;
  *)
    geopcp_die "unknown VARIANT=${VARIANT}"
    ;;
esac

if [[ -n "${CONFIG_OVERRIDE:-}" ]]; then
  CONFIG="${CONFIG_OVERRIDE}"
fi

EXP_NAME="${EXP_NAME:-geopcp_scan_${VARIANT}}"

export USE_WANDB="${GEOPCP_USE_WANDB:-1}"
export WANDB_PROJECT="${GEOPCP_SCAN_WANDB_PROJECT:-patchnepa-geopcp-scanobjectnn}"
export WANDB_ENTITY="${GEOPCP_WANDB_ENTITY:-}"
export WANDB_GROUP="${GEOPCP_WANDB_GROUP:-routea_geopcp}"
export WANDB_RUN_NAME="${GEOPCP_WANDB_RUN_NAME:-${EXP_NAME}}"
export WANDB_MODE="${GEOPCP_WANDB_MODE:-online}"
export WANDB_JOB_TYPE="${GEOPCP_WANDB_JOB_TYPE:-scanobjectnn_finetune}"
export WANDB_DIR="${GEOPCP_WANDB_DIR:-${PCP_ROOT}/wandb}"

geopcp_require_python
geopcp_require_gpu
geopcp_require_compiled_backends

cd "${PCP_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  main.py \
  --launcher pytorch \
  --config "${CONFIG}" \
  --num_workers "${NUM_WORKERS}" \
  --finetune_model \
  --ckpts "${CKPT}" \
  --exp_name "${EXP_NAME}" \
  --seed "${SEED}"
