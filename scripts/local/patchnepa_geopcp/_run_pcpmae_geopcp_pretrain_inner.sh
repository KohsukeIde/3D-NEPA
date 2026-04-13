#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/pretrain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/geopcp_pretrain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/geopcp_pretrain.pid}"

mkdir -p "${LOG_ROOT}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

echo "$$" > "${PID_FILE}"

geopcp_require_python
geopcp_require_gpu
geopcp_require_compiled_backends

{
  echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S %Z') start geopcp pretrain"
  echo "[launcher] env=${GEOPCP_ENV_NAME:-geopcp-pcpmae-cu118}"
  echo "[launcher] config=${CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml}"
  echo "[launcher] exp_name=${EXP_NAME:-geopcp_worldvis_base_normal_100ep}"
  echo "[launcher] save_dir=${SAVE_DIR:-${PCP_ROOT}/experiments}"
  echo "[launcher] gpus=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nproc=${NPROC_PER_NODE:-4} master_port=${MASTER_PORT:-29741}"
} >> "${LOG_FILE}"

cd "${PCP_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --master_port="${MASTER_PORT:-29741}" \
  main_geopcp.py \
  --launcher pytorch \
  --config "${CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml}" \
  --exp_name "${EXP_NAME:-geopcp_worldvis_base_normal_100ep}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --seed "${SEED:-0}" \
  >> "${LOG_FILE}" 2>&1
