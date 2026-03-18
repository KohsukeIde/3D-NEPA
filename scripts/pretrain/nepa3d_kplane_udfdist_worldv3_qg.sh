#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N kp_udfd

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
RUN_TAG="${RUN_TAG:-kplane_udfdist_$(date +%Y%m%d_%H%M%S)}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
SAVE_DIR="${SAVE_DIR:-${WORKDIR}/runs/kplane_udfdist}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/kplane_udfdist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
PLANE_TYPE="${PLANE_TYPE:-kplane}"
FUSION="${FUSION:-auto}"
PLANE_RESOLUTIONS="${PLANE_RESOLUTIONS:-64}"
PLANE_CHANNELS="${PLANE_CHANNELS:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
mkdir -p "${LOG_ROOT}" "${SAVE_DIR}"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"
cd "${WORKDIR}"
[[ -f "${VENV_ACTIVATE}" ]] && source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE}" 2>/dev/null || true
python -u -m nepa3d.tracks.kplane.train.pretrain_udfdist_worldv3 \
  --mix_config_path "${MIX_CONFIG}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_TAG}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --plane_type "${PLANE_TYPE}" \
  --fusion "${FUSION}" \
  --plane_resolutions "${PLANE_RESOLUTIONS}" \
  --plane_channels "${PLANE_CHANNELS}" \
  --hidden_dim "${HIDDEN_DIM}" \
  2>&1 | tee "${LOG_PATH}"
