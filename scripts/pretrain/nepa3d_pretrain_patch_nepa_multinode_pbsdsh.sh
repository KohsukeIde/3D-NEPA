#!/bin/bash
#
# Multi-node launcher for patch-NEPA pretrain (Accelerate DDP via per-node pbsdsh).
#
# Example (8 GPUs total = 2 nodes x 4 gpus/node):
#   qsub -l rt_QF=2 -l walltime=24:00:00 \
#     -v RUN_TAG=patchnepa_ptonly_ddp8,SAVE_DIR=runs/patchnepa_pointonly/patchnepa_ptonly_ddp8 \
#     scripts/pretrain/nepa3d_pretrain_patch_nepa_multinode_pbsdsh.sh
#
#PBS -l rt_QF=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N patchnepa_ddp
#PBS -o patchnepa_ddp.out
#PBS -e patchnepa_ddp.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29410}"
PREFERRED_IFNAME="${PREFERRED_IFNAME:-}"
RUN_TAG="${RUN_TAG:-patchnepa_ptonly_ddp}"
DDP_LOG_ROOT="${DDP_LOG_ROOT:-${WORKDIR}/logs/ddp_patch_nepa_pretrain}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_STABLE_MODE="${NCCL_STABLE_MODE:-0}"
export ENABLE_ECC_PREFLIGHT="${ENABLE_ECC_PREFLIGHT:-1}"

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CUDA_MODULE}" 2>/dev/null || true

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
  echo "ERROR: PBS_NODEFILE is missing."
  exit 1
fi

RUN_DIR="${DDP_LOG_ROOT}/ddp_patchnepa_${PBS_JOBID:-manual}_${RUN_TAG}"
mkdir -p "${RUN_DIR}/logs"
PRETRAIN_DONE_MARKER="${RUN_DIR}/pretrain_done.marker"
LAUNCH_SCRIPT_SRC="${WORKDIR}/scripts/pretrain/nepa3d_pretrain_patch_nepa_qf.sh"
LAUNCH_SCRIPT_RUN="${RUN_DIR}/nepa3d_pretrain_patch_nepa_qf.snapshot.sh"

if [[ ! -f "${LAUNCH_SCRIPT_SRC}" ]]; then
  echo "ERROR: launch script missing: ${LAUNCH_SCRIPT_SRC}"
  exit 1
fi
cp -f "${LAUNCH_SCRIPT_SRC}" "${LAUNCH_SCRIPT_RUN}"
chmod +x "${LAUNCH_SCRIPT_RUN}"

HOSTS_FILE="${RUN_DIR}/hosts.txt"
awk '!seen[$0]++{print}' "${PBS_NODEFILE}" > "${HOSTS_FILE}"
NNODES="$(wc -l < "${HOSTS_FILE}" | tr -d ' ')"
MASTER_HOST="$(head -n 1 "${HOSTS_FILE}")"
MASTER_ADDR="$(getent ahostsv4 "${MASTER_HOST}" 2>/dev/null | awk '{print $1; exit}' || true)"
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "ERROR: cannot resolve MASTER_HOST=${MASTER_HOST} to IPv4"
  exit 1
fi

TOTAL_PROCESSES="$((NNODES * NPROC_PER_NODE))"
NUM_PROCESSES="${NUM_PROCESSES:-${TOTAL_PROCESSES}}"
NUM_MACHINES="${NUM_MACHINES:-${NNODES}}"

echo "=== PATCH-NEPA DDP TOPOLOGY ==="
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "MASTER_HOST=${MASTER_HOST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_DIR=${RUN_DIR}"
echo "LAUNCH_SCRIPT_RUN=${LAUNCH_SCRIPT_RUN}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${LAUNCH_SCRIPT_RUN}" | sed 's/^/launch_script_sha256=/'
fi
nl -ba "${HOSTS_FILE}"
echo

ENVCONF="${RUN_DIR}/env.conf"
cat > "${ENVCONF}" <<EOF
WORKDIR="${WORKDIR}"
RUN_DIR="${RUN_DIR}"
PRETRAIN_DONE_MARKER="${PRETRAIN_DONE_MARKER}"
HOSTS_FILE="${HOSTS_FILE}"
MASTER_ADDR="${MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT}"
NPROC_PER_NODE="${NPROC_PER_NODE}"
NUM_PROCESSES="${NUM_PROCESSES}"
NUM_MACHINES="${NUM_MACHINES}"
CUDA_MODULE="${CUDA_MODULE}"
VENV_ACTIVATE="${VENV_ACTIVATE}"
PREFERRED_IFNAME="${PREFERRED_IFNAME}"
NCCL_STABLE_MODE="${NCCL_STABLE_MODE}"
ENABLE_ECC_PREFLIGHT="${ENABLE_ECC_PREFLIGHT}"
LAUNCH_SCRIPT_RUN="${LAUNCH_SCRIPT_RUN}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-}"
TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-}"
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-}"
EOF

# Propagate pretrain recipe/runtime knobs to node-local launch scripts.
# pbsdsh does not reliably forward arbitrary shell variables, so persist them here.
PROPAGATE_VARS=(
  RUN_TAG SAVE_DIR LOG_ROOT MIX_CONFIG
  N_POINT N_RAY USE_RAY_PATCH STAGE2_REQUIRE_RAY STAGE2_REQUIRE_GLOBAL_BATCH128
  INCLUDE_RAY_UNC RAY_ASSIGN_MODE RAY_USE_ORIGIN RAY_PROXY_RADIUS_SCALE
  RAY_NUM_GROUPS RAY_GROUP_SIZE RAY_POOL_MODE RAY_FUSE RAY_MISS_T RAY_HIT_THRESHOLD
  BATCH EPOCHS NUM_WORKERS LR SEED
  USE_EMA EMA_DECAY DIAG_COPY DIAG_EVERY DIAG_K
  LOSS_TARGET_MODE Q_MASK_PROB Q_MASK_WARMUP_EPOCHS Q_MASK_MODE
  USE_WANDB WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_GROUP WANDB_TAGS
  WANDB_MODE WANDB_LOG_EVERY WANDB_API_KEY WANDB_BASE_URL
  LR_SCHEDULER WARMUP_RATIO WARMUP_EPOCHS MIN_LR WEIGHT_DECAY
  AUTO_RESUME RESUME_OPTIMIZER RESUME MAX_GRAD_NORM GRAD_ACCUM
  PATCH_EMBED PATCH_LOCAL_ENCODER PATCH_FPS_RANDOM_START GROUP_SIZE NUM_GROUPS SERIAL_ORDER SERIAL_BITS SERIAL_SHUFFLE_WITHIN_PATCH
  PATCH_ORDER_MODE PATCH_ORDER_SCHEDULE
  D_MODEL N_LAYERS N_HEADS MLP_RATIO BACKBONE_MODE DROP_PATH_RATE
  QK_NORM QK_NORM_AFFINE QK_NORM_BIAS LAYERSCALE_VALUE ROPE_THETA
  USE_GATED_MLP HIDDEN_ACT
  PT_XYZ_KEY PT_DIST_KEY ABLATE_POINT_DIST PT_SAMPLE_MODE ALLOW_RFPS_CACHED PT_FPS_KEY PT_RFPS_KEY PT_RFPS_M
  POINT_ORDER_MODE QA_TOKENS QA_LAYOUT QA_SEP_TOKEN QA_FUSE ENCDEC_ARCH
  USE_PT_DIST USE_PT_GRAD ANSWER_MLP_LAYERS ANSWER_POOL
  NEPA_SKIP_K NEPA_MULTI_K SKIPK_DISABLE_DUAL_MASK NEPA2D_POS
  TYPE_SPECIFIC_POS TYPE_POS_MAX_LEN MAX_LEN
  DUAL_MASK_NEAR DUAL_MASK_FAR DUAL_MASK_WINDOW DUAL_MASK_TYPE_AWARE DUAL_MASK_WARMUP_FRAC
  AUG_ROTATE_Z AUG_SCALE_MIN AUG_SCALE_MAX AUG_TRANSLATE AUG_JITTER_SIGMA AUG_JITTER_CLIP AUG_RECOMPUTE_DIST
)
for v in "${PROPAGATE_VARS[@]}"; do
  if [[ -v "${v}" ]]; then
    printf '%s=%q\n' "${v}" "${!v}" >> "${ENVCONF}"
  fi
done

NODE_ENTRY="${RUN_DIR}/node_entry.sh"
cat > "${NODE_ENTRY}" <<'SH2'
#!/bin/bash
set -euo pipefail
set -a
source "$(dirname "$0")/env.conf"
set +a
# qsub may pass empty strings; unset to avoid torch.distributed parse errors.
[[ -z "${NCCL_P2P_DISABLE}" ]] && unset NCCL_P2P_DISABLE
[[ -z "${NCCL_NET_GDR_LEVEL}" ]] && unset NCCL_NET_GDR_LEVEL
[[ -z "${TORCH_NCCL_ENABLE_MONITORING}" ]] && unset TORCH_NCCL_ENABLE_MONITORING
[[ -z "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}" ]] && unset TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC


host="$(hostname)"
log="${RUN_DIR}/logs/${host}.patchnepa.log"
exec > "${log}" 2>&1

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

NODE_RANK="$(awk -v h="${host}" '$0==h{print NR-1; exit}' "${HOSTS_FILE}")"
if [[ -z "${NODE_RANK}" ]]; then
  host_s="$(hostname -s)"
  NODE_RANK="$(awk -v h="${host_s}" '$0==h{print NR-1; exit}' "${HOSTS_FILE}")"
fi
if [[ -z "${NODE_RANK}" ]]; then
  echo "ERROR: cannot determine NODE_RANK for ${host}"
  cat "${HOSTS_FILE}"
  exit 41
fi

IFNAME=""
if [[ -n "${PREFERRED_IFNAME:-}" && -d "/sys/class/net/${PREFERRED_IFNAME}" ]]; then
  IFNAME="${PREFERRED_IFNAME}"
else
  IFNAME="$(ls /sys/class/net 2>/dev/null | grep -E '^ibp' | head -n 1 || true)"
fi
if [[ -n "${IFNAME}" && "${IFNAME}" != "lo" ]]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${IFNAME}}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${IFNAME}}"
fi
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

if [[ "${ENABLE_ECC_PREFLIGHT:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  bad_gpu=0

  # 1) Volatile uncorrectable ECC must be zero on all visible GPUs.
  ecc_vals="$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -n "${ecc_vals}" ]]; then
    while IFS= read -r v; do
      v="$(echo "${v}" | xargs)"
      if [[ "${v}" =~ ^[0-9]+$ ]] && [[ "${v}" -gt 0 ]]; then
        bad_gpu=1
        break
      fi
    done <<< "${ecc_vals}"
  fi

  # 2) Pending retired pages indicate unstable device state; fail fast.
  pend_vals="$(nvidia-smi --query-gpu=retired_pages.pending --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${pend_vals}" ]]; then
    while IFS= read -r p; do
      p="$(echo "${p}" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
      if [[ "${p}" == "yes" || "${p}" == "1" ]]; then
        bad_gpu=1
        break
      fi
    done <<< "${pend_vals}"
  fi

  if [[ "${bad_gpu}" == "1" ]]; then
    echo "ERROR: ECC preflight failed on ${host}."
    echo "  ecc.errors.uncorrected.volatile.total="
    echo "${ecc_vals}"
    echo "  retired_pages.pending="
    echo "${pend_vals}"
    nvidia-smi -L || true
    exit 86
  fi
fi

if [[ "${NCCL_STABLE_MODE:-0}" == "1" ]]; then
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1200}"
fi

echo "=== PATCH-NEPA NODE ENTRY ==="
echo "host=${host}"
echo "NODE_RANK=${NODE_RANK}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NCCL_STABLE_MODE=${NCCL_STABLE_MODE:-0}"
echo "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-unset}"
echo "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-unset}"
echo "TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-unset}"
echo "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-unset}"
echo "LAUNCH_SCRIPT_RUN=${LAUNCH_SCRIPT_RUN}"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${LAUNCH_SCRIPT_RUN}" | sed 's/^/launch_script_sha256=/'
fi
echo "python=$(which python)"
python -V || true

exec env \
  WORKDIR="${WORKDIR}" \
  PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  NUM_MACHINES="${NUM_MACHINES}" \
  MACHINE_RANK="${NODE_RANK}" \
  MAIN_PROCESS_IP="${MASTER_ADDR}" \
  MAIN_PROCESS_PORT="${MASTER_PORT}" \
  PRETRAIN_DONE_MARKER="${PRETRAIN_DONE_MARKER}" \
  bash "${LAUNCH_SCRIPT_RUN}"
SH2
chmod +x "${NODE_ENTRY}"

echo "=== LAUNCH via pbsdsh ==="
echo "+ pbsdsh -c ${NNODES} -- bash ${NODE_ENTRY}"
rm -f "${PRETRAIN_DONE_MARKER}"
pbsdsh -c "${NNODES}" -- bash "${NODE_ENTRY}"

# Some pbsdsh failures are not propagated reliably to qsub exit status.
# Require an explicit completion marker from machine_rank=0 script.
if [[ ! -f "${PRETRAIN_DONE_MARKER}" ]]; then
  echo "ERROR: pretrain completion marker missing: ${PRETRAIN_DONE_MARKER}"
  echo "Most likely: one or more node launches failed before clean completion."
  echo "Per-node logs:"
  ls -lh "${RUN_DIR}/logs" | sed -n '1,200p'
  exit 97
fi

echo "=== DONE ==="
echo "Per-node logs:"
ls -lh "${RUN_DIR}/logs" | sed -n '1,200p'
