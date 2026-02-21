#!/bin/bash
#
# Multi-node launcher for nepa3d pretrain (Accelerate DDP via per-node pbsdsh).
#
# Example:
#   qsub -v RUN_TAG=runA,MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml,N_POINT=1024,N_RAY=1024,QA_TOKENS=1,MAX_LEN=4500,PT_XYZ_KEY=pt_xyz_pool,PT_SAMPLE_MODE_TRAIN=fps,POINT_ORDER_MODE=fps,PT_FPS_KEY=auto \
#     scripts/pretrain/nepa3d_pretrain_multinode_pbsdsh.sh
#
#PBS -l rt_QF=8
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -W group_list=qgah50055
#PBS -N nepa3d_pretrain_ddp
#PBS -o nepa3d_pretrain_ddp.out
#PBS -e nepa3d_pretrain_ddp.err

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29400}"
PREFERRED_IFNAME="${PREFERRED_IFNAME:-}"
RUN_TAG="${RUN_TAG:-pretrain}"
DDP_LOG_ROOT="${DDP_LOG_ROOT:-${WORKDIR}/logs/ddp_pretrain}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CUDA_MODULE}" 2>/dev/null || true

echo "=== JOB INFO ==="
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "PBS_NODEFILE=${PBS_NODEFILE:-}"
echo "WORKDIR=${WORKDIR}"
echo "python=$(which python)"
python -V || true
echo

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
  echo "ERROR: PBS_NODEFILE is missing."
  exit 1
fi

RUN_DIR="${DDP_LOG_ROOT}/ddp_pretrain_${PBS_JOBID:-manual}_${RUN_TAG}"
mkdir -p "${RUN_DIR}/logs"

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

echo "=== DDP TOPOLOGY ==="
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "MASTER_HOST=${MASTER_HOST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "RUN_DIR=${RUN_DIR}"
nl -ba "${HOSTS_FILE}"
echo

ENVCONF="${RUN_DIR}/env.conf"
cat > "${ENVCONF}" <<EOF
WORKDIR="${WORKDIR}"
RUN_DIR="${RUN_DIR}"
HOSTS_FILE="${HOSTS_FILE}"
NNODES="${NNODES}"
MASTER_ADDR="${MASTER_ADDR}"
MASTER_PORT="${MASTER_PORT}"
NPROC_PER_NODE="${NPROC_PER_NODE}"
NUM_PROCESSES="${NUM_PROCESSES}"
NUM_MACHINES="${NUM_MACHINES}"
CUDA_MODULE="${CUDA_MODULE}"
VENV_ACTIVATE="${VENV_ACTIVATE}"
PREFERRED_IFNAME="${PREFERRED_IFNAME}"
EOF

NODE_ENTRY="${RUN_DIR}/node_entry.sh"
cat > "${NODE_ENTRY}" <<'SH'
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.conf"

host="$(hostname)"
log="${RUN_DIR}/logs/${host}.pretrain.log"
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

echo "=== NODE ENTRY ==="
echo "host=${host}"
echo "NODE_RANK=${NODE_RANK}"
echo "NUM_MACHINES=${NUM_MACHINES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA:-}"
echo "python=$(which python)"
python -V || true

exec env \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  NUM_MACHINES="${NUM_MACHINES}" \
  MACHINE_RANK="${NODE_RANK}" \
  MAIN_PROCESS_IP="${MASTER_ADDR}" \
  MAIN_PROCESS_PORT="${MASTER_PORT}" \
  bash "${WORKDIR}/scripts/pretrain/nepa3d_pretrain.sh"
SH
chmod +x "${NODE_ENTRY}"

echo "=== LAUNCH via pbsdsh ==="
echo "+ pbsdsh -c ${NNODES} -- bash ${NODE_ENTRY}"
pbsdsh -c "${NNODES}" -- bash "${NODE_ENTRY}"

echo "=== DONE ==="
echo "Per-node logs:"
ls -lh "${RUN_DIR}/logs" | sed -n '1,200p'
