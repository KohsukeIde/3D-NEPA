#!/bin/bash
#
# Multi-node backfill for precomputed FPS order key (e.g., pt_fps_order).
#
# Example:
#   qsub -l rt_QF=2 -l walltime=12:00:00 \
#     -v RUN_TAG=shapenet,CACHE_ROOT=data/shapenet_cache_v0,SPLITS=train,test,PT_KEY=pt_xyz_pool,OUT_KEY=pt_fps_order,FPS_K=2048,WORKERS=32 \
#     scripts/preprocess/migrate_pt_fps_order_multinode_pbsdsh.sh
#
#PBS -l rt_QF=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=qgah50055
#PBS -N migrate_pt_fps
#PBS -o migrate_pt_fps.out
#PBS -e migrate_pt_fps.err

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
RUN_TAG="${RUN_TAG:-ptfps}"
DDP_LOG_ROOT="${DDP_LOG_ROOT:-${WORKDIR}/logs/ddp_migrate_ptfps}"

CACHE_ROOT="${CACHE_ROOT:?set CACHE_ROOT}"
SPLITS="${SPLITS:-train,test}"
FPS_K="${FPS_K:-2048}"
PT_KEY="${PT_KEY:-pt_xyz_pool}"
OUT_KEY="${OUT_KEY:-pt_fps_order}"
WORKERS="${WORKERS:-32}"
WRITE_MODE="${WRITE_MODE:-append}"
OVERWRITE="${OVERWRITE:-0}"
LOG_EVERY="${LOG_EVERY:-1000}"
CHUNKSIZE="${CHUNKSIZE:-32}"

cd "${WORKDIR}"
if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
module load "${CUDA_MODULE}" 2>/dev/null || true

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
  echo "ERROR: PBS_NODEFILE is missing"
  exit 1
fi

RUN_DIR="${DDP_LOG_ROOT}/ddp_migrate_ptfps_${PBS_JOBID:-manual}_${RUN_TAG}"
mkdir -p "${RUN_DIR}/logs"

HOSTS_FILE="${RUN_DIR}/hosts.txt"
awk '!seen[$0]++{print}' "${PBS_NODEFILE}" > "${HOSTS_FILE}"
NNODES="$(wc -l < "${HOSTS_FILE}" | tr -d ' ')"
NUM_SHARDS="${NUM_SHARDS:-${NNODES}}"

echo "=== MIGRATE TOPOLOGY ==="
echo "RUN_TAG=${RUN_TAG}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "SPLITS=${SPLITS}"
echo "PT_KEY=${PT_KEY} OUT_KEY=${OUT_KEY} FPS_K=${FPS_K}"
echo "NNODES=${NNODES} NUM_SHARDS=${NUM_SHARDS} WORKERS=${WORKERS}"
echo "WRITE_MODE=${WRITE_MODE} OVERWRITE=${OVERWRITE}"
nl -ba "${HOSTS_FILE}"
echo

ENVCONF="${RUN_DIR}/env.conf"
cat > "${ENVCONF}" <<EOF
WORKDIR="${WORKDIR}"
RUN_DIR="${RUN_DIR}"
HOSTS_FILE="${HOSTS_FILE}"
CUDA_MODULE="${CUDA_MODULE}"
VENV_ACTIVATE="${VENV_ACTIVATE}"
CACHE_ROOT="${CACHE_ROOT}"
SPLITS="${SPLITS}"
FPS_K="${FPS_K}"
PT_KEY="${PT_KEY}"
OUT_KEY="${OUT_KEY}"
WORKERS="${WORKERS}"
WRITE_MODE="${WRITE_MODE}"
OVERWRITE="${OVERWRITE}"
LOG_EVERY="${LOG_EVERY}"
CHUNKSIZE="${CHUNKSIZE}"
NUM_SHARDS="${NUM_SHARDS}"
EOF

NODE_ENTRY="${RUN_DIR}/node_entry.sh"
cat > "${NODE_ENTRY}" <<'SH'
#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/env.conf"

host="$(hostname)"
log="${RUN_DIR}/logs/${host}.migrate.log"
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
  echo "ERROR: cannot determine NODE_RANK"
  exit 41
fi

if [[ "${NODE_RANK}" -ge "${NUM_SHARDS}" ]]; then
  echo "skip: NODE_RANK=${NODE_RANK} >= NUM_SHARDS=${NUM_SHARDS}"
  exit 0
fi

echo "=== MIGRATE SHARD ==="
echo "host=${host}"
echo "NODE_RANK=${NODE_RANK}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "SPLITS=${SPLITS}"
echo "PT_KEY=${PT_KEY} OUT_KEY=${OUT_KEY} FPS_K=${FPS_K}"
echo "WORKERS=${WORKERS} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${NODE_RANK}"

export CACHE_ROOT SPLITS FPS_K PT_KEY OUT_KEY WORKERS WRITE_MODE OVERWRITE LOG_EVERY CHUNKSIZE
export NUM_SHARDS
export SHARD_ID="${NODE_RANK}"
exec bash "${WORKDIR}/scripts/preprocess/migrate_add_pt_fps_order.sh"
SH
chmod +x "${NODE_ENTRY}"

echo "=== LAUNCH via pbsdsh ==="
echo "+ pbsdsh -c ${NNODES} -- bash ${NODE_ENTRY}"
pbsdsh -c "${NNODES}" -- bash "${NODE_ENTRY}"

echo "=== DONE ==="
ls -lh "${RUN_DIR}/logs" | sed -n '1,200p'
