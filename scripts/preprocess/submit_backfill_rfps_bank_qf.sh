#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SCRIPT="${SCRIPT_DIR}/backfill_rfps_bank_qf.sh"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-rfps_bank}"
RUN_TAG="${RUN_TAG:-rfps_bank_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/rfps_bank/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"
PBS_LOG="${PBS_LOG:-${LOG_DIR}/${JOB_NAME}.pbs.log}"
PBS_ERR="${PBS_ERR:-${LOG_DIR}/${JOB_NAME}.pbs.err.log}"

CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v0}"
SPLITS="${SPLITS:-train}"
PT_KEY="${PT_KEY:-pc_xyz}"
OUT_KEY="${OUT_KEY:-pc_rfps_order_bank}"
RFPS_K="${RFPS_K:-1024}"
RFPS_M="${RFPS_M:-4096}"
BANK_SIZE="${BANK_SIZE:-8}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-24}"
WRITE_MODE="${WRITE_MODE:-append}"
OVERWRITE="${OVERWRITE:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
LOG_EVERY="${LOG_EVERY:-1000}"
CHUNKSIZE="${CHUNKSIZE:-32}"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${PBS_LOG}"
  -e "${PBS_ERR}"
  -v "WORKDIR=${WORKDIR},CACHE_ROOT=${CACHE_ROOT},SPLITS=${SPLITS},PT_KEY=${PT_KEY},OUT_KEY=${OUT_KEY},RFPS_K=${RFPS_K},RFPS_M=${RFPS_M},BANK_SIZE=${BANK_SIZE},SEED=${SEED},WORKERS=${WORKERS},WRITE_MODE=${WRITE_MODE},OVERWRITE=${OVERWRITE},NUM_SHARDS=${NUM_SHARDS},SHARD_ID=${SHARD_ID},LOG_EVERY=${LOG_EVERY},CHUNKSIZE=${CHUNKSIZE}"
  "${SCRIPT}"
)

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[log] ${PBS_LOG}"
echo "[err] ${PBS_ERR}"
echo "[cache_root] ${CACHE_ROOT} splits=${SPLITS}"
