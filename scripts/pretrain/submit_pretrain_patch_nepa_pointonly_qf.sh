#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_patch_nepa_qf.sh"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"

RUN_SET="${RUN_SET:-patchnepa_pointonly_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-run_pointonly_${RUN_SET}}"
SAVE_DIR="${SAVE_DIR:-runs/patchnepa_pointonly/${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/patch_nepa_pretrain/${RUN_SET}}"

N_POINT="${N_POINT:-1024}"
N_RAY="${N_RAY:-0}"
USE_RAY_PATCH="${USE_RAY_PATCH:-0}"
BATCH="${BATCH:-96}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"

PATCH_EMBED="${PATCH_EMBED:-fps_knn}"
GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_GROUPS="${NUM_GROUPS:-64}"
SERIAL_ORDER="${SERIAL_ORDER:-morton}"
SERIAL_BITS="${SERIAL_BITS:-10}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only.yaml}"
PT_XYZ_KEY="${PT_XYZ_KEY:-pc_xyz}"
PT_DIST_KEY="${PT_DIST_KEY:-pt_dist_pool}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
PT_SAMPLE_MODE="${PT_SAMPLE_MODE:-random}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found in PATH"
  exit 1
fi

mkdir -p "${WORKDIR}/${LOG_ROOT}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${WORKDIR}/${LOG_ROOT}}"
mkdir -p "${PBS_LOG_DIR}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"

JOB_NAME="${JOB_NAME:-patchnepa_ptonly}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${PBS_LOG_PATH}"
  -v "WORKDIR=${WORKDIR},RUN_TAG=${RUN_TAG},SAVE_DIR=${SAVE_DIR},LOG_ROOT=${LOG_ROOT},MIX_CONFIG=${MIX_CONFIG},N_POINT=${N_POINT},N_RAY=${N_RAY},USE_RAY_PATCH=${USE_RAY_PATCH},BATCH=${BATCH},EPOCHS=${EPOCHS},NUM_WORKERS=${NUM_WORKERS},LR=${LR},SEED=${SEED},PATCH_EMBED=${PATCH_EMBED},GROUP_SIZE=${GROUP_SIZE},NUM_GROUPS=${NUM_GROUPS},SERIAL_ORDER=${SERIAL_ORDER},SERIAL_BITS=${SERIAL_BITS},PT_XYZ_KEY=${PT_XYZ_KEY},PT_DIST_KEY=${PT_DIST_KEY},ABLATE_POINT_DIST=${ABLATE_POINT_DIST},PT_SAMPLE_MODE=${PT_SAMPLE_MODE},POINT_ORDER_MODE=${POINT_ORDER_MODE}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[save_dir] ${SAVE_DIR}"
echo "[log_root] ${LOG_ROOT}"
echo "[pbs_log] ${PBS_LOG_PATH}"
