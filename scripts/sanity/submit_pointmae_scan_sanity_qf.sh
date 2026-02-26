#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointmae_scan_sanity_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

VARIANT="${VARIANT:-pb_t50_rs}"  # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_sanity_$(date +%Y%m%d_%H%M%S)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-12:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
QSUB_DEPEND="${QSUB_DEPEND:-}"
USE_NEPA_CACHE="${USE_NEPA_CACHE:-0}"
NEPA_CACHE_ROOT="${NEPA_CACHE_ROOT:-}"
CACHE_H5_ROOT_BASE="${CACHE_H5_ROOT_BASE:-}"
CACHE_H5_OVERWRITE="${CACHE_H5_OVERWRITE:-0}"

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointmae}"
mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "VARIANT=${VARIANT}"
  "RUN_TAG=${RUN_TAG}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "SEED=${SEED}"
  "LOG_ROOT=${LOG_DIR}"
  "USE_NEPA_CACHE=${USE_NEPA_CACHE}"
  "NEPA_CACHE_ROOT=${NEPA_CACHE_ROOT}"
  "CACHE_H5_ROOT_BASE=${CACHE_H5_ROOT_BASE}"
  "CACHE_H5_OVERWRITE=${CACHE_H5_OVERWRITE}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "pm_sanity_${VARIANT}"
  -o "${LOG_DIR}/${RUN_TAG}.out"
  -e "${LOG_DIR}/${RUN_TAG}.err"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${RUN_SCRIPT}" )

echo "[submit] variant=${VARIANT} run_tag=${RUN_TAG}"
jid="$("${cmd[@]}")"
echo "[submitted] ${jid}"
echo "[logs] ${LOG_DIR}/${RUN_TAG}.{out,err,log}"
