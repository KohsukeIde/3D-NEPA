#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointmae_scan_scratch_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

VARIANT="${VARIANT:-pb_t50_rs}"  # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_scratch_$(date +%Y%m%d_%H%M%S)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
CFG_PROFILE="${CFG_PROFILE:-standard}"  # standard|sanity
TOTAL_BS_OVERRIDE="${TOTAL_BS_OVERRIDE:-}"
NPOINT_OVERRIDE="${NPOINT_OVERRIDE:-}"
NO_TEST_AS_VAL="${NO_TEST_AS_VAL:-1}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

if [[ "${NO_TEST_AS_VAL}" != "0" && "${NO_TEST_AS_VAL}" != "1" ]]; then
  echo "[error] NO_TEST_AS_VAL must be 0 or 1 (got: ${NO_TEST_AS_VAL})"
  exit 2
fi

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointmae_scratch}"
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
  "CFG_PROFILE=${CFG_PROFILE}"
  "TOTAL_BS_OVERRIDE=${TOTAL_BS_OVERRIDE}"
  "NPOINT_OVERRIDE=${NPOINT_OVERRIDE}"
  "NO_TEST_AS_VAL=${NO_TEST_AS_VAL}"
  "VAL_RATIO=${VAL_RATIO}"
  "VAL_SEED=${VAL_SEED}"
  "LOG_ROOT=${LOG_DIR}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "pm_scratch_${VARIANT}"
  -o "${LOG_DIR}/${RUN_TAG}.out"
  -e "${LOG_DIR}/${RUN_TAG}.err"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${RUN_SCRIPT}" )

echo "[submit] variant=${VARIANT} run_tag=${RUN_TAG}"
if [[ "${NO_TEST_AS_VAL}" == "1" ]]; then
  echo "[policy] STRICT (no test-as-val): val is split from train (val_ratio=${VAL_RATIO}, val_seed=${VAL_SEED})"
else
  echo "[policy] LEGACY test-as-val enabled by explicit NO_TEST_AS_VAL=0"
fi
jid="$("${cmd[@]}")"
echo "[submitted] ${jid}"
echo "[logs] ${LOG_DIR}/${RUN_TAG}.{out,err,log}"
