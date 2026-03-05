#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointgpt_scan_smoke_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_TAG="${RUN_TAG:-pointgpt_scan_smoke_$(date +%Y%m%d_%H%M%S)}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointgpt_scan_smoke}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-cfgs/local/finetune_scan_objbg_smoke.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OBJ_BG_ROOT="${OBJ_BG_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/obj_bg}"
OBJ_ONLY_ROOT="${OBJ_ONLY_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/obj_only}"
PB_T50_ROOT="${PB_T50_ROOT:-${WORKDIR}/data/ScanObjectNN_h5_from_nepa_cache/pb_t50_rs}"

mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "POINTGPT_DIR=${POINTGPT_DIR}"
  "VENV_ACTIVATE=${VENV_ACTIVATE}"
  "LOG_ROOT=${LOG_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "EXP_NAME=${EXP_NAME}"
  "CONFIG_PATH=${CONFIG_PATH}"
  "NUM_WORKERS=${NUM_WORKERS}"
  "CUDA_MODULE=${CUDA_MODULE}"
  "GCC_MODULE=${GCC_MODULE}"
  "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "OBJ_BG_ROOT=${OBJ_BG_ROOT}"
  "OBJ_ONLY_ROOT=${OBJ_ONLY_ROOT}"
  "PB_T50_ROOT=${PB_T50_ROOT}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "ptgpt_scan_smoke"
  -o "${LOG_DIR}/${RUN_TAG}.out"
  -e "${LOG_DIR}/${RUN_TAG}.err"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${RUN_SCRIPT}" )

echo "[submit] run_tag=${RUN_TAG}"
jid="$("${cmd[@]}")"
echo "[submitted] ${jid}"
echo "[logs] ${LOG_DIR}/${RUN_TAG}.{out,err}"
