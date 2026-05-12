#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointgpt_oneoff_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_TAG="${RUN_TAG:-pointgpt_oneoff_$(date +%Y%m%d_%H%M%S)}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-01:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointgpt_oneoff}"
POINTGPT_CMD="${POINTGPT_CMD:-}"

CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi
if [[ -z "${POINTGPT_CMD}" ]]; then
  echo "[error] POINTGPT_CMD is required"
  exit 2
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "POINTGPT_DIR=${POINTGPT_DIR}"
  "VENV_ACTIVATE=${VENV_ACTIVATE}"
  "LOG_ROOT=${LOG_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "POINTGPT_CMD=${POINTGPT_CMD}"
  "CUDA_MODULE=${CUDA_MODULE}"
  "GCC_MODULE=${GCC_MODULE}"
  "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "ptgpt_oneoff"
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
