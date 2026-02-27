#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointmae_env_setup_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_TAG="${RUN_TAG:-pointmae_env_setup_$(date +%Y%m%d_%H%M%S)}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CREATE_VENV_IF_MISSING="${CREATE_VENV_IF_MISSING:-0}"
INSTALL_TORCH_IF_MISSING="${INSTALL_TORCH_IF_MISSING:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointmae_env}"
mkdir -p "${LOG_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

qvars=(
  "WORKDIR=${WORKDIR}"
  "VENV_ACTIVATE=${VENV_ACTIVATE}"
  "CREATE_VENV_IF_MISSING=${CREATE_VENV_IF_MISSING}"
  "INSTALL_TORCH_IF_MISSING=${INSTALL_TORCH_IF_MISSING}"
  "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
  "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "CUDA_MODULE=${CUDA_MODULE}"
  "LOG_ROOT=${LOG_DIR}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "pm_env_setup"
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
