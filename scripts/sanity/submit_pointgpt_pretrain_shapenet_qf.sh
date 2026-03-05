#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/pointgpt_pretrain_shapenet_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

RUN_TAG="${RUN_TAG:-pointgpt_pretrain_shapenet_$(date +%Y%m%d_%H%M%S)}"
RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/pointgpt_pretrain_shapenet}"
EXP_NAME="${EXP_NAME:-${RUN_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-cfgs/PointGPT-S/pretrain.yaml}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6/12.6.2}"
GCC_MODULE="${GCC_MODULE:-gcc/11.4.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SHAPENET_CACHE_ROOT="${SHAPENET_CACHE_ROOT:-${WORKDIR}/data/shapenet_cache_v2_20260303}"
FORCE_REBUILD_SPLITS="${FORCE_REBUILD_SPLITS:-0}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pointgpt-pretrain}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_GROUP="${WANDB_GROUP:-pointgpt_shapenet_pretrain}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${EXP_NAME}}"
WANDB_DIR="${WANDB_DIR:-${WORKDIR}/wandb}"

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
  "SHAPENET_CACHE_ROOT=${SHAPENET_CACHE_ROOT}"
  "FORCE_REBUILD_SPLITS=${FORCE_REBUILD_SPLITS}"
  "USE_WANDB=${USE_WANDB}"
  "WANDB_PROJECT=${WANDB_PROJECT}"
  "WANDB_ENTITY=${WANDB_ENTITY}"
  "WANDB_GROUP=${WANDB_GROUP}"
  "WANDB_RUN_NAME=${WANDB_RUN_NAME}"
  "WANDB_DIR=${WANDB_DIR}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "ptgpt_pt_shn"
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
