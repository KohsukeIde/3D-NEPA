#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

CONFIG="${CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml}"
cfg_stem="$(basename "${CONFIG}" .yaml)"
cfg_parent="$(basename "$(dirname "${CONFIG}")")"
EXP_NAME="${EXP_NAME:-${cfg_stem}}"
RUN_TAG="${RUN_TAG:-${EXP_NAME}}"
SAVE_DIR="${GEOPCP_SAVE_DIR_OVERRIDE:-${PCP_ROOT}/experiments/${cfg_stem}/${cfg_parent}/${EXP_NAME}}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/pretrain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
TMUX_SESSION="${TMUX_SESSION:-${RUN_TAG//[^[:alnum:]_]/_}}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PCP_ROOT CONFIG cfg_stem cfg_parent EXP_NAME RUN_TAG SAVE_DIR LOG_ROOT LOG_FILE PID_FILE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT="${MASTER_PORT:-29741}"
export SEED="${SEED:-0}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export USE_WANDB="${GEOPCP_USE_WANDB:-1}"
export WANDB_PROJECT="${GEOPCP_PRETRAIN_WANDB_PROJECT:-patchnepa-geopcp-pretrain}"
export WANDB_ENTITY="${GEOPCP_WANDB_ENTITY:-}"
export WANDB_GROUP="${GEOPCP_WANDB_GROUP:-routea_geopcp}"
export WANDB_RUN_NAME="${GEOPCP_WANDB_RUN_NAME:-${RUN_TAG}}"
export WANDB_MODE="${GEOPCP_WANDB_MODE:-online}"
export WANDB_JOB_TYPE="${GEOPCP_WANDB_JOB_TYPE:-pretrain}"
export WANDB_DIR="${GEOPCP_WANDB_DIR:-${PCP_ROOT}/wandb}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] geopcp pretrain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting Geo-PCP pretrain in foreground"
  echo "[info] config=${CONFIG} exp_name=${EXP_NAME}"
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_pretrain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    command -v tmux >/dev/null 2>&1 || geopcp_die "tmux not found"
    if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] geopcp pretrain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      ROOT_DIR PCP_ROOT PYTHON_BIN CONDA_PREFIX CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST MAX_JOBS PYTHONPATH PATH LD_LIBRARY_PATH \
      CONFIG cfg_stem cfg_parent EXP_NAME RUN_TAG SAVE_DIR LOG_ROOT LOG_FILE PID_FILE \
      CUDA_VISIBLE_DEVICES NPROC_PER_NODE MASTER_PORT SEED NUM_WORKERS OMP_NUM_THREADS MKL_NUM_THREADS \
      OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS USE_WANDB WANDB_PROJECT WANDB_ENTITY WANDB_GROUP WANDB_RUN_NAME WANDB_MODE
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_pretrain_inner.sh'"
    sleep 1
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null || geopcp_die "failed to start tmux session=${TMUX_SESSION}"
    echo "[info] started detached Geo-PCP pretrain in tmux"
    echo "[info] session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_pretrain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    kill -0 "${child_pid}" 2>/dev/null || geopcp_die "failed to start detached Geo-PCP pretrain"
    echo "[info] started detached Geo-PCP pretrain"
    echo "[info] pid=${child_pid}"
    echo "[info] log=${LOG_FILE}"
    ;;
  *)
    geopcp_die "unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    ;;
esac
