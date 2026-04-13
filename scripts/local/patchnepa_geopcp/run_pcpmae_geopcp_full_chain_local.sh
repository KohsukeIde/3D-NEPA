#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

ARM_NAME="${ARM_NAME:-pcp_worldvis_base_100ep}"
case "${ARM_NAME}" in
  pcp_worldvis_base_100ep)
    PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-cfgs/geopcp/pcp_worldvis_base_100ep.yaml}"
    ;;
  geopcp_worldvis_base_normal_100ep)
    PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml}"
    ;;
  geopcp_worldvis_base_normal_thickness_100ep)
    PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_thickness_100ep.yaml}"
    ;;
  *)
    geopcp_die "unknown ARM_NAME=${ARM_NAME}"
    ;;
esac

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ARM_NAME}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${ARM_NAME}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${ARM_NAME//[^[:alnum:]_]/_}_chain}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PCP_ROOT PYTHON_BIN CONDA_PREFIX CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST MAX_JOBS PYTHONPATH PATH LD_LIBRARY_PATH
export ARM_NAME PRETRAIN_CONFIG LOG_ROOT LOG_FILE PID_FILE
export PRETRAIN_SKIP="${PRETRAIN_SKIP:-0}"
export PRETRAIN_CKPT_OVERRIDE="${PRETRAIN_CKPT_OVERRIDE:-}"
export RUN_SCANOBJECTNN="${RUN_SCANOBJECTNN:-1}"
export RUN_SHAPENETPART="${RUN_SHAPENETPART:-1}"
export RUN_ROUTEB="${RUN_ROUTEB:-0}"
export FT_VARIANTS="${FT_VARIANTS:-obj_bg,obj_only,hardest}"
export FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] geopcp full chain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_full_chain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    command -v tmux >/dev/null 2>&1 || geopcp_die "tmux not found"
    if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] geopcp full chain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      ROOT_DIR PCP_ROOT PYTHON_BIN CONDA_PREFIX CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST MAX_JOBS PYTHONPATH PATH LD_LIBRARY_PATH \
      ARM_NAME PRETRAIN_CONFIG LOG_ROOT LOG_FILE PID_FILE PRETRAIN_SKIP PRETRAIN_CKPT_OVERRIDE \
      RUN_SCANOBJECTNN RUN_SHAPENETPART RUN_ROUTEB FT_VARIANTS FT_NUM_WORKERS
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_full_chain_inner.sh'"
    sleep 1
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null || geopcp_die "failed to start tmux session=${TMUX_SESSION}"
    echo "[info] started detached Geo-PCP full chain in tmux"
    echo "[info] session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_full_chain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    kill -0 "${child_pid}" 2>/dev/null || geopcp_die "failed to start detached Geo-PCP full chain"
    echo "[info] started detached Geo-PCP full chain"
    echo "[info] pid=${child_pid}"
    echo "[info] log=${LOG_FILE}"
    ;;
  *)
    geopcp_die "unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    ;;
esac
