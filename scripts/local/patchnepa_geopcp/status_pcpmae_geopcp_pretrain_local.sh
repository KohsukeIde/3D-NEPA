#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PCP_ROOT="${ROOT_DIR}/PCP-MAE"
CONFIG="${CONFIG:-cfgs/geopcp/geopcp_worldvis_base_normal_100ep.yaml}"
cfg_stem="$(basename "${CONFIG}" .yaml)"
cfg_parent="$(basename "$(dirname "${CONFIG}")")"
EXP_NAME="${EXP_NAME:-${cfg_stem}}"
RUN_TAG="${RUN_TAG:-${EXP_NAME}}"
SAVE_DIR="${SAVE_DIR:-${PCP_ROOT}/experiments/${cfg_stem}/${cfg_parent}/${EXP_NAME}}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/pretrain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
TMUX_SESSION="${TMUX_SESSION:-${RUN_TAG//[^[:alnum:]_]/_}}"

status="stopped"
pid=""
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    status="running"
  fi
fi
if [[ "${status}" != "running" ]] && command -v tmux >/dev/null 2>&1 && env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
  status="running"
fi

echo "status=${status}"
echo "pid=${pid}"
echo "run_tag=${RUN_TAG}"
echo "save_dir=${SAVE_DIR}"
echo "log=${LOG_FILE}"
if [[ -f "${SAVE_DIR}/ckpt-last.pth" ]]; then
  echo "ckpt_last=${SAVE_DIR}/ckpt-last.pth"
fi
