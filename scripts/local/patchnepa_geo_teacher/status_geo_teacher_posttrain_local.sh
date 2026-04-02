#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${PRETRAIN_RUN_TAG}.posttrain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${PRETRAIN_RUN_TAG}.posttrain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${PRETRAIN_RUN_TAG//[^[:alnum:]_]/_}_post}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${PRETRAIN_SAVE_ROOT}/${PRETRAIN_RUN_TAG}}"
CKPT_NAME="${CKPT_NAME:-ckpt_final.pt}"
CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/${CKPT_NAME}}"
FT_SAVE_ROOT="${FT_SAVE_ROOT:-${ROOT_DIR}/runs/patchcls_itachi}"
SUITE_RESULT_ROOT="${SUITE_RESULT_ROOT:-${ROOT_DIR}/results/cqa_multitype_itachi}"
COMPLETION_RESULT_ROOT="${COMPLETION_RESULT_ROOT:-${ROOT_DIR}/results/cqa_completion_itachi}"

running=0
pid=""
tmux_running=0
if [[ "${LAUNCH_MODE}" == "tmux" ]] && command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
    tmux_running=1
  fi
fi

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    running=1
  fi
fi

echo "pretrain_run_tag=${PRETRAIN_RUN_TAG}"
echo "ckpt=${CKPT_PATH}"
echo "ckpt_exists=$([[ -f "${CKPT_PATH}" ]] && echo 1 || echo 0)"
echo "launch_mode=${LAUNCH_MODE}"
if [[ "${tmux_running}" == "1" ]]; then
  echo "status=running"
  echo "session=${TMUX_SESSION}"
  [[ "${running}" == "1" ]] && echo "pid=${pid}"
elif [[ "${running}" == "1" ]]; then
  echo "status=running"
  echo "pid=${pid}"
else
  echo "status=stopped"
  [[ -n "${pid}" ]] && echo "last_pid=${pid}"
fi
echo "log=${LOG_FILE}"
echo "ft_run_dirs=$(find "${FT_SAVE_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${PRETRAIN_RUN_TAG}__ft_*" 2>/dev/null | wc -l || true)"
echo "suite_json=$(find "${SUITE_RESULT_ROOT}" -maxdepth 1 -type f -name "${PRETRAIN_RUN_TAG}*.json" 2>/dev/null | wc -l || true)"
echo "completion_json=$(find "${COMPLETION_RESULT_ROOT}" -maxdepth 1 -type f -name "${PRETRAIN_RUN_TAG}*.json" 2>/dev/null | wc -l || true)"

if [[ -f "${LOG_FILE}" ]]; then
  echo "--- log tail ---"
  tail -n 20 "${LOG_FILE}"
fi
