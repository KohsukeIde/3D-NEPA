#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
RUN_TAG="${RUN_TAG:-${PRETRAIN_RUN_TAG}__curvature_probe}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post/cqa_probe}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.launch.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${RUN_TAG}.pid}"
TMUX_SESSION="${TMUX_SESSION:-${RUN_TAG//[^[:alnum:]_]/_}}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/runs/cqa_probe_itachi}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/cqa_probe_itachi}"
OUT_JSON="${OUT_JSON:-${RESULT_ROOT}/${RUN_TAG}.json}"
PROBE_LOG_FILE="${PROBE_LOG_FILE:-${LOG_ROOT}/${RUN_TAG}.log}"

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
echo "run_tag=${RUN_TAG}"
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
echo "probe_log=${PROBE_LOG_FILE}"
echo "out_json=${OUT_JSON}"
echo "out_json_exists=$([[ -f "${OUT_JSON}" ]] && echo 1 || echo 0)"
echo "save_dir=${SAVE_DIR}"

if [[ -f "${LOG_FILE}" ]]; then
  echo "--- launch log tail ---"
  tail -n 20 "${LOG_FILE}"
fi

if [[ -f "${PROBE_LOG_FILE}" ]]; then
  echo "--- probe log tail ---"
  tail -n 20 "${PROBE_LOG_FILE}"
fi
