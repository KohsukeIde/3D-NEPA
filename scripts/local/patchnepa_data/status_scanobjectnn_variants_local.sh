#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_data}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/scanobjectnn_variants_local.pid}"
TMUX_SESSION="${TMUX_SESSION:-patchnepa_scanobjectnn_variants}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"

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

count_npz() {
  local root="$1"
  if [[ ! -d "${root}" ]]; then
    echo "train=0 test=0"
    return 0
  fi
  local tr_n te_n
  tr_n="$(find "${root}/train" -type f -name '*.npz' 2>/dev/null | wc -l || true)"
  te_n="$(find "${root}/test" -type f -name '*.npz' 2>/dev/null | wc -l || true)"
  echo "train=${tr_n} test=${te_n}"
}

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
echo "obj_bg_cache=${OBJ_BG_CACHE} $(count_npz "${OBJ_BG_CACHE}")"
echo "obj_only_cache=${OBJ_ONLY_CACHE} $(count_npz "${OBJ_ONLY_CACHE}")"
echo "pb_t50_rs_cache=${PB_T50_RS_CACHE} $(count_npz "${PB_T50_RS_CACHE}")"

if [[ -f "${LOG_FILE}" ]]; then
  echo "--- log tail ---"
  tail -n 20 "${LOG_FILE}"
fi
