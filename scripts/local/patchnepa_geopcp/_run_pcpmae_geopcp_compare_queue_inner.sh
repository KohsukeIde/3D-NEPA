#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/compare_queue}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/geopcp_routea_compare_v1.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/geopcp_routea_compare_v1.pid}"
QUEUE_LOG_FILE="${LOG_FILE}"
QUEUE_PID_FILE="${PID_FILE}"

mkdir -p "${LOG_ROOT}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${QUEUE_LOG_FILE}"
  rm -f "${QUEUE_PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

echo "$$" > "${QUEUE_PID_FILE}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${QUEUE_LOG_FILE}"
}

arm_status() {
  local arm="$1"
  env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION \
    ARM_NAME="${arm}" \
    bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/status_pcpmae_geopcp_full_chain_local.sh" | awk -F= '$1=="status"{print $2}'
}

any_routea_arm_running() {
  local known_arms=(
    "pcp_worldvis_base_100ep"
    "geopcp_worldvis_base_normal_100ep"
    "geopcp_worldvis_base_normal_thickness_100ep"
  )
  local arm status
  for arm in "${known_arms[@]}"; do
    status="$(arm_status "${arm}")"
    if [[ "${status}" == "running" ]]; then
      echo "${arm}"
      return 0
    fi
  done
  return 1
}

IFS=',' read -r -a arms <<< "${ARM_LIST:-pcp_worldvis_base_100ep,geopcp_worldvis_base_normal_100ep,geopcp_worldvis_base_normal_thickness_100ep}"

log "start Geo-PCP compare queue"
for arm in "${arms[@]}"; do
  while running_arm="$(any_routea_arm_running)"; do
    log "wait for running Route-A arm=${running_arm} before launching arm=${arm}"
    sleep 60
  done
  log "run full chain arm=${arm}"
  env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u ENV_FILE \
    ARM_NAME="${arm}" \
    FOREGROUND=1 \
    RUN_SCANOBJECTNN="${RUN_SCANOBJECTNN:-1}" \
    RUN_SHAPENETPART="${RUN_SHAPENETPART:-1}" \
    RUN_ROUTEB="${RUN_ROUTEB:-1}" \
    FT_VARIANTS="${FT_VARIANTS:-obj_bg,obj_only,hardest}" \
    FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}" \
    bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_full_chain_local.sh" >> "${QUEUE_LOG_FILE}" 2>&1
done
log "Geo-PCP compare queue done"
