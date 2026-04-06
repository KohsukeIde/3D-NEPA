#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/full_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/full_chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/full_chain.pid}"

mkdir -p "${LOG_ROOT}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

echo "$$" > "${PID_FILE}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

cfg_stem="$(basename "${PRETRAIN_CONFIG}" .yaml)"
cfg_parent="$(basename "$(dirname "${PRETRAIN_CONFIG}")")"
PRETRAIN_EXP_NAME="${PRETRAIN_EXP_NAME:-${ARM_NAME}}"
PRETRAIN_SAVE_DIR="${PCPMAE_EXPERIMENTS_ROOT}/${cfg_stem}/${cfg_parent}/${PRETRAIN_EXP_NAME}"
PRETRAIN_CKPT="${PRETRAIN_SAVE_DIR}/ckpt-last.pth"

log "start full chain arm=${ARM_NAME}"
if [[ "${PRETRAIN_SKIP:-0}" == "1" ]]; then
  PRETRAIN_CKPT="${PRETRAIN_CKPT_OVERRIDE:?set PRETRAIN_CKPT_OVERRIDE when PRETRAIN_SKIP=1}"
  log "skip pretrain and reuse ckpt=${PRETRAIN_CKPT}"
else
  CONFIG="${PRETRAIN_CONFIG}" EXP_NAME="${PRETRAIN_EXP_NAME}" RUN_TAG="${ARM_NAME}" \
  FOREGROUND=1 LOG_FILE="${LOG_ROOT}/${ARM_NAME}.pretrain.log" PID_FILE="${LOG_ROOT}/${ARM_NAME}.pretrain.pid" \
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_pretrain_local.sh" >> "${LOG_FILE}" 2>&1
fi

[[ -f "${PRETRAIN_CKPT}" ]] || geopcp_die "missing pretrain ckpt=${PRETRAIN_CKPT}"

if [[ "${RUN_SCANOBJECTNN:-1}" == "1" ]]; then
  IFS=',' read -r -a ft_variants <<< "${FT_VARIANTS:-obj_bg,obj_only,hardest}"
  for variant in "${ft_variants[@]}"; do
    log "run ScanObjectNN FT variant=${variant}"
    CKPT="${PRETRAIN_CKPT}" VARIANT="${variant}" \
    bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_scanobjectnn_ft_local.sh" >> "${LOG_FILE}" 2>&1
  done
fi

if [[ "${RUN_SHAPENETPART:-1}" == "1" ]]; then
  log "run ShapeNetPart FT"
  CKPT="${PRETRAIN_CKPT}" LOG_DIR="${ARM_NAME}__shapenetpart" \
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_shapenetpart_ft_local.sh" >> "${LOG_FILE}" 2>&1
fi

if [[ "${RUN_ROUTEB:-1}" == "1" ]]; then
  log "run Route-B harness"
  ARM_TAG="${ARM_NAME}" PRETRAIN_CKPT="${PRETRAIN_CKPT}" FOREGROUND=1 \
  LOG_FILE="${LOG_ROOT}/${ARM_NAME}.routeb.log" PID_FILE="${LOG_ROOT}/${ARM_NAME}.routeb.pid" \
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/run_pcpmae_geopcp_routeb_local.sh" >> "${LOG_FILE}" 2>&1
fi

log "full chain done arm=${ARM_NAME}"
