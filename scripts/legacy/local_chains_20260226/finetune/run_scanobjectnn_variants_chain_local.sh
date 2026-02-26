#!/usr/bin/env bash
set -eu

# End-to-end chain:
#   1) Wait for/complete core3 variant tables (scratch + shapenet_nepa + shapenet_mesh_udf_nepa)
#   2) Run paper-safe mixed pretrains (mainsplit): mix_nepa + mix_mae
#   3) Run mix-only variant tables (shapenet_mix_nepa + shapenet_mix_mae)
#
# The script is restart-safe:
#   - fine-tune jobs are skipped if last.pt exists
#   - pretrains are skipped if final ckpt exists

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

wait_for_pid_file() {
  local pid_file="$1"
  local label="$2"
  if [ ! -f "${pid_file}" ]; then
    return 0
  fi
  local pid
  pid="$(cat "${pid_file}" 2>/dev/null || true)"
  if [ -z "${pid}" ]; then
    return 0
  fi
  if ps -p "${pid}" >/dev/null 2>&1; then
    echo "[wait] ${label} pid=${pid}"
    while ps -p "${pid}" >/dev/null 2>&1; do
      sleep 30
    done
    echo "[wait] ${label} finished"
  fi
}

count_done() {
  local run_root="$1"
  find "${run_root}" -type f -name last.pt 2>/dev/null | wc -l
}

echo "[stage1] core3 variant tables"
CORE3_VARIANTS="${CORE3_VARIANTS:-obj_bg obj_only pb_t50_rs}"
CORE3_METHODS="${CORE3_METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa}"
CORE3_BASE_RUN_ROOT="${CORE3_BASE_RUN_ROOT:-runs/scan_variants_core3}"
CORE3_BASE_LOG_ROOT="${CORE3_BASE_LOG_ROOT:-logs/finetune/scan_variants_core3}"
CORE3_EXPECTED="${CORE3_EXPECTED:-135}" # 3 variants * 3 methods * 5 K * 3 seeds
CORE3_PID_FILE="${CORE3_BASE_LOG_ROOT}/pipeline.pid"

core3_done="$(count_done "${CORE3_BASE_RUN_ROOT}")"
echo "[stage1] done=${core3_done}/${CORE3_EXPECTED}"
if [ "${core3_done}" -lt "${CORE3_EXPECTED}" ]; then
  if [ -f "${CORE3_PID_FILE}" ] && ps -p "$(cat "${CORE3_PID_FILE}")" >/dev/null 2>&1; then
    echo "[stage1] attach existing pipeline pid=$(cat "${CORE3_PID_FILE}")"
  else
    echo "[stage1] launch core3 pipeline"
    VARIANTS="${CORE3_VARIANTS}" \
    METHODS="${CORE3_METHODS}" \
    BASE_RUN_ROOT="${CORE3_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${CORE3_BASE_LOG_ROOT}" \
      bash scripts/finetune/launch_scanobjectnn_variant_tables_local.sh
  fi
  wait_for_pid_file "${CORE3_PID_FILE}" "core3"
fi
core3_done="$(count_done "${CORE3_BASE_RUN_ROOT}")"
echo "[stage1] done=${core3_done}/${CORE3_EXPECTED}"
if [ "${core3_done}" -lt "${CORE3_EXPECTED}" ]; then
  echo "[error] core3 table not complete"
  exit 1
fi

echo "[stage2] mix pretrains (mainsplit)"
MIX_RUN_ROOT="${MIX_RUN_ROOT:-runs}"
MIX_LOG_DIR="${MIX_LOG_DIR:-logs/pretrain/mix_mainsplit}"
MIX_PID_FILE="${MIX_LOG_DIR}/pipeline.pid"
MIX_NEPA_CKPT="${MIX_NEPA_CKPT:-${MIX_RUN_ROOT}/shapenet_mix_nepa_mainsplit_s0/ckpt_ep049.pt}"
MIX_MAE_CKPT="${MIX_MAE_CKPT:-${MIX_RUN_ROOT}/shapenet_mix_mae_mainsplit_s0/ckpt_ep049.pt}"

if [ -f "${MIX_NEPA_CKPT}" ] && [ -f "${MIX_MAE_CKPT}" ]; then
  echo "[stage2] already complete"
else
  if [ -f "${MIX_PID_FILE}" ] && ps -p "$(cat "${MIX_PID_FILE}")" >/dev/null 2>&1; then
    echo "[stage2] attach existing pretrain pid=$(cat "${MIX_PID_FILE}")"
  else
    echo "[stage2] launch pretrain pipeline"
    LOG_DIR="${MIX_LOG_DIR}" \
      bash scripts/pretrain/launch_shapenet_mix_pretrains_mainsplit_local.sh
  fi
  wait_for_pid_file "${MIX_PID_FILE}" "mix_pretrain"
fi
if [ ! -f "${MIX_NEPA_CKPT}" ] || [ ! -f "${MIX_MAE_CKPT}" ]; then
  echo "[error] mix pretrains not complete"
  exit 1
fi
echo "[stage2] complete"

echo "[stage3] mix-only variant tables"
MIX2_VARIANTS="${MIX2_VARIANTS:-obj_bg obj_only pb_t50_rs}"
MIX2_METHODS="${MIX2_METHODS:-shapenet_mix_nepa shapenet_mix_mae}"
MIX2_BASE_RUN_ROOT="${MIX2_BASE_RUN_ROOT:-runs/scan_variants_mix2_mainsplit}"
MIX2_BASE_LOG_ROOT="${MIX2_BASE_LOG_ROOT:-logs/finetune/scan_variants_mix2_mainsplit}"
MIX2_EXPECTED="${MIX2_EXPECTED:-90}" # 3 variants * 2 methods * 5 K * 3 seeds
MIX2_PID_FILE="${MIX2_BASE_LOG_ROOT}/pipeline.pid"

mix2_done="$(count_done "${MIX2_BASE_RUN_ROOT}")"
echo "[stage3] done=${mix2_done}/${MIX2_EXPECTED}"
if [ "${mix2_done}" -lt "${MIX2_EXPECTED}" ]; then
  if [ -f "${MIX2_PID_FILE}" ] && ps -p "$(cat "${MIX2_PID_FILE}")" >/dev/null 2>&1; then
    echo "[stage3] attach existing pipeline pid=$(cat "${MIX2_PID_FILE}")"
  else
    echo "[stage3] launch mix-only variant pipeline"
    VARIANTS="${MIX2_VARIANTS}" \
    METHODS="${MIX2_METHODS}" \
    BASE_RUN_ROOT="${MIX2_BASE_RUN_ROOT}" \
    BASE_LOG_ROOT="${MIX2_BASE_LOG_ROOT}" \
    SHAPENET_MIX_NEPA_CKPT="${MIX_NEPA_CKPT}" \
    SHAPENET_MIX_MAE_CKPT="${MIX_MAE_CKPT}" \
      bash scripts/finetune/launch_scanobjectnn_variant_tables_local.sh
  fi
  wait_for_pid_file "${MIX2_PID_FILE}" "mix2"
fi
mix2_done="$(count_done "${MIX2_BASE_RUN_ROOT}")"
echo "[stage3] done=${mix2_done}/${MIX2_EXPECTED}"
if [ "${mix2_done}" -lt "${MIX2_EXPECTED}" ]; then
  echo "[error] mix-only variant table not complete"
  exit 1
fi

echo "[done] full chain completed"

