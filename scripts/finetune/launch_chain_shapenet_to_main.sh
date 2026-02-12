#!/usr/bin/env bash
set -eu

# Legacy chain launcher (kept for reference).
# Current active track uses M1 launchers instead.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

CHAIN_DIR="${CHAIN_DIR:-logs/finetune/chain_shapenet_to_main}"
CHAIN_LOG="${CHAIN_LOG:-${CHAIN_DIR}/pipeline.log}"
CHAIN_PID="${CHAIN_PID:-${CHAIN_DIR}/pipeline.pid}"

mkdir -p "${CHAIN_DIR}"

echo "[warn] legacy launcher: this chain is not part of the current M1 path"

nohup bash scripts/finetune/launch_scan_main_after_shapenet_table.sh > "${CHAIN_LOG}" 2>&1 &
echo $! > "${CHAIN_PID}"

echo "started chain_pid=$(cat "${CHAIN_PID}")"
echo "chain_log=${CHAIN_LOG}"
echo "note: set MAIN_LAUNCH=... to enable an automatic follow-up stage"
