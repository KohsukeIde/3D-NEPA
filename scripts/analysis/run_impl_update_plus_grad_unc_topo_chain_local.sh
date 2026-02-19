#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

WAIT_PID_FILE="${WAIT_PID_FILE:-logs/analysis/impl_update_chain/pipeline.pid}"
WAIT_POLL_SEC="${WAIT_POLL_SEC:-120}"
WAIT_FOR_PRIOR="${WAIT_FOR_PRIOR:-0}"

if [ "${WAIT_FOR_PRIOR}" = "1" ]; then
  if [ -f "${WAIT_PID_FILE}" ]; then
    wait_pid="$(cat "${WAIT_PID_FILE}" 2>/dev/null || true)"
    if [ -n "${wait_pid}" ] && kill -0 "${wait_pid}" 2>/dev/null; then
      echo "[$(date +"%F %T")] waiting for prior impl-update chain pid=${wait_pid}"
      while kill -0 "${wait_pid}" 2>/dev/null; do
        sleep "${WAIT_POLL_SEC}"
      done
      echo "[$(date +"%F %T")] prior chain finished; launching plus-grad/unc/topo chain"
    fi
  fi
fi

export MODEL1_TAG="${MODEL1_TAG:-nepa_impl_causal_plusgut_s0_e50}"
export MODEL1_SAVE_DIR="${MODEL1_SAVE_DIR:-runs/eccv_upmix_nepa_impl_causal_plusgut_s0}"
export MODEL1_ARCH="${MODEL1_ARCH:-causal}"
export MODEL1_QA_LAYOUT="${MODEL1_QA_LAYOUT:-interleave}"
export MODEL1_TOPO_K="${MODEL1_TOPO_K:-0}"
export MODEL1_TOPO_RAY_COORD="${MODEL1_TOPO_RAY_COORD:-origin}"
export MODEL1_TOPO_RAY_BBOX="${MODEL1_TOPO_RAY_BBOX:-0.5}"

export MODEL2_TAG="${MODEL2_TAG:-nepa_impl_encdec_plusgut_proj_s0_e50}"
export MODEL2_SAVE_DIR="${MODEL2_SAVE_DIR:-runs/eccv_upmix_nepa_impl_encdec_plusgut_proj_s0}"
export MODEL2_ARCH="${MODEL2_ARCH:-encdec}"
export MODEL2_QA_LAYOUT="${MODEL2_QA_LAYOUT:-split}"
export MODEL2_TOPO_K="${MODEL2_TOPO_K:-16}"
export MODEL2_TOPO_RAY_COORD="${MODEL2_TOPO_RAY_COORD:-proj}"
export MODEL2_TOPO_RAY_BBOX="${MODEL2_TOPO_RAY_BBOX:-0.5}"

export INCLUDE_PT_GRAD="${INCLUDE_PT_GRAD:-1}"
export PT_GRAD_MODE="${PT_GRAD_MODE:-log}"
export PT_GRAD_EPS="${PT_GRAD_EPS:-1e-3}"
export PT_GRAD_CLIP="${PT_GRAD_CLIP:-10.0}"
export PT_GRAD_ORIENT="${PT_GRAD_ORIENT:-ray}"

export INCLUDE_RAY_UNC="${INCLUDE_RAY_UNC:-1}"
export RAY_UNC_K="${RAY_UNC_K:-8}"
export RAY_UNC_MODE="${RAY_UNC_MODE:-normal_var}"

bash scripts/analysis/run_impl_update_chain_local.sh
