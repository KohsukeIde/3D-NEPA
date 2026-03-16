#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260311_worldvis}"
SPLITS="${SPLITS:-train:test}"
OUT_DIR="${OUT_DIR:-results/data_audit/world_v3_20260316}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/world_v3_audit_summary.json}"

mkdir -p "${OUT_DIR}"

set -x
"${PYTHON_BIN}" -m nepa3d.data.audit_world_v3 \
  --cache_root "${CACHE_ROOT}" \
  --splits "${SPLITS}" \
  --out_json "${OUT_JSON}"
