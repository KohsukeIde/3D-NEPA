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
NUM_WORKERS="${NUM_WORKERS:-8}"
REFRESH="${REFRESH:-0}"
UDF_SURF_MAX_T="${UDF_SURF_MAX_T:-2.0}"
OUT_DIR="${OUT_DIR:-results/data_freeze/patchnepa_world_v3_20260315}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-${OUT_DIR}/augment_world_v3_summary.json}"

mkdir -p "${OUT_DIR}"

set -x
"${PYTHON_BIN}" -m nepa3d.data.augment_shapenet_world_v3 \
  --cache_root "${CACHE_ROOT}" \
  --splits "${SPLITS}" \
  --num_workers "${NUM_WORKERS}" \
  --refresh "${REFRESH}" \
  --udf_surf_max_t "${UDF_SURF_MAX_T}" \
  --out_summary_json "${OUT_SUMMARY_JSON}"
