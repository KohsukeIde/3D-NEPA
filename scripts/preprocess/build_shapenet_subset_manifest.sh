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
OUT_DIR="${OUT_DIR:-results/data_freeze/patchnepa_world_v3_20260315}"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/subset_watertight_manifest.json}"
OUT_SUMMARY_JSON="${OUT_SUMMARY_JSON:-${OUT_DIR}/subset_watertight_summary.json}"
OUT_TSV="${OUT_TSV:-${OUT_DIR}/subset_watertight_manifest.tsv}"
REQUIRE_WATERTIGHT="${REQUIRE_WATERTIGHT:-1}"
REQUIRE_WINDING_CONSISTENT="${REQUIRE_WINDING_CONSISTENT:-1}"
MIN_FACES="${MIN_FACES:-1}"
MIN_VERTICES="${MIN_VERTICES:-1}"
MIN_UDF_HIT_RATE="${MIN_UDF_HIT_RATE:--1}"
MAX_VISIBILITY_ALLZERO_RATE="${MAX_VISIBILITY_ALLZERO_RATE:--1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

set -x
"${PYTHON_BIN}" -m nepa3d.data.build_shapenet_subset_manifest \
  --cache_root "${CACHE_ROOT}" \
  --splits "${SPLITS}" \
  --out_json "${OUT_JSON}" \
  --out_summary_json "${OUT_SUMMARY_JSON}" \
  --out_tsv "${OUT_TSV}" \
  --require_watertight "${REQUIRE_WATERTIGHT}" \
  --require_winding_consistent "${REQUIRE_WINDING_CONSISTENT}" \
  --min_faces "${MIN_FACES}" \
  --min_vertices "${MIN_VERTICES}" \
  --min_udf_hit_rate "${MIN_UDF_HIT_RATE}" \
  --max_visibility_allzero_rate "${MAX_VISIBILITY_ALLZERO_RATE}"
