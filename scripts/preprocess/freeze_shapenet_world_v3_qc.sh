#!/usr/bin/env bash
#PBS -l rt_QC=1
#PBS -l walltime=24:00:00
#PBS -j oe
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

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_TAG="${RUN_TAG:-world_v3_freeze_$(date +%Y%m%d_%H%M%S)}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260311_worldvis}"
SPLITS="${SPLITS:-train:test}"
OUT_DIR="${OUT_DIR:-results/data_freeze/${RUN_TAG}}"
NUM_WORKERS="${NUM_WORKERS:-12}"
REFRESH="${REFRESH:-0}"
UDF_SURF_MAX_T="${UDF_SURF_MAX_T:-2.0}"

mkdir -p "${OUT_DIR}"

echo "[run_tag] ${RUN_TAG}"
echo "[cache_root] ${CACHE_ROOT}"
echo "[splits] ${SPLITS}"
echo "[out_dir] ${OUT_DIR}"
echo "[num_workers] ${NUM_WORKERS}"

OUT_SUMMARY_JSON="${OUT_DIR}/augment_world_v3_summary.json" \
CACHE_ROOT="${CACHE_ROOT}" \
SPLITS="${SPLITS}" \
NUM_WORKERS="${NUM_WORKERS}" \
REFRESH="${REFRESH}" \
UDF_SURF_MAX_T="${UDF_SURF_MAX_T}" \
scripts/preprocess/augment_shapenet_world_v3.sh

OUT_JSON="${OUT_DIR}/world_v3_audit_summary.json" \
CACHE_ROOT="${CACHE_ROOT}" \
SPLITS="${SPLITS}" \
scripts/preprocess/audit_world_v3.sh

OUT_JSON="${OUT_DIR}/subset_watertight_manifest.json" \
OUT_SUMMARY_JSON="${OUT_DIR}/subset_watertight_summary.json" \
OUT_TSV="${OUT_DIR}/subset_watertight_manifest.tsv" \
CACHE_ROOT="${CACHE_ROOT}" \
SPLITS="${SPLITS}" \
scripts/preprocess/build_shapenet_subset_manifest.sh

echo "[done] freeze artifacts under ${OUT_DIR}"
