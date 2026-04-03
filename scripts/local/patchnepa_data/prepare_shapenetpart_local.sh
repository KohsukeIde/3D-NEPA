#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data}"
TARGET_DIR="${TARGET_DIR:-${DATA_ROOT}/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${DATA_ROOT}/downloads}"
ZIP_PATH="${ZIP_PATH:-${DOWNLOAD_DIR}/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip}"
SOURCE_URL="${SOURCE_URL:-https://huggingface.co/datasets/cminst/ShapeNet/resolve/main/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip}"
FALLBACK_SOURCE_URL="${FALLBACK_SOURCE_URL:-https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip}"

mkdir -p "${DOWNLOAD_DIR}"

if [[ -f "${TARGET_DIR}/synsetoffset2category.txt" ]]; then
  echo "[info] ShapeNetPart already ready: ${TARGET_DIR}"
  exit 0
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "[info] downloading ShapeNetPart to ${ZIP_PATH}"
  if ! curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -o "${ZIP_PATH}" "${SOURCE_URL}"; then
    echo "[warn] primary ShapeNetPart source failed: ${SOURCE_URL}"
    echo "[info] retrying with fallback source: ${FALLBACK_SOURCE_URL}"
    curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 -o "${ZIP_PATH}" "${FALLBACK_SOURCE_URL}"
  fi
else
  echo "[info] reusing existing zip: ${ZIP_PATH}"
fi

echo "[info] extracting ShapeNetPart into ${DATA_ROOT}"
unzip -q -o "${ZIP_PATH}" -d "${DATA_ROOT}"

if [[ ! -f "${TARGET_DIR}/synsetoffset2category.txt" ]]; then
  echo "[error] ShapeNetPart extract did not produce expected root: ${TARGET_DIR}"
  exit 1
fi

echo "[info] ShapeNetPart ready: ${TARGET_DIR}"
