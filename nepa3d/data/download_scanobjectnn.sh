#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/ScanObjectNN}"
ZIP_NAME="h5_files.zip"
URL="${SCANOBJECTNN_URL:-https://hkust-vgd.ust.hk/scanobjectnn/h5_files.zip}"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

if [ ! -f "${ZIP_NAME}" ]; then
  echo "[download] ${URL}"
  if command -v wget >/dev/null 2>&1; then
    wget -c "${URL}" -O "${ZIP_NAME}"
  else
    curl -L -C - "${URL}" -o "${ZIP_NAME}"
  fi
else
  echo "[skip] ${ZIP_NAME} already exists"
fi

if [ ! -d "h5_files" ]; then
  echo "[extract] ${ZIP_NAME}"
  unzip -q "${ZIP_NAME}"
else
  echo "[skip] h5_files already exists"
fi

echo "[done] ScanObjectNN root: ${OUT_DIR}"
echo "[hint] for preprocess, use scan_root=${OUT_DIR}/h5_files"
