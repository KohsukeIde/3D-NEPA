#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POINTGPT_DIR="${POINTGPT_DIR:-${REPO_ROOT}/PointGPT}"
SRC_ROOT="${SRC_ROOT:-${POINTGPT_DIR}/data/ShapeNet55-34}"
OUT_ROOT="${OUT_ROOT:-${POINTGPT_DIR}/data/ShapeNet55-34-smoke}"
TRAIN_COUNT="${TRAIN_COUNT:-256}"
TEST_COUNT="${TEST_COUNT:-64}"

TRAIN_SRC="${SRC_ROOT}/ShapeNet-55/train.txt"
TEST_SRC="${SRC_ROOT}/ShapeNet-55/test.txt"

if [[ ! -f "${TRAIN_SRC}" || ! -f "${TEST_SRC}" ]]; then
  echo "[error] ShapeNet55 train/test manifests missing under ${SRC_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${SRC_ROOT}/shapenet_pc" ]]; then
  echo "[error] ShapeNet55 point clouds missing under ${SRC_ROOT}/shapenet_pc" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/ShapeNet-55"
ln -sfn "${SRC_ROOT}/shapenet_pc" "${OUT_ROOT}/shapenet_pc"
head -n "${TRAIN_COUNT}" "${TRAIN_SRC}" > "${OUT_ROOT}/ShapeNet-55/train.txt"
head -n "${TEST_COUNT}" "${TEST_SRC}" > "${OUT_ROOT}/ShapeNet-55/test.txt"

echo "[done] prepared PointGPT smoke ShapeNet55 subset"
echo "src_root=${SRC_ROOT}"
echo "out_root=${OUT_ROOT}"
echo "train_count=$(wc -l < "${OUT_ROOT}/ShapeNet-55/train.txt")"
echo "test_count=$(wc -l < "${OUT_ROOT}/ShapeNet-55/test.txt")"
