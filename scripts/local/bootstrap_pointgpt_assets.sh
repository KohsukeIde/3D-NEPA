#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
POINTGPT_DIR="${POINTGPT_DIR:-${REPO_ROOT}/PointGPT}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv-pointgpt/bin/python}"

DOWNLOAD_SCANOBJECTNN="${DOWNLOAD_SCANOBJECTNN:-1}"
DOWNLOAD_SHAPENET55="${DOWNLOAD_SHAPENET55:-1}"
DOWNLOAD_OFFICIAL_CKPTS="${DOWNLOAD_OFFICIAL_CKPTS:-0}"
FORCE="${FORCE:-0}"

SCAN_ROOT="${SCAN_ROOT:-${POINTGPT_DIR}/data/ScanObjectNN}"
SHAPENET55_ROOT="${SHAPENET55_ROOT:-${POINTGPT_DIR}/data/ShapeNet55-34}"
CKPT_ROOT="${CKPT_ROOT:-${POINTGPT_DIR}/checkpoints/official}"

SHAPENET55_GDRIVE_URL="${SHAPENET55_GDRIVE_URL:-https://drive.google.com/file/d/1jUB5yD7DP97-EqqU2A9mmr61JpNwZBVK/view?usp=sharing}"
SHAPENET55_TRAIN_URL="${SHAPENET55_TRAIN_URL:-https://raw.githubusercontent.com/lulutang0608/Point-BERT/master/data/ShapeNet55-34/ShapeNet-55/train.txt}"
SHAPENET55_TEST_URL="${SHAPENET55_TEST_URL:-https://raw.githubusercontent.com/lulutang0608/Point-BERT/master/data/ShapeNet55-34/ShapeNet-55/test.txt}"
POINTGPT_S_PRETRAIN_URL="${POINTGPT_S_PRETRAIN_URL:-https://drive.google.com/file/d/1gTFI327kXVDFQ90JfYX0zIS4opM1EkqX/view?usp=drive_link}"
POINTGPT_S_SCAN_HARDEST_URL="${POINTGPT_S_SCAN_HARDEST_URL:-https://drive.google.com/file/d/12Tj2OFKsEPT5zd5nQQ2VNEZlCKHncdGh/view?usp=drive_link}"
POINTGPT_S_SCAN_OBJBG_URL="${POINTGPT_S_SCAN_OBJBG_URL:-https://drive.google.com/file/d/1s4RrBkfwVr8r0H2FxwiHULcyMe_EAJ9D/view?usp=drive_link}"
POINTGPT_S_SCAN_OBJONLY_URL="${POINTGPT_S_SCAN_OBJONLY_URL:-https://drive.google.com/file/d/173yfDAlqqed-oRHaogX6DC4Uj1b8Rvxt/view?usp=drive_link}"
POINTGPT_B_PRETRAIN_URL="${POINTGPT_B_PRETRAIN_URL:-https://drive.google.com/file/d/1Gyf9ZR8MCPg1XOCALjJR9VJepV7iAi5S/view?usp=sharing}"
POINTGPT_B_POST_PRETRAIN_URL="${POINTGPT_B_POST_PRETRAIN_URL:-https://drive.google.com/file/d/1Gc7thuU-D1Sq4NIMTV6-U1LhVN0E2z9l/view?usp=sharing}"
POINTGPT_B_SCAN_HARDEST_URL="${POINTGPT_B_SCAN_HARDEST_URL:-https://drive.google.com/file/d/1tHi7W935DxVttXHG0Mgb0HSfYWUqXLwB/view?usp=sharing}"
POINTGPT_B_SCAN_OBJBG_URL="${POINTGPT_B_SCAN_OBJBG_URL:-https://drive.google.com/file/d/1te8DuC_-cOzt4JayyaNWvxHcRztjDlGF/view?usp=sharing}"
POINTGPT_B_SCAN_OBJONLY_URL="${POINTGPT_B_SCAN_OBJONLY_URL:-https://drive.google.com/file/d/17c8KvDrAuY0GgcO7SGE-4zlMArjzkjLX/view?usp=sharing}"

mkdir -p "${POINTGPT_DIR}/data" "${CKPT_ROOT}"
mkdir -p "${REPO_ROOT}/data"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}" >&2
  exit 2
fi

run_gdown() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${out}"
    return 0
  fi
  mkdir -p "$(dirname "${out}")"
  "${PYTHON_BIN}" -m gdown --fuzzy --continue "${url}" -O "${out}"
}

run_curl() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" && "${FORCE}" != "1" ]]; then
    echo "[skip] ${out}"
    return 0
  fi
  mkdir -p "$(dirname "${out}")"
  curl -fsSL "${url}" -o "${out}"
}

if [[ "${DOWNLOAD_SCANOBJECTNN}" == "1" ]]; then
  echo "[step] downloading ScanObjectNN to ${SCAN_ROOT}"
  bash "${REPO_ROOT}/nepa3d/data/download_scanobjectnn.sh" "${SCAN_ROOT}"
  ln -sfn "${SCAN_ROOT}" "${REPO_ROOT}/data/ScanObjectNN"
fi

if [[ "${DOWNLOAD_SHAPENET55}" == "1" ]]; then
  echo "[step] downloading ShapeNet55-34 to ${SHAPENET55_ROOT}"
  mkdir -p "${SHAPENET55_ROOT}"
  SHAPENET55_ZIP="${SHAPENET55_ROOT}/ShapeNet55.zip"
  run_gdown "${SHAPENET55_GDRIVE_URL}" "${SHAPENET55_ZIP}"
  if [[ ! -d "${SHAPENET55_ROOT}/shapenet_pc" && ! -d "${SHAPENET55_ROOT}/ShapeNet55/shapenet_pc" || "${FORCE}" == "1" ]]; then
    (cd "${SHAPENET55_ROOT}" && unzip -qo "${SHAPENET55_ZIP}")
  fi
  if [[ ! -e "${SHAPENET55_ROOT}/shapenet_pc" && -d "${SHAPENET55_ROOT}/ShapeNet55/shapenet_pc" ]]; then
    ln -sfn "${SHAPENET55_ROOT}/ShapeNet55/shapenet_pc" "${SHAPENET55_ROOT}/shapenet_pc"
  fi
  mkdir -p "${SHAPENET55_ROOT}/ShapeNet-55"
  run_curl "${SHAPENET55_TRAIN_URL}" "${SHAPENET55_ROOT}/ShapeNet-55/train.txt"
  run_curl "${SHAPENET55_TEST_URL}" "${SHAPENET55_ROOT}/ShapeNet-55/test.txt"
  ln -sfn "${SHAPENET55_ROOT}" "${REPO_ROOT}/data/ShapeNet55-34"
fi

if [[ "${DOWNLOAD_OFFICIAL_CKPTS}" == "1" ]]; then
  echo "[step] downloading official PointGPT-S checkpoints to ${CKPT_ROOT}"
  run_gdown "${POINTGPT_S_PRETRAIN_URL}" "${CKPT_ROOT}/pointgpt_s_pretrain_official.pth"
  run_gdown "${POINTGPT_S_SCAN_HARDEST_URL}" "${CKPT_ROOT}/pointgpt_s_scan_hardest_official.pth"
  run_gdown "${POINTGPT_S_SCAN_OBJBG_URL}" "${CKPT_ROOT}/pointgpt_s_scan_objbg_official.pth"
  run_gdown "${POINTGPT_S_SCAN_OBJONLY_URL}" "${CKPT_ROOT}/pointgpt_s_scan_objonly_official.pth"
  run_gdown "${POINTGPT_B_PRETRAIN_URL}" "${CKPT_ROOT}/pointgpt_b_pretrain_official.pth"
  run_gdown "${POINTGPT_B_POST_PRETRAIN_URL}" "${CKPT_ROOT}/pointgpt_b_post_pretrain_official.pth"
  run_gdown "${POINTGPT_B_SCAN_HARDEST_URL}" "${CKPT_ROOT}/pointgpt_b_scan_hardest_official.pth"
  run_gdown "${POINTGPT_B_SCAN_OBJBG_URL}" "${CKPT_ROOT}/pointgpt_b_scan_objbg_official.pth"
  run_gdown "${POINTGPT_B_SCAN_OBJONLY_URL}" "${CKPT_ROOT}/pointgpt_b_scan_objonly_official.pth"
fi

echo "[done] PointGPT assets bootstrap finished"
