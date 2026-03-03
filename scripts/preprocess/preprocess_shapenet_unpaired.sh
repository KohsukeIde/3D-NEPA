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
SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-data/shapenet_cache_v2_20260303}"
SPLIT_JSON="${SPLIT_JSON:-data/shapenet_unpaired_splits_v2_20260303.json}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_unpaired_cache_v2_20260303}"
SPLITS="${SPLITS:-train_mesh:train_pc:train_udf:eval}"
LINK_MODE="${LINK_MODE:-symlink}"  # symlink|hardlink|copy
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

EXTRA_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  EXTRA_ARGS+=( --overwrite )
fi
SPLIT_ARGS="${SPLITS//:/ }"

set -x
"${PYTHON_BIN}" -m nepa3d.data.preprocess_shapenet_unpaired \
  --src_cache_root "${SRC_CACHE_ROOT}" \
  --split_json "${SPLIT_JSON}" \
  --out_root "${OUT_ROOT}" \
  --splits ${SPLIT_ARGS} \
  --link_mode "${LINK_MODE}" \
  "${EXTRA_ARGS[@]}"
