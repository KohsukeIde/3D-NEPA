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

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260303}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUT_JSON="${OUT_JSON:-data/shapenet_unpaired_splits_v2_20260303.json}"
SEED="${SEED:-0}"
RATIOS="${RATIOS:-0.34:0.33:0.33}"
ALLOW_EMPTY_SPLITS="${ALLOW_EMPTY_SPLITS:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

RATIO_ARGS="${RATIOS//:/ }"

set -x
"${PYTHON_BIN}" -m nepa3d.data.shapenet_unpaired_split \
  --cache_root "${CACHE_ROOT}" \
  --train_split "${TRAIN_SPLIT}" \
  --eval_split "${EVAL_SPLIT}" \
  --out_json "${OUT_JSON}" \
  --seed "${SEED}" \
  --ratios ${RATIO_ARGS} \
  --allow_empty_splits "${ALLOW_EMPTY_SPLITS}"
