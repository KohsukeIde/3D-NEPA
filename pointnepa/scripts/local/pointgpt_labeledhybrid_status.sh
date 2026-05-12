#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
HYBRID_ROOT="${HYBRID_ROOT:-${POINTGPT_DIR}/data/HybridDatasets}"
DATA_ROOT="${DATA_ROOT:-${HYBRID_ROOT}/post_pretrain}"
PC_PATH="${PC_PATH:-${HYBRID_ROOT}}"
CHECK_SAMPLES="${CHECK_SAMPLES:-32}"
FULL_SCAN="${FULL_SCAN:-0}"
DEFAULT_PYTHON_BIN="${WORKDIR}/.venv-pointgpt/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
  DEFAULT_PYTHON_BIN="${WORKDIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python bin missing or not executable: ${PYTHON_BIN}"
  exit 2
fi
if [[ "${FULL_SCAN}" != "0" && "${FULL_SCAN}" != "1" ]]; then
  echo "[error] FULL_SCAN must be 0 or 1 (got: ${FULL_SCAN})"
  exit 2
fi

echo "=== POINTGPT LABELEDHYBRID STATUS ==="
echo "date=$(date -Is)"
echo "pointgpt_dir=${POINTGPT_DIR}"
echo "hybrid_root=${HYBRID_ROOT}"
echo "data_root=${DATA_ROOT}"
echo "pc_path=${PC_PATH}"
echo "check_samples=${CHECK_SAMPLES}"
echo "full_scan=${FULL_SCAN}"
echo

"${PYTHON_BIN}" - "${DATA_ROOT}" "${PC_PATH}" "${CHECK_SAMPLES}" "${FULL_SCAN}" <<'PY'
import json
import os
import sys

data_root = sys.argv[1]
pc_path = sys.argv[2]
check_samples = int(sys.argv[3])
full_scan = sys.argv[4] == "1"

required = [
    "train.txt",
    "test.txt",
    "train_num.txt",
    "test_num.txt",
]

status = {
    "ready": False,
    "data_root": data_root,
    "pc_path": pc_path,
    "required_files": {},
    "splits": {},
    "errors": [],
}

for name in required:
    path = os.path.join(data_root, name)
    status["required_files"][name] = {
        "path": path,
        "exists": os.path.isfile(path),
    }
    if not os.path.isfile(path):
        status["errors"].append(f"missing required file: {path}")

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def inspect_split(split):
    file_list = read_lines(os.path.join(data_root, f"{split}.txt"))
    label_list = read_lines(os.path.join(data_root, f"{split}_num.txt"))
    sample_limit = len(file_list) if full_scan else min(len(file_list), check_samples)
    checked = []
    missing = []
    for rel in file_list[:sample_limit]:
        candidate = os.path.join(pc_path, rel)
        exists = os.path.isfile(candidate)
        checked.append({"rel": rel, "path": candidate, "exists": exists})
        if not exists:
            missing.append(candidate)
    return {
        "num_samples": len(file_list),
        "num_labels": len(label_list),
        "count_match": len(file_list) == len(label_list),
        "checked_samples": sample_limit,
        "all_checked_exist": len(missing) == 0,
        "missing_checked_paths": missing,
        "checked_preview": checked[: min(len(checked), 5)],
    }

if not status["errors"]:
    for split in ("train", "test"):
        info = inspect_split(split)
        status["splits"][split] = info
        if not info["count_match"]:
            status["errors"].append(
                f"{split}: sample/label count mismatch "
                f"({info['num_samples']} vs {info['num_labels']})"
            )
        if not info["all_checked_exist"]:
            status["errors"].append(
                f"{split}: missing point files in checked subset "
                f"({len(info['missing_checked_paths'])} missing)"
            )

status["ready"] = not status["errors"]
print(json.dumps(status, indent=2))
raise SystemExit(0 if status["ready"] else 1)
PY

