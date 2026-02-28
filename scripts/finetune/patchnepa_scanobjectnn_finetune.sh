#!/usr/bin/env bash
set -euo pipefail

# PatchNEPA direct finetune entrypoint for ScanObjectNN variants.
# This wraps the shared finetune entry with model_source=patchnepa by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

export MODEL_SOURCE="${MODEL_SOURCE:-patchnepa}"

# Keep strict evaluation defaults unless explicitly overridden.
export VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
export AUG_EVAL="${AUG_EVAL:-1}"
export MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"

exec "${ROOT_DIR}/scripts/finetune/patchcls_scanobjectnn_scratch.sh"

