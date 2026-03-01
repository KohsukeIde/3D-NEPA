#!/usr/bin/env bash
set -euo pipefail

# PatchNEPA direct finetune entrypoint for ScanObjectNN variants.
# This wraps the shared finetune entry with model_source=patchnepa by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

export MODEL_SOURCE="${MODEL_SOURCE:-patchnepa}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-finetune}"
export WANDB_GROUP="${WANDB_GROUP:-patchnepa-ft}"
export WANDB_TAGS="${WANDB_TAGS:-patchnepa,finetune,direct}"

# Keep strict evaluation defaults unless explicitly overridden.
export VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
export AUG_EVAL="${AUG_EVAL:-1}"
export MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"

# NEPA-paper aligned PatchNEPA direct-FT defaults.
export POOLING="${POOLING:-cls_max}"
export HEAD_MODE="${HEAD_MODE:-pointmae_mlp}"
export PATCHNEPA_CLS_TOKEN_SOURCE="${PATCHNEPA_CLS_TOKEN_SOURCE:-last_q}"
export PATCHNEPA_FREEZE_PATCH_EMBED="${PATCHNEPA_FREEZE_PATCH_EMBED:-1}"
export LLRD_START="${LLRD_START:-1.0}"
export LLRD_END="${LLRD_END:-1.0}"
export LLRD_SCHEDULER="${LLRD_SCHEDULER:-static}"
export LLRD_MODE="${LLRD_MODE:-linear}"

exec "${ROOT_DIR}/scripts/finetune/patchcls_scanobjectnn_scratch.sh"

