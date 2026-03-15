#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

export RUN_SET="${RUN_SET:-patchnepa_cqa_smoke_$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG="${RUN_TAG:-cqa_visthick_smoke_g2}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa.yaml}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export EPOCHS="${EPOCHS:-1}"
export SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-1000}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
export MIN_LR="${MIN_LR:-1e-6}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export WANDB_PROJECT="${WANDB_PROJECT:-patchnepa-cqa-smoke}"
export WANDB_TAGS="${WANDB_TAGS:-abci,cqa,smoke,visthick}"

exec "${ROOT_DIR}/scripts/abci/submit_patchnepa_current_cqa_pretrain.sh"
