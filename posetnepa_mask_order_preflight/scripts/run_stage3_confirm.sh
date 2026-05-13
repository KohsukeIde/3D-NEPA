#!/usr/bin/env bash
set -euo pipefail

# Stage 3: confirmation sweep after Stage 1/2 indicate a plausible geometry
# order. This is still pre-flight, but it is long enough to decide whether to
# implement full frontier-level PosetNEPA.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ORDERS="${ORDERS:-simplified_morton,morton,fixed_random,diffusion_shell,geodesic_shell}"
MASKS="${MASKS:-0.0,0.7}"
MAX_EPOCH="${MAX_EPOCH:-100}"
FT_MAX_EPOCH="${FT_MAX_EPOCH:-100}"
FT_SPLITS="${FT_SPLITS:-hardest,objbg,objonly}"
RUN_TAG="${RUN_TAG:-stage3_$(date +%Y%m%d_%H%M%S)}"
GENERATED_REL="${GENERATED_REL:-posetnepa_mask_order_preflight/generated/${RUN_TAG}}"
MANIFEST_REL="${GENERATED_REL}/manifest.json"
CONFIG_OUT_DIR="${CONFIG_OUT_DIR:-PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/${RUN_TAG}}"

export WORKDIR ORDERS MASKS MAX_EPOCH FT_MAX_EPOCH FT_SPLITS RUN_TAG
export MANIFEST="${WORKDIR}/${MANIFEST_REL}"
export USE_WANDB="${USE_WANDB:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export DRY_RUN="${DRY_RUN:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/00_patch_pointgpt_order_modes.py" --repo-root "${WORKDIR}" --apply
"${PYTHON_BIN}" "${SCRIPT_DIR}/06_verify_patch.py" --repo-root "${WORKDIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/01_make_mask_order_configs.py" \
  --repo-root "${WORKDIR}" \
  --orders "${ORDERS}" \
  --masks "${MASKS}" \
  --max-epoch "${MAX_EPOCH}" \
  --ft-max-epoch "${FT_MAX_EPOCH}" \
  --out-dir "${CONFIG_OUT_DIR}" \
  --manifest "${MANIFEST_REL}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/07_verify_chain.py" --repo-root "${WORKDIR}" --manifest "${MANIFEST_REL}"

bash "${SCRIPT_DIR}/02_pretrain_mask_order_matrix.sh"
FT_SPLITS="${FT_SPLITS}" bash "${SCRIPT_DIR}/03_finetune_scan_pb_t50.sh"
"${PYTHON_BIN}" "${SCRIPT_DIR}/04_extract_pretext_diag.py" --repo-root "${WORKDIR}" --manifest "${MANIFEST_REL}" --out-csv "${GENERATED_REL}/pretext_diag.csv" --out-md "${GENERATED_REL}/pretext_diag.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/04b_extract_finetune_results.py" --repo-root "${WORKDIR}" --manifest "${MANIFEST_REL}" --run-tag "${RUN_TAG}" --splits "${FT_SPLITS}" --out-csv "${GENERATED_REL}/finetune_results.csv" --out-md "${GENERATED_REL}/finetune_results.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/08_extract_learning_curves.py" --repo-root "${WORKDIR}" --manifest "${MANIFEST_REL}" --run-tag "${RUN_TAG}" --splits "${FT_SPLITS}" --out-dir "${GENERATED_REL}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/05_summarize_pf3.py" --repo-root "${WORKDIR}" --manifest "${MANIFEST_REL}" --diag-csv "${GENERATED_REL}/pretext_diag.csv" --results-csv "${GENERATED_REL}/finetune_results.csv" --out-md "${GENERATED_REL}/pf3_summary.md"
