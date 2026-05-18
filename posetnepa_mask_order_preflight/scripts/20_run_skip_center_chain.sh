#!/usr/bin/env bash
set -euo pipefail

# Diagnostic chain for the post-Lite decision:
# - skip-k: is simplified_morton using immediate local continuity?
# - pretrain_position_mode: is positional/center side-channel doing too much?
# - readouts: full fine-tune + frozen linear probe, not full fine-tune alone.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${WORKDIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${WORKDIR}/.venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_TAG="${RUN_TAG:-skipcenter_$(date +%Y%m%d_%H%M%S)}"
GENERATED_REL="${GENERATED_REL:-posetnepa_mask_order_preflight/generated/${RUN_TAG}}"
MANIFEST_REL="${GENERATED_REL}/manifest.json"
CONFIG_OUT_DIR="${CONFIG_OUT_DIR:-PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/${RUN_TAG}}"

MAX_EPOCH="${MAX_EPOCH:-30}"
FT_MAX_EPOCH="${FT_MAX_EPOCH:-50}"
FT_SPLITS="${FT_SPLITS:-hardest}"
USE_WANDB="${USE_WANDB:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DRY_RUN="${DRY_RUN:-0}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_LINEAR_PROBE="${RUN_LINEAR_PROBE:-1}"
PROBE_EPOCHS="${PROBE_EPOCHS:-200}"
PROBE_SEEDS="${PROBE_SEEDS:-0}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
PROBE_CUDA_VISIBLE_DEVICES="${PROBE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}"

DEFAULT_VARIANTS="skip2:k=2,pos=normal,aux=0.0,mask=0.0;skip4:k=4,pos=normal,aux=0.0,mask=0.0;skip8:k=8,pos=normal,aux=0.0,mask=0.0;poszero:k=1,pos=zero,aux=0.0,mask=0.0;posshuffle:k=1,pos=shuffle,aux=0.0,mask=0.0;centeraux:k=1,pos=normal,aux=0.1,mask=0.0;skip4_poszero:k=4,pos=zero,aux=0.0,mask=0.0"
VARIANTS="${VARIANTS:-${DEFAULT_VARIANTS}}"

export WORKDIR RUN_TAG USE_WANDB NPROC_PER_NODE CUDA_VISIBLE_DEVICES NUM_WORKERS DRY_RUN PYTHON_BIN
export MANIFEST="${WORKDIR}/${MANIFEST_REL}"

mkdir -p "${WORKDIR}/${GENERATED_REL}"

echo "[skip-center] start $(date -Is)"
echo "[skip-center] run_tag=${RUN_TAG}"
echo "[skip-center] variants=${VARIANTS}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/19_make_skip_center_configs.py" \
  --repo-root "${WORKDIR}" \
  --variants "${VARIANTS}" \
  --max-epoch "${MAX_EPOCH}" \
  --ft-max-epoch "${FT_MAX_EPOCH}" \
  --out-dir "${CONFIG_OUT_DIR}" \
  --manifest "${MANIFEST_REL}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/07_verify_chain.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${MANIFEST_REL}"

if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  RUN_TAG="${RUN_TAG}" bash "${SCRIPT_DIR}/02_pretrain_mask_order_matrix.sh"
fi

if [[ "${RUN_FINETUNE}" == "1" ]]; then
  RUN_TAG="${RUN_TAG}" FT_SPLITS="${FT_SPLITS}" bash "${SCRIPT_DIR}/03_finetune_scan_pb_t50.sh"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/04_extract_pretext_diag.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${MANIFEST_REL}" \
  --run-tag "${RUN_TAG}" \
  --out-csv "${GENERATED_REL}/pretext_diag.csv" \
  --out-md "${GENERATED_REL}/pretext_diag.md"

"${PYTHON_BIN}" "${SCRIPT_DIR}/09_extract_copy_baseline_diag.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${MANIFEST_REL}" \
  --run-tag "${RUN_TAG}" \
  --final-epoch-only \
  --tail-batches 0 \
  --out-csv "${GENERATED_REL}/copy_baseline_diag.csv" \
  --out-md "${GENERATED_REL}/copy_baseline_diag.md"

if [[ "${RUN_FINETUNE}" == "1" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/04b_extract_finetune_results.py" \
    --repo-root "${WORKDIR}" \
    --manifest "${MANIFEST_REL}" \
    --run-tag "${RUN_TAG}" \
    --splits "${FT_SPLITS}" \
    --out-csv "${GENERATED_REL}/finetune_results.csv" \
    --out-md "${GENERATED_REL}/finetune_results.md"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/08_extract_learning_curves.py" \
    --repo-root "${WORKDIR}" \
    --manifest "${MANIFEST_REL}" \
    --run-tag "${RUN_TAG}" \
    --splits "${FT_SPLITS}" \
    --out-dir "${GENERATED_REL}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/05_summarize_pf3.py" \
    --repo-root "${WORKDIR}" \
    --manifest "${MANIFEST_REL}" \
    --diag-csv "${GENERATED_REL}/pretext_diag.csv" \
    --results-csv "${GENERATED_REL}/finetune_results.csv" \
    --out-md "${GENERATED_REL}/pf3_summary.md"
fi

if [[ "${RUN_LINEAR_PROBE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${PROBE_CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/17_linear_probe_scanobjectnn.py" \
    --repo-root "${WORKDIR}" \
    --manifest-spec "${MANIFEST_REL}::${RUN_TAG}::skip_center" \
    --splits "${FT_SPLITS}" \
    --probe-epochs "${PROBE_EPOCHS}" \
    --seeds "${PROBE_SEEDS}" \
    --device "${PROBE_DEVICE}" \
    --cache-dir "${GENERATED_REL}/feature_cache" \
    --out-csv "${GENERATED_REL}/linear_probe_scanobjectnn.csv" \
    --out-md "${GENERATED_REL}/linear_probe_scanobjectnn.md"
fi

echo "[skip-center] done $(date -Is)"
