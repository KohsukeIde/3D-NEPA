#!/usr/bin/env bash
set -euo pipefail

# Wait for an existing Stage 1 run, refresh diagnostics, then launch the
# diagnostic follow-up chain. This intentionally avoids the broad Stage 2/3
# sweep and focuses on controls needed to interpret Stage 1.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
STAGE1_TAG="${STAGE1_TAG:-stage1_20260513_025903}"
STAGE1_DIR="${STAGE1_DIR:-posetnepa_mask_order_preflight/generated/${STAGE1_TAG}}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-${STAGE1_DIR}/manifest.json}"
FT_SPLITS="${FT_SPLITS:-hardest}"
USE_WANDB="${USE_WANDB:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${WORKDIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${WORKDIR}/.venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

RUN_FIXED_RANDOM="${RUN_FIXED_RANDOM:-1}"
RUN_MISMATCH="${RUN_MISMATCH:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-0}"
RUN_LINEAR_PROBE="${RUN_LINEAR_PROBE:-0}"

FIXED_RANDOM_TAG="${FIXED_RANDOM_TAG:-fixedrandom_after_${STAGE1_TAG}}"
MISMATCH_TAG="${MISMATCH_TAG:-order_mismatch_${STAGE1_TAG}_maskoff}"
SCRATCH_TAG="${SCRATCH_TAG:-scratch_early_${STAGE1_TAG}_maskoff}"
LINEAR_PROBE_TAG="${LINEAR_PROBE_TAG:-rep_probe_after_${STAGE1_TAG}}"

stage1_live() {
  pgrep -af "${STAGE1_TAG}" \
    | grep -E 'torchrun|main\.py|pointgpt_(train|finetune)_local_ddp|02_pretrain_mask_order_matrix|03_finetune_scan_pb_t50' \
    | grep -v "$$" \
    >/dev/null
}

echo "[post] start $(date -Is)"
echo "[post] waiting for Stage 1 tag: ${STAGE1_TAG}"
while stage1_live; do
  sleep 300
  echo "[post] still waiting $(date -Is)"
done

if [[ ! -f "${WORKDIR}/${STAGE1_MANIFEST}" ]]; then
  echo "[post][error] Stage 1 manifest missing: ${WORKDIR}/${STAGE1_MANIFEST}" >&2
  exit 2
fi

echo "[post] refreshing Stage 1 summaries $(date -Is)"
"${PYTHON_BIN}" "${SCRIPT_DIR}/04_extract_pretext_diag.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${STAGE1_MANIFEST}" \
  --run-tag "${STAGE1_TAG}" \
  --out-csv "${STAGE1_DIR}/pretext_diag.csv" \
  --out-md "${STAGE1_DIR}/pretext_diag.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/04b_extract_finetune_results.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${STAGE1_MANIFEST}" \
  --run-tag "${STAGE1_TAG}" \
  --splits "${FT_SPLITS}" \
  --out-csv "${STAGE1_DIR}/finetune_results.csv" \
  --out-md "${STAGE1_DIR}/finetune_results.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/08_extract_learning_curves.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${STAGE1_MANIFEST}" \
  --run-tag "${STAGE1_TAG}" \
  --splits "${FT_SPLITS}" \
  --out-dir "${STAGE1_DIR}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/09_extract_copy_baseline_diag.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${STAGE1_MANIFEST}" \
  --run-tag "${STAGE1_TAG}" \
  --final-epoch-only \
  --tail-batches 0 \
  --out-csv "${STAGE1_DIR}/copy_baseline_diag.csv" \
  --out-md "${STAGE1_DIR}/copy_baseline_diag.md"
"${PYTHON_BIN}" "${SCRIPT_DIR}/05_summarize_pf3.py" \
  --repo-root "${WORKDIR}" \
  --manifest "${STAGE1_MANIFEST}" \
  --diag-csv "${STAGE1_DIR}/pretext_diag.csv" \
  --results-csv "${STAGE1_DIR}/finetune_results.csv" \
  --out-md "${STAGE1_DIR}/pf3_summary.md"

if [[ "${RUN_FIXED_RANDOM}" == "1" ]]; then
  echo "[post] running fixed_random mini-control: ${FIXED_RANDOM_TAG} $(date -Is)"
  RUN_TAG="${FIXED_RANDOM_TAG}" \
  GENERATED_REL="posetnepa_mask_order_preflight/generated/${FIXED_RANDOM_TAG}" \
  CONFIG_OUT_DIR="PointGPT/cfgs/PointGPT-S/poset_mask_order_preflight/${FIXED_RANDOM_TAG}" \
  ORDERS="fixed_random" \
  MASKS="0.0,0.7" \
  MAX_EPOCH="30" \
  FT_MAX_EPOCH="50" \
  FT_SPLITS="${FT_SPLITS}" \
  USE_WANDB="${USE_WANDB}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  bash "${SCRIPT_DIR}/run_stage1_local.sh"
fi

if [[ "${RUN_MISMATCH}" == "1" ]]; then
  echo "[post] running order mismatch follow-up: ${MISMATCH_TAG} $(date -Is)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/10_make_order_mismatch_configs.py" \
    --repo-root "${WORKDIR}" \
    --stage1-manifest "${STAGE1_MANIFEST}" \
    --stage1-run-tag "${STAGE1_TAG}" \
    --source-mask 0.0 \
    --splits "${FT_SPLITS}" \
    --ft-max-epoch 50 \
    --run-tag "${MISMATCH_TAG}"
  DRY_RUN=0 \
  ONLY_MISMATCH=1 \
  MANIFEST="${WORKDIR}/posetnepa_mask_order_preflight/generated/${MISMATCH_TAG}/order_mismatch_manifest.json" \
  USE_WANDB="${USE_WANDB}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  bash "${SCRIPT_DIR}/11_run_order_mismatch_finetune.sh"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/15_extract_followup_finetune_results.py" \
    --repo-root "${WORKDIR}" \
    --manifest "posetnepa_mask_order_preflight/generated/${MISMATCH_TAG}/order_mismatch_manifest.json"
fi

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  echo "[post] running scratch/early follow-up: ${SCRATCH_TAG} $(date -Is)"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/13_make_scratch_early_configs.py" \
    --repo-root "${WORKDIR}" \
    --stage1-manifest "${STAGE1_MANIFEST}" \
    --stage1-run-tag "${STAGE1_TAG}" \
    --orders "simplified_morton,diffusion_shell" \
    --masks 0.0 \
    --splits "${FT_SPLITS}" \
    --early-epochs "1,5,10" \
    --ft-max-epoch 50 \
    --run-tag "${SCRATCH_TAG}"
  DRY_RUN=0 \
  MANIFEST="${WORKDIR}/posetnepa_mask_order_preflight/generated/${SCRATCH_TAG}/scratch_early_manifest.json" \
  USE_WANDB="${USE_WANDB}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  bash "${SCRIPT_DIR}/14_run_scratch_early_comparison.sh"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/15_extract_followup_finetune_results.py" \
    --repo-root "${WORKDIR}" \
    --manifest "posetnepa_mask_order_preflight/generated/${SCRATCH_TAG}/scratch_early_manifest.json"
fi

if [[ "${RUN_LINEAR_PROBE}" == "1" ]]; then
  echo "[post] running frozen linear probe: ${LINEAR_PROBE_TAG} $(date -Is)"
  LINEAR_PROBE_OUT_DIR="posetnepa_mask_order_preflight/generated/${LINEAR_PROBE_TAG}"
  LINEAR_MANIFEST_ARGS=(
    --manifest-spec "${STAGE1_MANIFEST}::${STAGE1_TAG}::stage1"
  )
  if [[ -f "${WORKDIR}/posetnepa_mask_order_preflight/generated/${FIXED_RANDOM_TAG}/manifest.json" ]]; then
    LINEAR_MANIFEST_ARGS+=(
      --manifest-spec "posetnepa_mask_order_preflight/generated/${FIXED_RANDOM_TAG}/manifest.json::${FIXED_RANDOM_TAG}::fixed_random"
    )
  fi
  CUDA_VISIBLE_DEVICES="${LINEAR_PROBE_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES%%,*}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/17_linear_probe_scanobjectnn.py" \
    --repo-root "${WORKDIR}" \
    "${LINEAR_MANIFEST_ARGS[@]}" \
    --splits "${LINEAR_PROBE_SPLITS:-${FT_SPLITS}}" \
    --probe-epochs "${LINEAR_PROBE_EPOCHS:-200}" \
    --seeds "${LINEAR_PROBE_SEEDS:-0}" \
    --device "${LINEAR_PROBE_DEVICE:-cuda}" \
    --feature-batch-size "${LINEAR_PROBE_FEATURE_BATCH_SIZE:-64}" \
    --probe-batch-size "${LINEAR_PROBE_BATCH_SIZE:-2048}" \
    --cache-dir "${LINEAR_PROBE_OUT_DIR}/feature_cache" \
    --out-csv "${LINEAR_PROBE_OUT_DIR}/linear_probe_scanobjectnn.csv" \
    --out-md "${LINEAR_PROBE_OUT_DIR}/linear_probe_scanobjectnn.md"
fi

echo "[post] done $(date -Is)"
