#!/usr/bin/env bash
set -euo pipefail

# Queue representation-level probes behind the heavier follow-up chain. The
# current full fine-tune runs can hide representation differences; this stage
# freezes the pretrained PointTransformer and trains only a linear classifier.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${WORKDIR}/.venv/bin/python3" ]]; then
    PYTHON_BIN="${WORKDIR}/.venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi

STAGE1_TAG="${STAGE1_TAG:-stage1_20260513_025903}"
FIXED_RANDOM_TAG="${FIXED_RANDOM_TAG:-fixedrandom_after_${STAGE1_TAG}}"
POST_WAIT_PID="${POST_WAIT_PID:-}"
POST_WAIT_PATTERN="${POST_WAIT_PATTERN:-16_run_post_stage1_followup.sh}"

OUT_TAG="${OUT_TAG:-rep_probe_after_${STAGE1_TAG}}"
OUT_DIR="${OUT_DIR:-posetnepa_mask_order_preflight/generated/${OUT_TAG}}"
LOG_DIR="${LOG_DIR:-logs/posetnepa_mask_order_preflight}"

SPLITS="${SPLITS:-hardest}"
PROBE_EPOCHS="${PROBE_EPOCHS:-200}"
PROBE_SEEDS="${PROBE_SEEDS:-0}"
PROBE_DEVICE="${PROBE_DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-64}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-2048}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STAGE1_MANIFEST="${STAGE1_MANIFEST:-posetnepa_mask_order_preflight/generated/${STAGE1_TAG}/manifest.json}"
FIXED_RANDOM_MANIFEST="${FIXED_RANDOM_MANIFEST:-posetnepa_mask_order_preflight/generated/${FIXED_RANDOM_TAG}/manifest.json}"

mkdir -p "${WORKDIR}/${OUT_DIR}" "${WORKDIR}/${LOG_DIR}"

pid_live() {
  [[ -n "${POST_WAIT_PID}" ]] || return 1
  kill -0 "${POST_WAIT_PID}" 2>/dev/null
}

pattern_live() {
  [[ -n "${POST_WAIT_PATTERN}" ]] || return 1
  pgrep -af "${POST_WAIT_PATTERN}" \
    | grep -v "$$" \
    | grep -v "18_run_rep_probe_after_post" \
    >/dev/null
}

echo "[rep-probe] start $(date -Is)"
if [[ -n "${POST_WAIT_PID}" ]]; then
  echo "[rep-probe] waiting for post-followup pid=${POST_WAIT_PID}"
  while pid_live; do
    sleep 300
    echo "[rep-probe] still waiting for pid=${POST_WAIT_PID} $(date -Is)"
  done
elif [[ -n "${POST_WAIT_PATTERN}" ]]; then
  echo "[rep-probe] waiting for post-followup pattern=${POST_WAIT_PATTERN}"
  while pattern_live; do
    sleep 300
    echo "[rep-probe] still waiting for pattern=${POST_WAIT_PATTERN} $(date -Is)"
  done
fi

if [[ ! -f "${WORKDIR}/${STAGE1_MANIFEST}" ]]; then
  echo "[rep-probe][error] missing Stage 1 manifest: ${WORKDIR}/${STAGE1_MANIFEST}" >&2
  exit 2
fi

MANIFEST_SPECS=(
  "${STAGE1_MANIFEST}::${STAGE1_TAG}::stage1"
)
if [[ -f "${WORKDIR}/${FIXED_RANDOM_MANIFEST}" ]]; then
  MANIFEST_SPECS+=("${FIXED_RANDOM_MANIFEST}::${FIXED_RANDOM_TAG}::fixed_random")
else
  echo "[rep-probe][warn] fixed_random manifest missing, skipping: ${WORKDIR}/${FIXED_RANDOM_MANIFEST}"
fi
MANIFEST_ARGS=()
for spec in "${MANIFEST_SPECS[@]}"; do
  MANIFEST_ARGS+=(--manifest-spec "${spec}")
done

echo "[rep-probe] running linear probes $(date -Is)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/17_linear_probe_scanobjectnn.py" \
  --repo-root "${WORKDIR}" \
  "${MANIFEST_ARGS[@]}" \
  --splits "${SPLITS}" \
  --probe-epochs "${PROBE_EPOCHS}" \
  --seeds "${PROBE_SEEDS}" \
  --device "${PROBE_DEVICE}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --probe-batch-size "${PROBE_BATCH_SIZE}" \
  --cache-dir "${OUT_DIR}/feature_cache" \
  --out-csv "${OUT_DIR}/linear_probe_scanobjectnn.csv" \
  --out-md "${OUT_DIR}/linear_probe_scanobjectnn.md"

echo "[rep-probe] done $(date -Is)"
