#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_udfdist_continuous_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_udfdist_continuous_independent_g2_s10000}"

submit_and_capture_job() {
  local cmd="$1"
  local out
  out="$(eval "${cmd}")"
  printf '%s\n' "${out}" >&2
  printf '%s\n' "${out}" | awk '/^\[submitted\] / {print $2}' | tail -n 1
}

train_cmd=$(
  cat <<EOF
env RUN_SET="${RUN_SET}" RUN_TAG="${TRAIN_RUN_TAG}" \
SAVE_DIR="runs/cqa/${RUN_SET}/${TRAIN_RUN_TAG}" \
LOG_ROOT="logs/cqa_pretrain/${RUN_SET}" \
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_continuous.yaml}" \
EPOCHS="${EPOCHS:-20}" MAX_STEPS="${MAX_STEPS:-10000}" SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}" \
BATCH="${BATCH:-8}" NUM_WORKERS="${NUM_WORKERS:-8}" SEED="${SEED:-0}" \
LR="${LR:-3e-4}" WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}" \
LR_SCHEDULER="${LR_SCHEDULER:-cosine}" WARMUP_RATIO="${WARMUP_RATIO:-0.05}" \
MIN_LR="${MIN_LR:-1e-6}" MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}" \
N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" \
D_MODEL="${D_MODEL:-384}" N_LAYERS="${N_LAYERS:-12}" N_HEADS="${N_HEADS:-6}" \
MLP_RATIO="${MLP_RATIO:-4.0}" DROPOUT="${DROPOUT:-0.0}" DROP_PATH="${DROP_PATH:-0.0}" \
BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}" NUM_GROUPS="${NUM_GROUPS:-64}" GROUP_SIZE="${GROUP_SIZE:-32}" \
PATCH_CENTER_MODE="${PATCH_CENTER_MODE:-fps}" PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}" \
LOCAL_ENCODER="${LOCAL_ENCODER:-pointmae_conv}" QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-6}" \
GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}" DISTANCE_FLOOR="${DISTANCE_FLOOR:-0.0}" \
EVAL_MAX_SAMPLES_PER_TASK="${EVAL_MAX_SAMPLES_PER_TASK:-128}" \
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}" \
bash scripts/pretrain/submit_pretrain_udfdist_continuous_qg.sh
EOF
)
train_job="$(submit_and_capture_job "${train_cmd}")"
if [[ -z "${train_job}" ]]; then
  echo "[error] failed to capture train job id" >&2
  exit 1
fi

CKPT_PATH="runs/cqa/${RUN_SET}/${TRAIN_RUN_TAG}/ckpt_final.pt"

offdiag_cmd=$(
  cat <<EOF
env RUN_SET="${RUN_SET}_offdiag" RUN_TAG="cqa_udfdist_continuous_offdiag_eval" \
CKPT="${CKPT_PATH}" QSUB_DEPEND="afterok:${train_job}" \
MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_continuous_pcbank.yaml}" \
SEED="${SEED:-0}" N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" MAX_SAMPLES="${MAX_SAMPLES:-256}" \
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}" EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_continuous_offdiag.sh
EOF
)
offdiag_job="$(submit_and_capture_job "${offdiag_cmd}")"
if [[ -z "${offdiag_job}" ]]; then
  echo "[error] failed to capture offdiag job id" >&2
  exit 1
fi

same_cmp_cmd=$(
  cat <<EOF
env RUN_SET="${RUN_SET}_same_completion" RUN_TAG="cqa_udfdist_continuous_same_translation_g${GRID_RES:-16}_s${MAX_SHAPES:-64}" \
MODE="same" CKPT="${CKPT_PATH}" QSUB_DEPEND="afterok:${train_job}" \
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_continuous.yaml}" \
SEED="${SEED:-0}" N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" BATCH="${CMP_BATCH:-4}" \
MAX_SHAPES="${MAX_SHAPES:-64}" GRID_RES="${GRID_RES:-16}" CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}" \
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_continuous_completion.sh
EOF
)
same_cmp_job="$(submit_and_capture_job "${same_cmp_cmd}")"
if [[ -z "${same_cmp_job}" ]]; then
  echo "[error] failed to capture same completion job id" >&2
  exit 1
fi

off_cmp_cmd=$(
  cat <<EOF
env RUN_SET="${RUN_SET}_offdiag_completion" RUN_TAG="cqa_udfdist_continuous_offdiag_translation_g${GRID_RES:-16}_s${MAX_SHAPES:-64}" \
MODE="offdiag" CKPT="${CKPT_PATH}" QSUB_DEPEND="afterok:${train_job}" \
MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_continuous_pcbank.yaml}" \
SEED="${SEED:-0}" N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" BATCH="${CMP_BATCH:-4}" \
MAX_SHAPES="${MAX_SHAPES:-64}" GRID_RES="${GRID_RES:-16}" CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}" \
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}" \
bash scripts/abci/submit_patchnepa_current_cqa_udfdist_continuous_completion.sh
EOF
)
off_cmp_job="$(submit_and_capture_job "${off_cmp_cmd}")"
if [[ -z "${off_cmp_job}" ]]; then
  echo "[error] failed to capture offdiag completion job id" >&2
  exit 1
fi

echo "[continuous] train_job=${train_job} offdiag_job=${offdiag_job} same_completion_job=${same_cmp_job} offdiag_completion_job=${off_cmp_job}"
echo "[continuous] ckpt=${CKPT_PATH}"
