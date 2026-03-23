#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_SET="${RUN_SET:-patchnepa_cqa_distnorm_continuous_${STAMP}}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-cqa_distnorm_continuous_independent_g2_s10000}"

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
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_continuous.yaml}" \
EPOCHS="${EPOCHS:-20}" MAX_STEPS="${MAX_STEPS:-10000}" SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-500}" \
BATCH="${BATCH:-8}" NUM_WORKERS="${NUM_WORKERS:-8}" SEED="${SEED:-0}" \
LR="${LR:-3e-4}" WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}" \
LR_SCHEDULER="${LR_SCHEDULER:-cosine}" WARMUP_RATIO="${WARMUP_RATIO:-0.05}" \
MIN_LR="${MIN_LR:-1e-6}" MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}" \
N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" QUERY_ORDER="${QUERY_ORDER:-shuffled}" \
D_MODEL="${D_MODEL:-384}" N_LAYERS="${N_LAYERS:-12}" N_HEADS="${N_HEADS:-6}" \
MLP_RATIO="${MLP_RATIO:-4.0}" DROPOUT="${DROPOUT:-0.0}" DROP_PATH="${DROP_PATH:-0.0}" \
BACKBONE_IMPL="${BACKBONE_IMPL:-nepa2d}" NUM_GROUPS="${NUM_GROUPS:-64}" GROUP_SIZE="${GROUP_SIZE:-32}" \
PATCH_CENTER_MODE="${PATCH_CENTER_MODE:-fps}" PATCH_FPS_RANDOM_START="${PATCH_FPS_RANDOM_START:-1}" \
LOCAL_ENCODER="${LOCAL_ENCODER:-pointmae_conv}" QUERY_TYPE_VOCAB="${QUERY_TYPE_VOCAB:-6}" \
GENERATOR_DEPTH="${GENERATOR_DEPTH:-2}" DISTANCE_FLOOR="${DISTANCE_FLOOR:-0.0}" \
bash scripts/pretrain/submit_pretrain_distnorm_continuous_qg.sh
EOF
)
train_job="$(submit_and_capture_job "${train_cmd}")"
if [[ -z "${train_job}" ]]; then
  echo "[error] failed to capture train job id" >&2
  exit 1
fi

CKPT_PATH="runs/cqa/${RUN_SET}/${TRAIN_RUN_TAG}/ckpt_final.pt"

suite_cmd=$(
  cat <<EOF
env RUN_SET="${RUN_SET}_suite" RUN_TAG="cqa_distnorm_continuous_suite" \
CKPT="${CKPT_PATH}" QSUB_DEPEND="afterok:${train_job}" \
SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_continuous.yaml}" \
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_continuous_pcbank_eval.yaml}" \
SEED="${SEED:-0}" N_CTX="${N_CTX:-2048}" N_QRY="${N_QRY:-64}" MAX_SAMPLES="${MAX_SAMPLES:-256}" \
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}" TASKS="${TASKS:-udf_distance,mesh_normal}" \
CONTROLS="${CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}" \
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}" QUERY_ORDER="${EVAL_QUERY_ORDER:-sampled}" TAU="${TAU:-0.05}" \
bash scripts/abci/submit_patchnepa_current_cqa_multitype_continuous_suite.sh
EOF
)
suite_job="$(submit_and_capture_job "${suite_cmd}")"
if [[ -z "${suite_job}" ]]; then
  echo "[error] failed to capture suite job id" >&2
  exit 1
fi

echo "[distnorm-continuous] train_job=${train_job} suite_job=${suite_job}"
echo "[distnorm-continuous] ckpt=${CKPT_PATH}"
