#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/routeb}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/routeb.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/routeb.pid}"

mkdir -p "${LOG_ROOT}" \
  "${ROOT_DIR}/results/cqa_eval_itachi" \
  "${ROOT_DIR}/results/cqa_completion_itachi" \
  "${ROOT_DIR}/results/cqa_probe_itachi" \
  "$(dirname "${ROUTEB_SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi/routeb}")"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}
trap cleanup EXIT

echo "$$" > "${PID_FILE}"

geopcp_require_python
geopcp_require_gpu
geopcp_require_compiled_backends
[[ -f "${PRETRAIN_CKPT}" ]] || geopcp_die "missing PRETRAIN_CKPT=${PRETRAIN_CKPT}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

log "start Route-B chain arm=${ARM_TAG} pretrain_ckpt=${PRETRAIN_CKPT}"

cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${ROUTEB_GPU:-0}"

if [[ ! -f "${ROUTEB_SAVE_DIR}/ckpt_final.pt" ]]; then
  log "train external_pointmae CQA readout run=${ROUTEB_TRAIN_RUN_TAG}"
  "${PYTHON_BIN}" -m nepa3d.train.pretrain_primitive_answering \
    --mix_config_path "${ROUTEB_TRAIN_MIX_CONFIG}" \
    --save_dir "$(dirname "${ROUTEB_SAVE_DIR}")" \
    --run_name "$(basename "${ROUTEB_SAVE_DIR}")" \
    --epochs "${ROUTEB_EPOCHS:-100}" \
    --batch_size "${ROUTEB_BATCH:-32}" \
    --num_workers "${ROUTEB_NUM_WORKERS:-8}" \
    --seed "${ROUTEB_SEED:-0}" \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --max_steps "${ROUTEB_MAX_STEPS:-10000}" \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --min_lr 1e-6 \
    --max_grad_norm 1.0 \
    --save_every 10 \
    --save_every_steps 0 \
    --n_ctx "${ROUTEB_N_CTX:-2048}" \
    --n_qry "${ROUTEB_N_QRY:-64}" \
    --query_order shuffled \
    --d_model 384 \
    --n_layers 12 \
    --n_heads 6 \
    --mlp_ratio 4.0 \
    --dropout 0.0 \
    --drop_path 0.0 \
    --backbone_impl nepa2d \
    --num_groups 64 \
    --group_size 32 \
    --patch_center_mode fps \
    --patch_fps_random_start 1 \
    --local_encoder pointmae_conv \
    --generator_depth 2 \
    --model_arch external_pointmae \
    --decoder_layers 4 \
    --external_backbone_ckpt "${PRETRAIN_CKPT}" \
    --freeze_external_encoder 1 \
    --external_backbone_depth 12 \
    --external_backbone_heads 6 \
    --external_backbone_drop_path 0.1 \
    --answer_factorization independent \
    --query_interface_mode no_q \
    --head_mode shared \
    --sampling_protocol packed \
    --loss_balance flat \
    --use_wandb 0 \
    >> "${LOG_FILE}" 2>&1
fi

ROUTEB_CKPT="${ROUTEB_SAVE_DIR}/ckpt_final.pt"
[[ -f "${ROUTEB_CKPT}" ]] || geopcp_die "Route-B CQA ckpt missing: ${ROUTEB_CKPT}"

log "run same-context controls"
WORKDIR="${ROOT_DIR}" VENV_ACTIVATE="" CKPT="${ROUTEB_CKPT}" RUN_TAG="${ARM_TAG}__routeb_same" \
MIX_CONFIG="${SAME_MIX_CONFIG}" LOG_ROOT="${LOG_ROOT}/eval" \
OUT_JSON="${ROOT_DIR}/results/cqa_eval_itachi/${ARM_TAG}__routeb_same.json" \
DEVICE="${ROUTEB_DEVICE:-cuda}" BATCH="${ROUTEB_BATCH:-32}" NUM_WORKERS="${ROUTEB_NUM_WORKERS:-8}" \
SEED="${ROUTEB_SEED:-0}" N_CTX="${ROUTEB_N_CTX:-2048}" N_QRY="${ROUTEB_N_QRY:-64}" \
MAX_SAMPLES="${ROUTEB_MAX_SAMPLES:-512}" SPLIT_OVERRIDE="test" TASK_FILTER="udf_distance" \
EVAL_SAMPLE_MODE="random" QUERY_ORDER="sampled" \
bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_offdiag_udfdist_qg.sh" >> "${LOG_FILE}" 2>&1

log "run offdiag controls"
WORKDIR="${ROOT_DIR}" VENV_ACTIVATE="" CKPT="${ROUTEB_CKPT}" RUN_TAG="${ARM_TAG}__routeb_offdiag" \
MIX_CONFIG="${OFFDIAG_MIX_CONFIG}" LOG_ROOT="${LOG_ROOT}/eval" \
OUT_JSON="${ROOT_DIR}/results/cqa_eval_itachi/${ARM_TAG}__routeb_offdiag.json" \
DEVICE="${ROUTEB_DEVICE:-cuda}" BATCH="${ROUTEB_BATCH:-32}" NUM_WORKERS="${ROUTEB_NUM_WORKERS:-8}" \
SEED="${ROUTEB_SEED:-0}" N_CTX="${ROUTEB_N_CTX:-2048}" N_QRY="${ROUTEB_N_QRY:-64}" \
MAX_SAMPLES="${ROUTEB_MAX_SAMPLES:-512}" SPLIT_OVERRIDE="test" TASK_FILTER="udf_distance" \
EVAL_SAMPLE_MODE="random" QUERY_ORDER="sampled" \
bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_offdiag_udfdist_qg.sh" >> "${LOG_FILE}" 2>&1

log "run completion same"
WORKDIR="${ROOT_DIR}" VENV_ACTIVATE="" CKPT="${ROUTEB_CKPT}" RUN_TAG="${ARM_TAG}__routeb_completion_same" \
MIX_CONFIG="${SAME_MIX_CONFIG}" LOG_ROOT="${LOG_ROOT}/completion" \
OUT_JSON="${ROOT_DIR}/results/cqa_completion_itachi/${ARM_TAG}__routeb_completion_same.json" \
DEVICE="${ROUTEB_DEVICE:-cuda}" BATCH="4" MAX_SHAPES="${ROUTEB_COMPLETION_MAX_SHAPES:-16}" \
SPLIT_OVERRIDE="test" TASK_FILTER="udf_distance" EVAL_SAMPLE_MODE="random" \
N_CTX="${ROUTEB_N_CTX:-2048}" N_QRY="${ROUTEB_N_QRY:-64}" GRID_RES="12" CHUNK_N_QUERY="64" \
TAU_LIST="0.01,0.02,0.05" \
bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_udfdist_completion_qg.sh" >> "${LOG_FILE}" 2>&1

log "run completion offdiag"
WORKDIR="${ROOT_DIR}" VENV_ACTIVATE="" CKPT="${ROUTEB_CKPT}" RUN_TAG="${ARM_TAG}__routeb_completion_offdiag" \
MIX_CONFIG="${OFFDIAG_MIX_CONFIG}" LOG_ROOT="${LOG_ROOT}/completion" \
OUT_JSON="${ROOT_DIR}/results/cqa_completion_itachi/${ARM_TAG}__routeb_completion_offdiag.json" \
DEVICE="${ROUTEB_DEVICE:-cuda}" BATCH="4" MAX_SHAPES="${ROUTEB_COMPLETION_MAX_SHAPES:-16}" \
SPLIT_OVERRIDE="test" TASK_FILTER="udf_distance" EVAL_SAMPLE_MODE="random" \
N_CTX="${ROUTEB_N_CTX:-2048}" N_QRY="${ROUTEB_N_QRY:-64}" GRID_RES="12" CHUNK_N_QUERY="64" \
TAU_LIST="0.01,0.02,0.05" \
bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_udfdist_completion_qg.sh" >> "${LOG_FILE}" 2>&1

log "run curvature probe"
WORKDIR="${ROOT_DIR}" VENV_ACTIVATE="" CKPT="${ROUTEB_CKPT}" \
CACHE_ROOT="data/shapenet_cache_v2_20260401_worldvis" PROBE_TARGET="curvature" \
RUN_TAG="${ARM_TAG}__routeb_curvature_probe" SAVE_DIR="${ROOT_DIR}/runs/cqa_probe_itachi" \
OUT_JSON="${ROOT_DIR}/results/cqa_probe_itachi/${ARM_TAG}__routeb_curvature_probe.json" \
LOG_ROOT="${LOG_ROOT}/probe" TRAIN_SPLIT="train" EVAL_SPLIT="test" \
MAX_STEPS="${ROUTEB_PROBE_MAX_STEPS:-5000}" EVAL_EVERY="500" BATCH="8" NUM_WORKERS="4" \
SEED="${ROUTEB_SEED:-0}" N_CTX="${ROUTEB_N_CTX:-2048}" N_QRY="${ROUTEB_N_QRY:-64}" \
MAX_TRAIN_SAMPLES="0" MAX_EVAL_SAMPLES="128" EVAL_SAMPLE_MODE="random" \
CONTROLS="correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query" \
bash "${ROOT_DIR}/scripts/eval/nepa3d_cqa_geo_probe_qg.sh" >> "${LOG_FILE}" 2>&1

log "Route-B chain done arm=${ARM_TAG}"
