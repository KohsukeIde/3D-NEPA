#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/geo_teacher_distnorm_unsigned_100ep_itachi_main.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/geo_teacher_distnorm_unsigned_100ep_itachi_main.pid}"

mkdir -p "${LOG_ROOT}" "$(dirname "${SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi}")"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}

trap cleanup EXIT

echo "$$" > "${PID_FILE}"

{
  echo "[launcher] $(date '+%Y-%m-%d %H:%M:%S %Z') start local geo-teacher pretrain"
  echo "[launcher] run_tag=${RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
  echo "[launcher] save_dir=${SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi}"
  echo "[launcher] mix_config=${MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml}"
  echo "[launcher] gpus=${CUDA_VISIBLE_DEVICES:-0,1,2,3} nproc=${NPROC_PER_NODE:-4} master_port=${MASTER_PORT:-29624}"
  echo "[launcher] epochs=${EPOCHS:-100} batch=${BATCH:-64} workers=${NUM_WORKERS:-4}"
  echo "[launcher] protocol=${SAMPLING_PROTOCOL:-packed} head_mode=${HEAD_MODE:-multihead} query_if=${QUERY_INTERFACE_MODE:-no_q} loss_balance=${LOSS_BALANCE:-per_task}"
} >> "${LOG_FILE}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"${PYTHON_BIN:-python}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-4}" \
  --master_port="${MASTER_PORT:-29624}" \
  -m nepa3d.train.pretrain_primitive_answering \
  --mix_config_path "${MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml}" \
  --codec_version "${CODEC_VERSION:-cqa_v2}" \
  --save_dir "$(dirname "${SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main}")" \
  --run_name "$(basename "${SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi/geo_teacher_distnorm_unsigned_100ep_itachi_main}")" \
  --epochs "${EPOCHS:-100}" \
  --batch_size "${BATCH:-64}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --seed "${SEED:-0}" \
  --lr "${LR:-3e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --max_steps "${MAX_STEPS:--1}" \
  --lr_scheduler "${LR_SCHEDULER:-cosine}" \
  --warmup_steps "${WARMUP_STEPS:--1}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --save_every "${SAVE_EVERY:-10}" \
  --save_every_steps "${SAVE_EVERY_STEPS:-0}" \
  --n_ctx "${N_CTX:-2048}" \
  --n_qry "${N_QRY:-64}" \
  --query_order "${QUERY_ORDER:-shuffled}" \
  --d_model "${D_MODEL:-384}" \
  --n_layers "${N_LAYERS:-12}" \
  --n_heads "${N_HEADS:-6}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --dropout "${DROPOUT:-0.0}" \
  --drop_path "${DROP_PATH:-0.0}" \
  --backbone_impl "${BACKBONE_IMPL:-nepa2d}" \
  --num_groups "${NUM_GROUPS:-64}" \
  --group_size "${GROUP_SIZE:-32}" \
  --patch_center_mode "${PATCH_CENTER_MODE:-fps}" \
  --patch_fps_random_start "${PATCH_FPS_RANDOM_START:-1}" \
  --local_encoder "${LOCAL_ENCODER:-pointmae_conv}" \
  --query_type_vocab "${QUERY_TYPE_VOCAB:-0}" \
  --answer_vocab "${ANSWER_VOCAB:-0}" \
  --generator_depth "${GENERATOR_DEPTH:-2}" \
  --model_arch "${MODEL_ARCH:-prefixlm}" \
  --decoder_layers "${DECODER_LAYERS:-4}" \
  --external_backbone_ckpt "${EXTERNAL_BACKBONE_CKPT:-}" \
  --freeze_external_encoder "${FREEZE_EXTERNAL_ENCODER:-1}" \
  --external_backbone_depth "${EXTERNAL_BACKBONE_DEPTH:-12}" \
  --external_backbone_heads "${EXTERNAL_BACKBONE_HEADS:-6}" \
  --external_backbone_drop_path "${EXTERNAL_BACKBONE_DROP_PATH:-0.1}" \
  --answer_factorization "${ANSWER_FACTORIZATION:-independent}" \
  --query_interface_mode "${QUERY_INTERFACE_MODE:-no_q}" \
  --head_mode "${HEAD_MODE:-multihead}" \
  --sampling_protocol "${SAMPLING_PROTOCOL:-packed}" \
  --loss_balance "${LOSS_BALANCE:-per_task}" \
  --use_wandb "${USE_WANDB:-0}" \
  --wandb_project "${WANDB_PROJECT:-patchnepa-cqa-pretrain}" \
  --wandb_entity "${WANDB_ENTITY:-}" \
  --wandb_run_name "${WANDB_RUN_NAME:-${RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}}" \
  --wandb_group "${WANDB_GROUP:-itachi_geo_teacher}" \
  --wandb_tags "${WANDB_TAGS:-local,itachi,cqa,geo_teacher,distnorm_unsigned}" \
  --wandb_mode "${WANDB_MODE:-offline}" \
  --wandb_log_every "${WANDB_LOG_EVERY:-10}" \
  --wandb_dir "${WANDB_DIR:-${ROOT_DIR}/wandb_cqa}" \
  >> "${LOG_FILE}" 2>&1
