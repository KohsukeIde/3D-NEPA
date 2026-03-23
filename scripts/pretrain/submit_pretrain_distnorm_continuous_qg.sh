#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_pretrain_distnorm_continuous_qg.sh"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
RUN_SET="${RUN_SET:-cqa_distnorm_continuous_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-${RUN_SET}}"
SAVE_DIR="${SAVE_DIR:-runs/cqa/${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/cqa_pretrain/${RUN_SET}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${WORKDIR}/${LOG_ROOT}}"
mkdir -p "${PBS_LOG_DIR}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"
mkdir -p "${ENV_DIR}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "ERROR: qsub not found in PATH"
  exit 1
fi

write_env_file() {
  local path="$1"
  shift
  : > "${path}"
  for kv in "$@"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    printf '%s=%q\n' "${key}" "${val}" >> "${path}"
  done
}

qvars=(
  "WORKDIR=${WORKDIR}"
  "RUN_TAG=${RUN_TAG}"
  "SAVE_DIR=${SAVE_DIR}"
  "LOG_ROOT=${LOG_ROOT}"
  "MIX_CONFIG=${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_dist_norm_continuous.yaml}"
  "EPOCHS=${EPOCHS:-20}"
  "SAVE_EVERY=${SAVE_EVERY:-5}"
  "SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-500}"
  "BATCH=${BATCH:-8}"
  "NUM_WORKERS=${NUM_WORKERS:-8}"
  "SEED=${SEED:-0}"
  "DEVICE=${DEVICE:-cuda}"
  "LR=${LR:-3e-4}"
  "WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}"
  "MAX_STEPS=${MAX_STEPS:-10000}"
  "LR_SCHEDULER=${LR_SCHEDULER:-cosine}"
  "WARMUP_STEPS=${WARMUP_STEPS:--1}"
  "WARMUP_RATIO=${WARMUP_RATIO:-0.05}"
  "MIN_LR=${MIN_LR:-1e-6}"
  "MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}"
  "N_CTX=${N_CTX:-2048}"
  "N_QRY=${N_QRY:-64}"
  "QUERY_ORDER=${QUERY_ORDER:-shuffled}"
  "D_MODEL=${D_MODEL:-384}"
  "N_LAYERS=${N_LAYERS:-12}"
  "N_HEADS=${N_HEADS:-6}"
  "MLP_RATIO=${MLP_RATIO:-4.0}"
  "DROPOUT=${DROPOUT:-0.0}"
  "DROP_PATH=${DROP_PATH:-0.0}"
  "BACKBONE_IMPL=${BACKBONE_IMPL:-nepa2d}"
  "NUM_GROUPS=${NUM_GROUPS:-64}"
  "GROUP_SIZE=${GROUP_SIZE:-32}"
  "PATCH_CENTER_MODE=${PATCH_CENTER_MODE:-fps}"
  "PATCH_FPS_RANDOM_START=${PATCH_FPS_RANDOM_START:-1}"
  "LOCAL_ENCODER=${LOCAL_ENCODER:-pointmae_conv}"
  "QUERY_TYPE_VOCAB=${QUERY_TYPE_VOCAB:-6}"
  "GENERATOR_DEPTH=${GENERATOR_DEPTH:-2}"
  "DISTANCE_FLOOR=${DISTANCE_FLOOR:-0.0}"
)
ENV_FILE="${ENV_FILE:-${ENV_DIR}/${RUN_TAG}.env}"
write_env_file "${ENV_FILE}" "${qvars[@]}"

cmd=(
  qsub
  -l "rt_QG=${RT_QG}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME:-cqa_dnc}"
  -o "${PBS_LOG_PATH}"
  -v "WORKDIR=${WORKDIR},ENV_FILE=${ENV_FILE}"
)
if [[ -n "${QSUB_DEPEND:-}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[save_dir] ${SAVE_DIR}"
echo "[log_root] ${LOG_ROOT}"
echo "[pbs_log] ${PBS_LOG_PATH}"
echo "[env_file] ${ENV_FILE}"
