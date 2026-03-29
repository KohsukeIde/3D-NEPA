#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/analysis/nepa3d_cqa_udfdist_degraded_qg.sh"
DEFAULT_CKPT="${ROOT_DIR}/runs/cqa/patchnepa_cqa_v2_distnorm_unsigned_prefixlm_20260325_234124/cqa_v2_distnorm_unsigned_prefixlm_independent_g2_s10000/ckpt_final.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
RUN_SET="${RUN_SET:-patchnepa_cqa_v2_distnorm_unsigned_prefixlm_degraded_$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-cqa_v2_distnorm_unsigned_prefixlm_degraded_suite}"
SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist_pcbank.yaml}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/cqa_degraded/${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/cqa_degraded/${RUN_SET}}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"
ENV_DIR="${ENV_DIR:-${PBS_LOG_DIR}}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
RT_QG="${RT_QG:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-cqa_deg}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
BATCH="${BATCH:-4}"
MAX_SHAPES="${MAX_SHAPES:-16}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
QUERY_ORDER="${QUERY_ORDER:-sampled}"
GRID_RES="${GRID_RES:-16}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-64}"
TAU_LIST="${TAU_LIST:-0.01,0.02,0.05}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"
MC_LEVEL="${MC_LEVEL:-0.05}"
DROPOUT_KEEP_LIST="${DROPOUT_KEEP_LIST:-0.50,0.25,0.10,0.05}"
GAUSSIAN_SIGMA_LIST="${GAUSSIAN_SIGMA_LIST:-0.01,0.02,0.05}"
EXPORT_ASSETS="${EXPORT_ASSETS:-1}"
MAX_ASSET_SHAPES="${MAX_ASSET_SHAPES:-8}"
OUT_JSON="${OUT_JSON:-${RESULTS_ROOT}/${RUN_TAG}.json}"
OUT_CSV="${OUT_CSV:-${RESULTS_ROOT}/${RUN_TAG}.csv}"
OUT_MD="${OUT_MD:-${RESULTS_ROOT}/${RUN_TAG}.md}"
ASSETS_ROOT="${ASSETS_ROOT:-${RESULTS_ROOT}/${RUN_TAG}_assets}"

mkdir -p "${PBS_LOG_DIR}" "${RESULTS_ROOT}" "${ENV_DIR}" "${ASSETS_ROOT}"

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
  "WORKDIR=${ROOT_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "CKPT=${CKPT}"
  "SAME_MIX_CONFIG=${SAME_MIX_CONFIG}"
  "OFFDIAG_MIX_CONFIG=${OFFDIAG_MIX_CONFIG}"
  "LOG_ROOT=${LOG_ROOT}"
  "OUT_JSON=${OUT_JSON}"
  "OUT_CSV=${OUT_CSV}"
  "OUT_MD=${OUT_MD}"
  "ASSETS_ROOT=${ASSETS_ROOT}"
  "DEVICE=${DEVICE}"
  "SEED=${SEED}"
  "N_CTX=${N_CTX}"
  "N_QRY=${N_QRY}"
  "BATCH=${BATCH}"
  "MAX_SHAPES=${MAX_SHAPES}"
  "SPLIT_OVERRIDE=${SPLIT_OVERRIDE}"
  "EVAL_SAMPLE_MODE=${EVAL_SAMPLE_MODE}"
  "QUERY_ORDER=${QUERY_ORDER}"
  "GRID_RES=${GRID_RES}"
  "CHUNK_N_QUERY=${CHUNK_N_QUERY}"
  "TAU_LIST=${TAU_LIST}"
  "MESH_NUM_SAMPLES=${MESH_NUM_SAMPLES}"
  "MC_LEVEL=${MC_LEVEL}"
  "DROPOUT_KEEP_LIST=${DROPOUT_KEEP_LIST}"
  "GAUSSIAN_SIGMA_LIST=${GAUSSIAN_SIGMA_LIST}"
  "EXPORT_ASSETS=${EXPORT_ASSETS}"
  "MAX_ASSET_SHAPES=${MAX_ASSET_SHAPES}"
)
ENV_FILE="${ENV_FILE:-${ENV_DIR}/${RUN_TAG}.env}"
write_env_file "${ENV_FILE}" "${qvars[@]}"

cmd=(
  qsub
  -l "rt_QG=${RT_QG}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${PBS_LOG_PATH}"
  -v "WORKDIR=${ROOT_DIR},ENV_FILE=${ENV_FILE}"
)
if [[ -n "${QSUB_DEPEND:-}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[run_set] ${RUN_SET}"
echo "[run_tag] ${RUN_TAG}"
echo "[pbs_log] ${PBS_LOG_PATH}"
echo "[env_file] ${ENV_FILE}"
echo "[out_json] ${OUT_JSON}"
