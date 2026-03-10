#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}" || exit 1

SCRIPT="${ROOT_DIR}/scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh"
DEFAULT_CKPT="runs/patchnepa_tokens/patchnepa_recong2_full300_20260306_072643/pt_pc33mesh33udf33_reconch_g2_e300/ckpt_final.pt"

CKPT="${CKPT:-${DEFAULT_CKPT}}"
DATA_ROOT="${DATA_ROOT:-data/shapenet_unpaired_cache_v2_pc33_mesh33_udf33}"
RUN_TAG="${RUN_TAG:-patchnepa_abci_cpac_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/patch_nepa_cpac}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results/patch_nepa_cpac}"
PBS_LOG_DIR="${PBS_LOG_DIR:-${LOG_ROOT}}"
PBS_LOG_PATH="${PBS_LOG_PATH:-${PBS_LOG_DIR}/${RUN_TAG}.pbs.log}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-06:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
JOB_NAME="${JOB_NAME:-pntok_cpac}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

HEAD_TRAIN_SPLIT="${HEAD_TRAIN_SPLIT:-train_udf}"
EVAL_SPLIT="${EVAL_SPLIT:-eval}"
REP_SOURCE="${REP_SOURCE:-h}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
TAU="${TAU:-0.01}"
SEED="${SEED:-0}"
ANSWER_IN_DIM="${ANSWER_IN_DIM:-0}"
MINI_CPAC="${MINI_CPAC:-1}"
MAX_TRAIN_SHAPES="${MAX_TRAIN_SHAPES:-64}"
MAX_EVAL_SHAPES="${MAX_EVAL_SHAPES:-64}"
N_CTX_POINTS="${N_CTX_POINTS:-1024}"
N_QUERY="${N_QUERY:-1024}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-1024}"
CONTEXT_PRIMITIVE="${CONTEXT_PRIMITIVE:-pc}"
QUERY_PRIMITIVE="${QUERY_PRIMITIVE:-udf}"
SURF_XYZ_KEY="${SURF_XYZ_KEY:-pc_xyz}"
QRY_XYZ_KEY="${QRY_XYZ_KEY:-udf_qry_xyz}"
QRY_DIST_KEY="${QRY_DIST_KEY:-udf_qry_dist}"
OUT_JSON="${OUT_JSON:-${RESULTS_ROOT}/${RUN_TAG}.json}"

mkdir -p "${PBS_LOG_DIR}" "${RESULTS_ROOT}" "${LOG_ROOT}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] CKPT not found: ${CKPT}"
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[error] DATA_ROOT not found: ${DATA_ROOT}"
  exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found in PATH"
  exit 1
fi

qvars=(
  "WORKDIR=${ROOT_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "CKPT=${CKPT}"
  "DATA_ROOT=${DATA_ROOT}"
  "LOG_ROOT=${LOG_ROOT}"
  "RESULTS_ROOT=${RESULTS_ROOT}"
  "HEAD_TRAIN_SPLIT=${HEAD_TRAIN_SPLIT}"
  "EVAL_SPLIT=${EVAL_SPLIT}"
  "REP_SOURCE=${REP_SOURCE}"
  "RIDGE_ALPHA=${RIDGE_ALPHA}"
  "TAU=${TAU}"
  "SEED=${SEED}"
  "ANSWER_IN_DIM=${ANSWER_IN_DIM}"
  "MINI_CPAC=${MINI_CPAC}"
  "MAX_TRAIN_SHAPES=${MAX_TRAIN_SHAPES}"
  "MAX_EVAL_SHAPES=${MAX_EVAL_SHAPES}"
  "N_CTX_POINTS=${N_CTX_POINTS}"
  "N_QUERY=${N_QUERY}"
  "CHUNK_N_QUERY=${CHUNK_N_QUERY}"
  "CONTEXT_PRIMITIVE=${CONTEXT_PRIMITIVE}"
  "QUERY_PRIMITIVE=${QUERY_PRIMITIVE}"
  "SURF_XYZ_KEY=${SURF_XYZ_KEY}"
  "QRY_XYZ_KEY=${QRY_XYZ_KEY}"
  "QRY_DIST_KEY=${QRY_DIST_KEY}"
  "OUT_JSON=${OUT_JSON}"
)
QVARS="$(IFS=,; echo "${qvars[*]}")"

cmd=(
  qsub
  -l "rt_QF=${RT_QF}"
  -l "walltime=${WALLTIME}"
  -W "group_list=${GROUP_LIST}"
  -N "${JOB_NAME}"
  -o "${PBS_LOG_PATH}"
  -v "${QVARS}"
)
if [[ -n "${QSUB_DEPEND}" ]]; then
  cmd+=( -W "depend=${QSUB_DEPEND}" )
fi
cmd+=( "${SCRIPT}" )

echo "[abci-cpac] ckpt=${CKPT}"
echo "[abci-cpac] data_root=${DATA_ROOT}"
echo "[abci-cpac] out_json=${OUT_JSON}"

job_id="$("${cmd[@]}")"
echo "[submitted] ${job_id}"
echo "[pbs_log] ${PBS_LOG_PATH}"
