#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SPLIT_SCRIPT="${SCRIPT_DIR}/make_shapenet_unpaired_split.sh"
MAT_SCRIPT="${SCRIPT_DIR}/preprocess_shapenet_unpaired.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"

RUN_TAG="${RUN_TAG:-shapenet_v2_post_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/preprocess/shapenet_unpaired/${RUN_TAG}}"
mkdir -p "${LOG_DIR}"

# Required: IDs of all preprocess shard jobs.
# Accepts either comma or whitespace separated list.
PREPROC_JOB_IDS="${PREPROC_JOB_IDS:-}"
if [[ -z "${PREPROC_JOB_IDS}" ]]; then
  echo "[error] PREPROC_JOB_IDS is required. e.g. \"101714.qjcm,101715.qjcm,...\""
  exit 1
fi

# split settings
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v2_20260303}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
OUT_JSON="${OUT_JSON:-data/shapenet_unpaired_splits_v2_20260303.json}"
SEED="${SEED:-0}"
RATIOS="${RATIOS:-0.34:0.33:0.33}"

# materialize settings
SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-${CACHE_ROOT}}"
SPLIT_JSON="${SPLIT_JSON:-${OUT_JSON}}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_unpaired_cache_v2_20260303}"
SPLITS="${SPLITS:-train_mesh:train_pc:train_udf:eval}"
LINK_MODE="${LINK_MODE:-symlink}"
OVERWRITE="${OVERWRITE:-0}"

# resources
RT_QF_SPLIT="${RT_QF_SPLIT:-1}"
WALLTIME_SPLIT="${WALLTIME_SPLIT:-02:00:00}"
RT_QF_MAT="${RT_QF_MAT:-1}"
WALLTIME_MAT="${WALLTIME_MAT:-04:00:00}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

_dep_ids="$(echo "${PREPROC_JOB_IDS}" | tr ',' ' ' | xargs)"
_dep_expr="$(echo "${_dep_ids}" | tr ' ' ':')"
if [[ -z "${_dep_expr}" ]]; then
  echo "[error] failed to parse PREPROC_JOB_IDS=${PREPROC_JOB_IDS}"
  exit 1
fi
DEPEND_PREPROC="afterok:${_dep_expr}"

split_vars=(
  "WORKDIR=${WORKDIR}"
  "CACHE_ROOT=${CACHE_ROOT}"
  "TRAIN_SPLIT=${TRAIN_SPLIT}"
  "EVAL_SPLIT=${EVAL_SPLIT}"
  "OUT_JSON=${OUT_JSON}"
  "SEED=${SEED}"
  "RATIOS=${RATIOS}"
)
SPLIT_QVARS="$(IFS=,; echo "${split_vars[*]}")"

split_job_name="${JOB_NAME_SPLIT:-shpv2_split}"
split_cmd=(
  qsub
  -l "rt_QF=${RT_QF_SPLIT}"
  -l "walltime=${WALLTIME_SPLIT}"
  -W "group_list=${GROUP_LIST}"
  -W "depend=${DEPEND_PREPROC}"
  -N "${split_job_name}"
  -o "${LOG_DIR}/${split_job_name}.out"
  -e "${LOG_DIR}/${split_job_name}.err"
  -v "${SPLIT_QVARS}"
  "${SPLIT_SCRIPT}"
)
split_jid="$("${split_cmd[@]}")"

mat_vars=(
  "WORKDIR=${WORKDIR}"
  "SRC_CACHE_ROOT=${SRC_CACHE_ROOT}"
  "SPLIT_JSON=${SPLIT_JSON}"
  "OUT_ROOT=${OUT_ROOT}"
  "SPLITS=${SPLITS}"
  "LINK_MODE=${LINK_MODE}"
  "OVERWRITE=${OVERWRITE}"
)
MAT_QVARS="$(IFS=,; echo "${mat_vars[*]}")"

mat_job_name="${JOB_NAME_MAT:-shpv2_mat}"
mat_cmd=(
  qsub
  -l "rt_QF=${RT_QF_MAT}"
  -l "walltime=${WALLTIME_MAT}"
  -W "group_list=${GROUP_LIST}"
  -W "depend=afterok:${split_jid}"
  -N "${mat_job_name}"
  -o "${LOG_DIR}/${mat_job_name}.out"
  -e "${LOG_DIR}/${mat_job_name}.err"
  -v "${MAT_QVARS}"
  "${MAT_SCRIPT}"
)
mat_jid="$("${mat_cmd[@]}")"

echo "[submitted] split job: ${split_jid} (depend=${DEPEND_PREPROC})"
echo "[submitted] materialize job: ${mat_jid} (depend=afterok:${split_jid})"
echo "[log_dir] ${LOG_DIR}"
{
  echo "PREPROC_JOB_IDS=${_dep_ids}"
  echo "DEPEND_PREPROC=${DEPEND_PREPROC}"
  echo "SPLIT_JOB=${split_jid}"
  echo "MAT_JOB=${mat_jid}"
  echo "CACHE_ROOT=${CACHE_ROOT}"
  echo "OUT_JSON=${OUT_JSON}"
  echo "OUT_ROOT=${OUT_ROOT}"
} > "${LOG_DIR}/submitted_jobs.txt"
