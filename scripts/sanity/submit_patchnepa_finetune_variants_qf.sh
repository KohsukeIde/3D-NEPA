#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RUN_SCRIPT="${WORKDIR}/scripts/finetune/patchnepa_scanobjectnn_finetune.sh"

if [[ ! -x "${RUN_SCRIPT}" ]]; then
  echo "[error] run script not found or not executable: ${RUN_SCRIPT}"
  exit 1
fi
if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

RUN_SET="${RUN_SET:-patchnepa_ft_variants_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/patchnepa_ft/${RUN_SET}}"
mkdir -p "${LOG_ROOT}"

VARIANTS="${VARIANTS:-obj_bg,obj_only,pb_t50_rs}"   # comma-separated

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

PYTHON_BIN="${PYTHON_BIN:-${WORKDIR}/.venv/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
BATCH_MODE="${BATCH_MODE:-global}"
LR="${LR:-5e-4}"
WD="${WD:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
N_POINT="${N_POINT:-1024}"
NUM_GROUPS="${NUM_GROUPS:-64}"
GROUP_SIZE="${GROUP_SIZE:-32}"
USE_RAY_PATCH="${USE_RAY_PATCH:-1}"
N_RAY="${N_RAY:-1024}"
RAY_SAMPLE_MODE_TRAIN="${RAY_SAMPLE_MODE_TRAIN:-random}"
RAY_SAMPLE_MODE_EVAL="${RAY_SAMPLE_MODE_EVAL:-first}"
RAY_POOL_MODE="${RAY_POOL_MODE:-max}"
RAY_FUSE_MODE="${RAY_FUSE_MODE:-concat}"
RAY_HIDDEN_DIM="${RAY_HIDDEN_DIM:-128}"
RAY_MISS_T="${RAY_MISS_T:-4.0}"
RAY_HIT_THRESHOLD="${RAY_HIT_THRESHOLD:-0.5}"
PT_SAMPLE_MODE_TRAIN="${PT_SAMPLE_MODE_TRAIN:-random}"
PT_SAMPLE_MODE_EVAL="${PT_SAMPLE_MODE_EVAL:-fps}"
POOLING="${POOLING:-cls_max}"
POS_MODE="${POS_MODE:-center_mlp}"
HEAD_MODE="${HEAD_MODE:-pointmae_mlp}"
HEAD_HIDDEN_DIM="${HEAD_HIDDEN_DIM:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.5}"
INIT_MODE="${INIT_MODE:-default}"
IS_CAUSAL="${IS_CAUSAL:-0}"
PATCHNEPA_FT_MODE="${PATCHNEPA_FT_MODE:-qa_zeroa}"  # qa_zeroa | q_only
AUG_PRESET="${AUG_PRESET:-pointmae}"
AUG_EVAL="${AUG_EVAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"
SAVE_DIR="${SAVE_DIR:-runs/sanity/patchnepa_ft}"
CKPT="${CKPT:-}"

if [[ -z "${CKPT}" ]]; then
  echo "[error] CKPT is required for patchnepa finetune submission"
  exit 2
fi

cache_root_for_variant() {
  case "$1" in
    obj_bg) echo "data/scanobjectnn_obj_bg_v3_nonorm" ;;
    obj_only) echo "data/scanobjectnn_obj_only_v3_nonorm" ;;
    pb_t50_rs) echo "data/scanobjectnn_pb_t50_rs_v3_nonorm" ;;
    *)
      echo "[error] unknown variant: $1" >&2
      return 1
      ;;
  esac
}

IFS=',' read -r -a _variants <<< "${VARIANTS}"

job_ids_file="${LOG_ROOT}/job_ids.txt"
: > "${job_ids_file}"

for variant in "${_variants[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  [[ -z "${variant}" ]] && continue
  cache_root="$(cache_root_for_variant "${variant}")"

  run_tag="${variant}_${RUN_SET}"
  run_name="patchnepa_${variant}_${RUN_SET}"
  out="${LOG_ROOT}/${variant}.out"
  err="${LOG_ROOT}/${variant}.err"
  job_name="pn_${variant}"

  qvars=(
    "WORKDIR=${WORKDIR}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DATA_FORMAT=npz"
    "CACHE_ROOT=${cache_root}"
    "SCAN_VARIANT=${variant}"
    "RUN_NAME=${run_name}"
    "SAVE_DIR=${SAVE_DIR}"
    "CKPT=${CKPT}"
    "EPOCHS=${EPOCHS}"
    "BATCH=${BATCH}"
    "BATCH_MODE=${BATCH_MODE}"
    "LR=${LR}"
    "WD=${WD}"
    "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
    "N_POINT=${N_POINT}"
    "NUM_GROUPS=${NUM_GROUPS}"
    "GROUP_SIZE=${GROUP_SIZE}"
    "MODEL_SOURCE=patchnepa"
    "USE_RAY_PATCH=${USE_RAY_PATCH}"
    "N_RAY=${N_RAY}"
    "RAY_SAMPLE_MODE_TRAIN=${RAY_SAMPLE_MODE_TRAIN}"
    "RAY_SAMPLE_MODE_EVAL=${RAY_SAMPLE_MODE_EVAL}"
    "RAY_POOL_MODE=${RAY_POOL_MODE}"
    "RAY_FUSE_MODE=${RAY_FUSE_MODE}"
    "RAY_HIDDEN_DIM=${RAY_HIDDEN_DIM}"
    "RAY_MISS_T=${RAY_MISS_T}"
    "RAY_HIT_THRESHOLD=${RAY_HIT_THRESHOLD}"
    "BACKBONE_MODE=nepa2d"
    "PT_SAMPLE_MODE_TRAIN=${PT_SAMPLE_MODE_TRAIN}"
    "PT_SAMPLE_MODE_EVAL=${PT_SAMPLE_MODE_EVAL}"
    "NUM_WORKERS=${NUM_WORKERS}"
    "VAL_RATIO=${VAL_RATIO}"
    "VAL_SEED=${VAL_SEED}"
    "VAL_SPLIT_MODE=${VAL_SPLIT_MODE}"
    "SEED=${SEED}"
    "NPROC_PER_NODE=${NPROC_PER_NODE}"
    "POOLING=${POOLING}"
    "POS_MODE=${POS_MODE}"
    "HEAD_MODE=${HEAD_MODE}"
    "HEAD_HIDDEN_DIM=${HEAD_HIDDEN_DIM}"
    "HEAD_DROPOUT=${HEAD_DROPOUT}"
    "INIT_MODE=${INIT_MODE}"
    "IS_CAUSAL=${IS_CAUSAL}"
    "PATCHNEPA_FT_MODE=${PATCHNEPA_FT_MODE}"
    "AUG_PRESET=${AUG_PRESET}"
    "AUG_EVAL=${AUG_EVAL}"
    "MC_EVAL_K_TEST=${MC_EVAL_K_TEST}"
    "ALLOW_SCAN_UNISCALE_V2=${ALLOW_SCAN_UNISCALE_V2}"
  )
  QVARS="$(IFS=,; echo "${qvars[*]}")"

  cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "${job_name}"
    -o "${out}"
    -e "${err}"
    -v "${QVARS}"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${RUN_SCRIPT}" )

  jid="$("${cmd[@]}")"
  echo "${jid} ${variant} ${run_tag}" | tee -a "${job_ids_file}"
  echo "[submitted] ${jid} variant=${variant}"
done

echo "[done] run_set=${RUN_SET}"
echo "[logs] ${LOG_ROOT}"
echo "[job_ids] ${job_ids_file}"
