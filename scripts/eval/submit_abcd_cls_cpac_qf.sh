#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_eval_cls_cpac_qf.sh"

DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
WALLTIME="${WALLTIME:-24:00:00}"
RT_QF="${RT_QF:-1}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-runs/eval_abcd_1024_${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/abcd_1024_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/abcd_cls_cpac_${RUN_SET}}"
SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SCAN="${BATCH_SCAN:-96}"
BATCH_MODELNET="${BATCH_MODELNET:-128}"
EPOCHS_CLS="${EPOCHS_CLS:-100}"
LR_CLS="${LR_CLS:-1e-4}"
N_POINT_CLS="${N_POINT_CLS:-1024}"
N_RAY_CLS="${N_RAY_CLS:-0}"
PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS:-pc_xyz}"
PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS:-pt_dist_pool}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
ALLOW_SCAN_DIST="${ALLOW_SCAN_DIST:-0}"
USE_FC_NORM="${USE_FC_NORM:-0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
WEIGHT_DECAY_CLS="${WEIGHT_DECAY_CLS:-0.05}"
WEIGHT_DECAY_NORM="${WEIGHT_DECAY_NORM:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
LLRD="${LLRD:-1.0}"
LLRD_MODE="${LLRD_MODE:-exp}"
DROP_PATH="${DROP_PATH:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-official_auto}"
PT_SAMPLE_MODE_TRAIN_CLS="${PT_SAMPLE_MODE_TRAIN_CLS:-fps}"
PT_SAMPLE_MODE_EVAL_CLS="${PT_SAMPLE_MODE_EVAL_CLS:-fps}"
PT_RFPS_M_CLS="${PT_RFPS_M_CLS:-4096}"
AUG_EVAL="${AUG_EVAL:-1}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-1}"
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
RUN_CPAC="${RUN_CPAC:-1}"
ALLOW_SCAN_UNISCALE_V2="${ALLOW_SCAN_UNISCALE_V2:-0}"
SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT:-}"
MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT:-data/modelnet40_cache_v2}"
UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}"
MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
MC_TTA_GUARD_MODE="${MC_TTA_GUARD_MODE:-error}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:--1}"
QSUB_DEPEND="${QSUB_DEPEND:-}"
CKPT_RUNA="${CKPT_RUNA:-${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt}"
CKPT_RUNB="${CKPT_RUNB:-${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt}"
CKPT_RUNC="${CKPT_RUNC:-${WORKDIR}/runs/pretrain_abcd_1024_runC/last.pt}"
CKPT_RUND="${CKPT_RUND:-${WORKDIR}/runs/pretrain_abcd_1024_runD/last.pt}"
RUN_IDS="${RUN_IDS:-A,B,C,D}"

if [[ "${RUN_SCAN}" == "1" ]]; then
  if [[ -z "${SCAN_CACHE_ROOT}" ]]; then
    echo "[error] SCAN_CACHE_ROOT is required when RUN_SCAN=1."
    echo "        Use variant cache roots (obj_bg/obj_only/pb_t50_rs)."
    exit 2
  fi
  if [[ "${SCAN_CACHE_ROOT}" == *"scanobjectnn_main_split_v2"* ]]; then
    echo "[error] SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT} is disallowed (main_split deprecated)."
    exit 2
  fi
  if [[ "${SCAN_CACHE_ROOT}" == *"scanobjectnn_"*"_v2" ]] && [[ "${ALLOW_SCAN_UNISCALE_V2}" != "1" ]]; then
    echo "[error] SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT} is a uniscale v2 cache and is disallowed by policy."
    echo "        Use scanobjectnn_*_v3_nonorm variant caches, or set ALLOW_SCAN_UNISCALE_V2=1 for intentional legacy reruns."
    exit 2
  fi
  if [[ "${ABLATE_POINT_DIST}" != "1" ]] && [[ "${ALLOW_SCAN_DIST}" != "1" ]]; then
    echo "[error] ScanObjectNN classification with dist enabled is blocked by default."
    echo "        Use ABLATE_POINT_DIST=1 (recommended), or set ALLOW_SCAN_DIST=1 for intentional dist-on runs."
    exit 2
  fi
fi

mkdir -p "${WORKDIR}/${LOG_ROOT}" "${WORKDIR}/${RESULTS_ROOT}" "${WORKDIR}/${EVAL_ROOT}"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

submit() {
  local run_tag="$1"
  local ckpt="$2"
  local out_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
  local err_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"

  if [[ ! -f "${ckpt}" ]]; then
    if [[ -n "${QSUB_DEPEND}" ]]; then
      echo "[warn] ckpt not found yet for ${run_tag}: ${ckpt} (allowed because QSUB_DEPEND is set)"
    else
      echo "[error] missing ckpt for ${run_tag}: ${ckpt}"
      exit 2
    fi
  fi

  echo "[submit] ${run_tag} ckpt=${ckpt}"
  cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "eval_${run_tag}"
    -o "${out_log}"
    -e "${err_log}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS},BATCH_SCAN=${BATCH_SCAN},BATCH_MODELNET=${BATCH_MODELNET},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},PT_XYZ_KEY_CLS=${PT_XYZ_KEY_CLS},PT_DIST_KEY_CLS=${PT_DIST_KEY_CLS},NPROC_PER_NODE=${NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CLS_POOLING=${CLS_POOLING},POINT_ORDER_MODE=${POINT_ORDER_MODE},ABLATE_POINT_DIST=${ABLATE_POINT_DIST},ALLOW_SCAN_DIST=${ALLOW_SCAN_DIST},ALLOW_SCAN_UNISCALE_V2=${ALLOW_SCAN_UNISCALE_V2},USE_FC_NORM=${USE_FC_NORM},LABEL_SMOOTHING=${LABEL_SMOOTHING},WEIGHT_DECAY_CLS=${WEIGHT_DECAY_CLS},WEIGHT_DECAY_NORM=${WEIGHT_DECAY_NORM},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},LLRD=${LLRD},LLRD_MODE=${LLRD_MODE},DROP_PATH=${DROP_PATH},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},VAL_SPLIT_MODE=${VAL_SPLIT_MODE},PT_SAMPLE_MODE_TRAIN_CLS=${PT_SAMPLE_MODE_TRAIN_CLS},PT_SAMPLE_MODE_EVAL_CLS=${PT_SAMPLE_MODE_EVAL_CLS},PT_RFPS_M_CLS=${PT_RFPS_M_CLS},AUG_EVAL=${AUG_EVAL},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST},RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT},MODELNET_CACHE_ROOT=${MODELNET_CACHE_ROOT},UNPAIRED_CACHE_ROOT=${UNPAIRED_CACHE_ROOT},SCAN_AUG_PRESET=${SCAN_AUG_PRESET},MODELNET_AUG_PRESET=${MODELNET_AUG_PRESET},MC_EVAL_K_VAL=${MC_EVAL_K_VAL},MC_EVAL_K_TEST=${MC_EVAL_K_TEST},MC_TTA_GUARD_MODE=${MC_TTA_GUARD_MODE},CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN}"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

IFS=',' read -r -a _run_arr <<< "${RUN_IDS}"
_n_submit=0
for _rid in "${_run_arr[@]}"; do
  _rid="$(echo "${_rid}" | xargs | tr '[:lower:]' '[:upper:]')"
  case "${_rid}" in
    A)
      submit "runA" "${CKPT_RUNA}"
      _n_submit=$((_n_submit + 1))
      ;;
    B)
      submit "runB" "${CKPT_RUNB}"
      _n_submit=$((_n_submit + 1))
      ;;
    C)
      submit "runC" "${CKPT_RUNC}"
      _n_submit=$((_n_submit + 1))
      ;;
    D)
      submit "runD" "${CKPT_RUND}"
      _n_submit=$((_n_submit + 1))
      ;;
    "")
      ;;
    *)
      echo "[error] unknown RUN_IDS entry: ${_rid} (use comma-separated A,B,C,D)"
      exit 3
      ;;
  esac
done

if [[ "${_n_submit}" -le 0 ]]; then
  echo "[error] no runs selected by RUN_IDS=${RUN_IDS}"
  exit 4
fi

echo "[done] submitted ${_n_submit} classification+CPAC jobs (RUN_IDS=${RUN_IDS})"
