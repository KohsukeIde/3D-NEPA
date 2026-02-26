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
EVAL_ROOT="${EVAL_ROOT:-runs/eval_ab_dualmask256_2proto_${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/ab_dualmask256_2proto_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/ab_dualmask256_2proto_${RUN_SET}}"

SEED_BASE="${SEED_BASE:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SCAN="${BATCH_SCAN:-96}"
BATCH_MODELNET="${BATCH_MODELNET:-128}"
EPOCHS_CLS="${EPOCHS_CLS:-100}"
LR_CLS="${LR_CLS:-1e-4}"
N_POINT_CLS="${N_POINT_CLS:-256}"
N_RAY_CLS="${N_RAY_CLS:-0}"

DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
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
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}"
PT_SAMPLE_MODE_TRAIN_CLS="${PT_SAMPLE_MODE_TRAIN_CLS:-fps}"
PT_SAMPLE_MODE_EVAL_CLS="${PT_SAMPLE_MODE_EVAL_CLS:-fps}"
PT_RFPS_M_CLS="${PT_RFPS_M_CLS:-4096}"
AUG_EVAL="${AUG_EVAL:-1}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-1}"
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
RUN_CPAC="${RUN_CPAC:-0}"
SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT:-}"
MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT:-data/modelnet40_cache_v2}"
UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}"
MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-256}"
CPAC_N_QUERY="${CPAC_N_QUERY:-256}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:--1}"

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
fi

# 256 dual-mask pretrain run metadata
PRETRAIN_RUN_SET="${PRETRAIN_RUN_SET:-rfps_aug_dm256_20260226_001557}"
CKPT_RUNA_DMOFF="${CKPT_RUNA_DMOFF:-${WORKDIR}/runs/pretrain_ab_256_rfps_aug_dualmask_${PRETRAIN_RUN_SET}_runA_dmoff/last.pt}"
CKPT_RUNA_DMON="${CKPT_RUNA_DMON:-${WORKDIR}/runs/pretrain_ab_256_rfps_aug_dualmask_${PRETRAIN_RUN_SET}_runA_dmon/last.pt}"
CKPT_RUNB_DMOFF="${CKPT_RUNB_DMOFF:-${WORKDIR}/runs/pretrain_ab_256_rfps_aug_dualmask_${PRETRAIN_RUN_SET}_runB_dmoff/last.pt}"
CKPT_RUNB_DMON="${CKPT_RUNB_DMON:-${WORKDIR}/runs/pretrain_ab_256_rfps_aug_dualmask_${PRETRAIN_RUN_SET}_runB_dmon/last.pt}"

DEP_RUNA_DMOFF="${DEP_RUNA_DMOFF:-afterok:97137.qjcm}"
DEP_RUNA_DMON="${DEP_RUNA_DMON:-afterok:97138.qjcm}"
DEP_RUNB_DMOFF="${DEP_RUNB_DMOFF:-afterok:97139.qjcm}"
DEP_RUNB_DMON="${DEP_RUNB_DMON:-afterok:97140.qjcm}"

JOB_IDS_OUT="${JOB_IDS_OUT:-}"

mkdir -p "${WORKDIR}/${LOG_ROOT}" "${WORKDIR}/${RESULTS_ROOT}" "${WORKDIR}/${EVAL_ROOT}"

if [[ -n "${JOB_IDS_OUT}" ]]; then
  mkdir -p "$(dirname "${JOB_IDS_OUT}")"
  : > "${JOB_IDS_OUT}"
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

submit() {
  local run_tag="$1"
  local ckpt="$2"
  local depend="$3"
  local seed="$4"
  local pt_xyz_key="$5"
  local ablate_point_dist="$6"
  local point_order_mode="$7"

  local out_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
  local err_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[warn] ckpt not found yet for ${run_tag}: ${ckpt} (submission continues with dependency=${depend})"
  fi

  echo "[submit] ${run_tag} ckpt=${ckpt} depend=${depend}"
  local cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "eval_${run_tag}"
    -o "${out_log}"
    -e "${err_log}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${seed},NUM_WORKERS=${NUM_WORKERS},BATCH_SCAN=${BATCH_SCAN},BATCH_MODELNET=${BATCH_MODELNET},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},PT_XYZ_KEY_CLS=${pt_xyz_key},PT_DIST_KEY_CLS=pt_dist_pool,NPROC_PER_NODE=${NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CLS_POOLING=${CLS_POOLING},POINT_ORDER_MODE=${point_order_mode},ABLATE_POINT_DIST=${ablate_point_dist},USE_FC_NORM=${USE_FC_NORM},LABEL_SMOOTHING=${LABEL_SMOOTHING},WEIGHT_DECAY_CLS=${WEIGHT_DECAY_CLS},WEIGHT_DECAY_NORM=${WEIGHT_DECAY_NORM},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},LLRD=${LLRD},LLRD_MODE=${LLRD_MODE},DROP_PATH=${DROP_PATH},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},VAL_SPLIT_MODE=${VAL_SPLIT_MODE},PT_SAMPLE_MODE_TRAIN_CLS=${PT_SAMPLE_MODE_TRAIN_CLS},PT_SAMPLE_MODE_EVAL_CLS=${PT_SAMPLE_MODE_EVAL_CLS},PT_RFPS_M_CLS=${PT_RFPS_M_CLS},AUG_EVAL=${AUG_EVAL},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST},RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT},MODELNET_CACHE_ROOT=${MODELNET_CACHE_ROOT},UNPAIRED_CACHE_ROOT=${UNPAIRED_CACHE_ROOT},SCAN_AUG_PRESET=${SCAN_AUG_PRESET},MODELNET_AUG_PRESET=${MODELNET_AUG_PRESET},MC_EVAL_K_VAL=${MC_EVAL_K_VAL},MC_EVAL_K_TEST=${MC_EVAL_K_TEST},CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN}"
    -W "depend=${depend}"
    "${SCRIPT}"
  )
  local job_id
  job_id="$("${cmd[@]}")"
  echo "[job] ${run_tag} ${job_id}"
  if [[ -n "${JOB_IDS_OUT}" ]]; then
    echo "${run_tag} ${job_id}" >> "${JOB_IDS_OUT}"
  fi
}

# A (dual-mask off/on) x {sotafair, nepafull}
submit "runA_dmoff_sotafair" "${CKPT_RUNA_DMOFF}" "${DEP_RUNA_DMOFF}" "$((SEED_BASE+0))" "pc_xyz" "1" "morton"
submit "runA_dmoff_nepafull" "${CKPT_RUNA_DMOFF}" "${DEP_RUNA_DMOFF}" "$((SEED_BASE+1))" "pt_xyz_pool" "0" "fps"
submit "runA_dmon_sotafair" "${CKPT_RUNA_DMON}" "${DEP_RUNA_DMON}" "$((SEED_BASE+2))" "pc_xyz" "1" "morton"
submit "runA_dmon_nepafull" "${CKPT_RUNA_DMON}" "${DEP_RUNA_DMON}" "$((SEED_BASE+3))" "pt_xyz_pool" "0" "fps"

# B (dual-mask off/on) x {sotafair, nepafull}
submit "runB_dmoff_sotafair" "${CKPT_RUNB_DMOFF}" "${DEP_RUNB_DMOFF}" "$((SEED_BASE+4))" "pc_xyz" "1" "morton"
submit "runB_dmoff_nepafull" "${CKPT_RUNB_DMOFF}" "${DEP_RUNB_DMOFF}" "$((SEED_BASE+5))" "pt_xyz_pool" "0" "fps"
submit "runB_dmon_sotafair" "${CKPT_RUNB_DMON}" "${DEP_RUNB_DMON}" "$((SEED_BASE+6))" "pc_xyz" "1" "morton"
submit "runB_dmon_nepafull" "${CKPT_RUNB_DMON}" "${DEP_RUNB_DMON}" "$((SEED_BASE+7))" "pt_xyz_pool" "0" "fps"

echo "[done] submitted A/B dual-mask256 eval jobs (2 proto x dmoff/dmon)"
