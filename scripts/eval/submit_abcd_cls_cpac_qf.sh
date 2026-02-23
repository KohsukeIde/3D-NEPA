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
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:--1}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

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
    echo "[error] missing ckpt for ${run_tag}: ${ckpt}"
    exit 2
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
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS},BATCH_SCAN=${BATCH_SCAN},BATCH_MODELNET=${BATCH_MODELNET},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},NPROC_PER_NODE=${NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CLS_POOLING=${CLS_POOLING},POINT_ORDER_MODE=${POINT_ORDER_MODE},ABLATE_POINT_DIST=${ABLATE_POINT_DIST},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN}"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

submit "runA" "${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt"
submit "runB" "${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt"
submit "runC" "${WORKDIR}/runs/pretrain_abcd_1024_runC/last.pt"
submit "runD" "${WORKDIR}/runs/pretrain_abcd_1024_runD/last.pt"

echo "[done] submitted A/B/C/D classification+CPAC jobs"
