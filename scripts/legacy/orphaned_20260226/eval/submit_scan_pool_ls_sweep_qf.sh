#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_SINGLE="${SCRIPT_DIR}/nepa3d_eval_cls_cpac_qf.sh"
SCRIPT_MULTI="${SCRIPT_DIR}/nepa3d_eval_cls_cpac_multinode_pbsdsh.sh"

DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
WALLTIME="${WALLTIME:-24:00:00}"
NODES_PER_RUN="${NODES_PER_RUN:-1}"
RT_QF="${RT_QF:-${NODES_PER_RUN}}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)_scan_pool_ls}"

CKPT="${CKPT:-${WORKDIR}/runs/pretrain_abcd_1024_fix20260222_200311_runA/last.pt}"
EVAL_ROOT="${EVAL_ROOT:-runs/eval_scan_pool_ls_${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/scan_pool_ls_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/scan_pool_ls_${RUN_SET}}"

SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SCAN="${BATCH_SCAN:-96}"
EPOCHS_CLS="${EPOCHS_CLS:-120}"
LR_CLS="${LR_CLS:-3e-4}"
N_POINT_CLS="${N_POINT_CLS:-1024}"
N_RAY_CLS="${N_RAY_CLS:-0}"
PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS:-pc_xyz}"
PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS:-pt_dist_pool}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-none}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
USE_FC_NORM="${USE_FC_NORM:-1}"
WEIGHT_DECAY_CLS="${WEIGHT_DECAY_CLS:-0.05}"
WEIGHT_DECAY_NORM="${WEIGHT_DECAY_NORM:-0.0}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

mkdir -p "${WORKDIR}/${LOG_ROOT}" "${WORKDIR}/${EVAL_ROOT}" "${WORKDIR}/${RESULTS_ROOT}"

SCRIPT="${SCRIPT_SINGLE}"
if [[ "${NODES_PER_RUN}" -gt 1 ]]; then
  SCRIPT="${SCRIPT_MULTI}"
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing ckpt: ${CKPT}"
  exit 2
fi

submit_one() {
  local run_tag="$1"
  local cls_pooling="$2"
  local label_smoothing="$3"
  local out_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
  local err_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"

  echo "[submit] ${run_tag} cls_pooling=${cls_pooling} label_smoothing=${label_smoothing}"
  cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "scanpls_${run_tag}"
    -o "${out_log}"
    -e "${err_log}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${CKPT},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS},BATCH_SCAN=${BATCH_SCAN},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},PT_XYZ_KEY_CLS=${PT_XYZ_KEY_CLS},PT_DIST_KEY_CLS=${PT_DIST_KEY_CLS},NPROC_PER_NODE=${NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CLS_POOLING=${cls_pooling},POINT_ORDER_MODE=${POINT_ORDER_MODE},ABLATE_POINT_DIST=${ABLATE_POINT_DIST},USE_FC_NORM=${USE_FC_NORM},LABEL_SMOOTHING=${label_smoothing},WEIGHT_DECAY_CLS=${WEIGHT_DECAY_CLS},WEIGHT_DECAY_NORM=${WEIGHT_DECAY_NORM},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},MIXED_PRECISION=${MIXED_PRECISION},MC_EVAL_K_VAL=${MC_EVAL_K_VAL},MC_EVAL_K_TEST=${MC_EVAL_K_TEST},SCAN_AUG_PRESET=${SCAN_AUG_PRESET},RUN_SCAN=1,RUN_MODELNET=0,RUN_CPAC=0"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

submit_one "run_q_ls00" "mean_q" "0.0"
submit_one "run_q_ls01" "mean_q" "0.1"
submit_one "run_a_ls00" "mean_a" "0.0"
submit_one "run_a_ls01" "mean_a" "0.1"

echo "[done] submitted pooling/label_smoothing sweep (4 runs)"
echo "log_root=${WORKDIR}/${LOG_ROOT}"
echo "eval_root=${WORKDIR}/${EVAL_ROOT}"
echo "nodes_per_run=${NODES_PER_RUN} (rt_QF=${RT_QF})"
