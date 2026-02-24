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
RUN_SET="${RUN_SET:-$(date +%Y%m%d_%H%M%S)}"

# RunA checkpoint is used for minimal 2x2 diagnosis:
#  - point_order_mode: fps vs morton
#  - scan augmentation: none vs scanobjectnn(rotate_z)
CKPT="${CKPT:-${WORKDIR}/runs/pretrain_abcd_1024_fix20260222_200311_runA/last.pt}"

EVAL_ROOT="${EVAL_ROOT:-runs/eval_scan_ablation_aug_order_${RUN_SET}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/scan_ablation_aug_order_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/scan_ablation_aug_order_${RUN_SET}}"

SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SCAN="${BATCH_SCAN:-96}"
EPOCHS_CLS="${EPOCHS_CLS:-300}"
LR_CLS="${LR_CLS:-3e-4}"
N_POINT_CLS="${N_POINT_CLS:-1024}"
N_RAY_CLS="${N_RAY_CLS:-0}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"
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
  local point_order_mode="$2"
  local scan_aug_preset="$3"
  local out_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
  local err_log="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"

  echo "[submit] ${run_tag} point_order_mode=${point_order_mode} scan_aug=${scan_aug_preset}"
  cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -W "group_list=${GROUP_LIST}"
    -N "scanab_${run_tag}"
    -o "${out_log}"
    -e "${err_log}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${CKPT},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS},BATCH_SCAN=${BATCH_SCAN},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},NPROC_PER_NODE=${NPROC_PER_NODE},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CLS_POOLING=${CLS_POOLING},POINT_ORDER_MODE=${point_order_mode},ABLATE_POINT_DIST=${ABLATE_POINT_DIST},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},MIXED_PRECISION=${MIXED_PRECISION},MC_EVAL_K_VAL=${MC_EVAL_K_VAL},MC_EVAL_K_TEST=${MC_EVAL_K_TEST},SCAN_AUG_PRESET=${scan_aug_preset},RUN_SCAN=1,RUN_MODELNET=0,RUN_CPAC=0"
  )
  if [[ -n "${QSUB_DEPEND}" ]]; then
    cmd+=( -W "depend=${QSUB_DEPEND}" )
  fi
  cmd+=( "${SCRIPT}" )
  "${cmd[@]}"
}

submit_one "runA_fps_augnone" "fps" "none"
submit_one "runA_fps_augscan" "fps" "scanobjectnn"
submit_one "runA_morton_augnone" "morton" "none"
submit_one "runA_morton_augscan" "morton" "scanobjectnn"

echo "[done] submitted 2x2 scan ablation jobs"
echo "log_root=${WORKDIR}/${LOG_ROOT}"
echo "eval_root=${WORKDIR}/${EVAL_ROOT}"
echo "nodes_per_run=${NODES_PER_RUN} (rt_QF=${RT_QF})"
