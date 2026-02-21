#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="${SCRIPT_DIR}/nepa3d_eval_cls_cpac_qf.sh"

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
WALLTIME="${WALLTIME:-24:00:00}"
RT_QF="${RT_QF:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

mkdir -p "${WORKDIR}/logs/eval/abcd_cls_cpac"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

submit() {
  local run_tag="$1"
  local ckpt="$2"
  local out_log="${WORKDIR}/logs/eval/abcd_cls_cpac/${run_tag}.out"
  local err_log="${WORKDIR}/logs/eval/abcd_cls_cpac/${run_tag}.err"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[error] missing ckpt for ${run_tag}: ${ckpt}"
    exit 2
  fi

  echo "[submit] ${run_tag} ckpt=${ckpt}"
  cmd=(
    qsub
    -l "rt_QF=${RT_QF}"
    -l "walltime=${WALLTIME}"
    -N "eval_${run_tag}"
    -o "${out_log}"
    -e "${err_log}"
    -v "WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},NPROC_PER_NODE=${NPROC_PER_NODE}"
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
