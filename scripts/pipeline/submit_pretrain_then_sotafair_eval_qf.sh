#!/bin/bash
set -euo pipefail

# End-to-end submitter:
#   1) submit A/B/C/D pretrain jobs
#   2) after all pretrain jobs finish successfully, submit SOTA-fair eval jobs
#      (with val split fix + LLRD/drop_path ablation matrix)
#
# Default total jobs:
#   pretrain: 4
#   eval    : 4 runs x 4 ablations = 16
#   total   : 20

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PRETRAIN_SUBMIT="${ROOT_DIR}/scripts/pretrain/submit_pretrain_abcd_qf.sh"
EVAL_ABL_SUBMIT="${ROOT_DIR}/scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh"
EVAL_ABL_VARIANT_SUBMIT="${ROOT_DIR}/scripts/eval/submit_sotafair_variants_llrd_droppath_ablation_qf.sh"

if [[ ! -x "${PRETRAIN_SUBMIT}" ]]; then
  echo "[error] missing executable: ${PRETRAIN_SUBMIT}"
  exit 1
fi
if [[ ! -x "${EVAL_ABL_SUBMIT}" ]]; then
  echo "[error] missing executable: ${EVAL_ABL_SUBMIT}"
  exit 1
fi
if [[ ! -x "${EVAL_ABL_VARIANT_SUBMIT}" ]]; then
  echo "[error] missing executable: ${EVAL_ABL_VARIANT_SUBMIT}"
  exit 1
fi

WORKDIR="${WORKDIR:-${ROOT_DIR}}"
RUN_TAG_BASE="${RUN_TAG_BASE:-$(date +%Y%m%d_%H%M%S)}"
META_DIR="${WORKDIR}/logs/pipeline/pretrain_then_eval_${RUN_TAG_BASE}"
JOB_IDS_FILE="${META_DIR}/pretrain_job_ids.txt"

mkdir -p "${META_DIR}"

echo "[stage1] submit pretrain A/B/C/D"
env \
  JOB_IDS_OUT="${JOB_IDS_FILE}" \
  WORKDIR="${WORKDIR}" \
  bash "${PRETRAIN_SUBMIT}"

if [[ ! -s "${JOB_IDS_FILE}" ]]; then
  echo "[error] pretrain job-id file is empty: ${JOB_IDS_FILE}"
  exit 2
fi

mapfile -t PRETRAIN_JOB_IDS < <(awk '{print $2}' "${JOB_IDS_FILE}")
if [[ "${#PRETRAIN_JOB_IDS[@]}" -ne 4 ]]; then
  echo "[warn] expected 4 pretrain jobs, got ${#PRETRAIN_JOB_IDS[@]}"
fi

DEPEND_EXPR="afterok:${PRETRAIN_JOB_IDS[0]}"
for ((i=1; i<${#PRETRAIN_JOB_IDS[@]}; i++)); do
  DEPEND_EXPR="${DEPEND_EXPR}:${PRETRAIN_JOB_IDS[$i]}"
done

SCAN_VARIANTS="${SCAN_VARIANTS:-}"
if [[ -n "${SCAN_VARIANTS}" ]]; then
  echo "[stage2] submit SOTA-fair eval ablations on scan variants (${SCAN_VARIANTS}) (depend=${DEPEND_EXPR})"
  env \
    WORKDIR="${WORKDIR}" \
    QSUB_DEPEND="${DEPEND_EXPR}" \
    RUN_SET_BASE_PREFIX="${RUN_TAG_BASE}_sotafair" \
    VARIANTS="${SCAN_VARIANTS}" \
    CKPT_RUNA="${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt" \
    CKPT_RUNB="${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt" \
    CKPT_RUNC="${WORKDIR}/runs/pretrain_abcd_1024_runC/last.pt" \
    CKPT_RUND="${WORKDIR}/runs/pretrain_abcd_1024_runD/last.pt" \
    VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}" \
    PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS:-pc_xyz}" \
    PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS:-pt_dist_pool}" \
    ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}" \
    N_RAY_CLS="${N_RAY_CLS:-0}" \
    POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}" \
    RUN_SCAN="${RUN_SCAN:-1}" \
    RUN_MODELNET="${RUN_MODELNET:-0}" \
    RUN_CPAC="${RUN_CPAC:-0}" \
    SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}" \
    MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}" \
    AUG_EVAL="${AUG_EVAL:-1}" \
    ABLATIONS="${ABLATIONS:-base,llrd,dp,llrd_dp}" \
    bash "${EVAL_ABL_VARIANT_SUBMIT}"
else
  echo "[stage2] submit SOTA-fair eval ablations (depend=${DEPEND_EXPR})"
  env \
    WORKDIR="${WORKDIR}" \
    QSUB_DEPEND="${DEPEND_EXPR}" \
    RUN_SET_BASE="${RUN_TAG_BASE}_sotafair" \
    CKPT_RUNA="${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt" \
    CKPT_RUNB="${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt" \
    CKPT_RUNC="${WORKDIR}/runs/pretrain_abcd_1024_runC/last.pt" \
    CKPT_RUND="${WORKDIR}/runs/pretrain_abcd_1024_runD/last.pt" \
    VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}" \
    PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS:-pc_xyz}" \
    PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS:-pt_dist_pool}" \
    ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}" \
    N_RAY_CLS="${N_RAY_CLS:-0}" \
    POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}" \
    RUN_SCAN="${RUN_SCAN:-1}" \
    RUN_MODELNET="${RUN_MODELNET:-1}" \
    RUN_CPAC="${RUN_CPAC:-0}" \
    SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}" \
    MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}" \
    AUG_EVAL="${AUG_EVAL:-1}" \
    ABLATIONS="${ABLATIONS:-base,llrd,dp,llrd_dp}" \
    bash "${EVAL_ABL_SUBMIT}"
fi

IFS=',' read -r -a _ab_arr <<< "${ABLATIONS:-base,llrd,dp,llrd_dp}"
N_PRETRAIN="${#PRETRAIN_JOB_IDS[@]}"
if [[ -n "${SCAN_VARIANTS}" ]]; then
  IFS=',' read -r -a _var_arr <<< "${SCAN_VARIANTS}"
  N_EVAL=$((4 * ${#_ab_arr[@]} * ${#_var_arr[@]}))
else
  N_EVAL=$((4 * ${#_ab_arr[@]}))
fi
N_TOTAL=$((N_PRETRAIN + N_EVAL))

echo "[summary] pretrain_jobs=${N_PRETRAIN} eval_jobs=${N_EVAL} total_jobs=${N_TOTAL}"
echo "[meta] pretrain_job_ids=${JOB_IDS_FILE}"
