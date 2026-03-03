#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${WORKDIR}" || exit 1

PRETRAIN_SUBMIT="${WORKDIR}/scripts/pretrain/submit_pretrain_patch_nepa_tokens_qf.sh"
CPAC_JOB_SCRIPT="${WORKDIR}/scripts/analysis/nepa3d_cpac_udf_patchnepa_qf.sh"

if [[ ! -x "${PRETRAIN_SUBMIT}" ]]; then
  echo "[error] missing pretrain submit script: ${PRETRAIN_SUBMIT}"
  exit 1
fi
if [[ ! -x "${CPAC_JOB_SCRIPT}" ]]; then
  echo "[error] missing cpac job script: ${CPAC_JOB_SCRIPT}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-patchnepa_tokens_screen_${TS}}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/patchnepa_tokens_screen/${RUN_ROOT}}"
mkdir -p "${LOG_DIR}"

GROUP_LIST="${GROUP_LIST:-qgah50055}"
PT_RT_QF="${PT_RT_QF:-1}"
PT_WALLTIME="${PT_WALLTIME:-12:00:00}"
CPAC_RT_QF="${CPAC_RT_QF:-1}"
CPAC_WALLTIME="${CPAC_WALLTIME:-06:00:00}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_tokens.yaml}"
CPAC_DATA_ROOT="${CPAC_DATA_ROOT:?set CPAC_DATA_ROOT=...}"
TOKEN_QA_LAYOUTS="${TOKEN_QA_LAYOUTS:-interleave,split_sep}"

PT_MAX_STEPS="${PT_MAX_STEPS:-10000}"
PT_BATCH="${PT_BATCH:-8}"
PT_N_SURF="${PT_N_SURF:-2048}"
PT_N_QRY="${PT_N_QRY:-1024}"
PT_PM_PC_NORM="${PT_PM_PC_NORM:-1}"
PT_PM_SCALE_TRANSLATE="${PT_PM_SCALE_TRANSLATE:-1}"
PT_PM_TRANSFORM_ANSWERS="${PT_PM_TRANSFORM_ANSWERS:-1}"
PT_LOSS_TARGET_MODE="${PT_LOSS_TARGET_MODE:-content_tokens}"
PT_PATCH_LOCAL_ENCODER="${PT_PATCH_LOCAL_ENCODER:-pointmae_conv}"

CPAC_HEAD_TRAIN_SPLIT="${CPAC_HEAD_TRAIN_SPLIT:-train_udf}"
CPAC_EVAL_SPLIT="${CPAC_EVAL_SPLIT:-eval}"
CPAC_N_CTX="${CPAC_N_CTX:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_CHUNK_N_QUERY="${CPAC_CHUNK_N_QUERY:-1024}"
CPAC_MAX_TRAIN_SHAPES="${CPAC_MAX_TRAIN_SHAPES:-64}"
CPAC_MAX_EVAL_SHAPES="${CPAC_MAX_EVAL_SHAPES:-64}"
CPAC_REP_SOURCE="${CPAC_REP_SOURCE:-h}"
CPAC_RIDGE_ALPHA="${CPAC_RIDGE_ALPHA:-1.0}"
CPAC_TAU="${CPAC_TAU:-0.01}"
CPAC_SEED="${CPAC_SEED:-0}"

RESULTS_TSV="${LOG_DIR}/submitted_jobs.tsv"
echo -e "layout\tpretrain_job\tpretrain_runset\tcpac_job\tcpac_tag\tckpt\tcpac_data_root" > "${RESULTS_TSV}"

IFS=',' read -r -a LAYOUT_ARR <<< "${TOKEN_QA_LAYOUTS}"
if [[ "${#LAYOUT_ARR[@]}" -eq 0 ]]; then
  echo "[error] TOKEN_QA_LAYOUTS is empty"
  exit 2
fi

for raw_layout in "${LAYOUT_ARR[@]}"; do
  layout="$(echo "${raw_layout}" | xargs)"
  [[ -z "${layout}" ]] && continue

  ltag="${layout}"
  ltag="${ltag//[^a-zA-Z0-9_]/_}"
  pre_run_set="${RUN_ROOT}_${ltag}"
  pre_run_tag="pt_${pre_run_set}"
  pre_job_name="ptok_${ltag}"
  ckpt="runs/patchnepa_tokens/${pre_run_set}/ckpt_final.pt"

  echo "[submit][pretrain] layout=${layout} run_set=${pre_run_set}"
  pre_out="$(env \
    WORKDIR="${WORKDIR}" \
    GROUP_LIST="${GROUP_LIST}" \
    RT_QF="${PT_RT_QF}" \
    WALLTIME="${PT_WALLTIME}" \
    RUN_SET="${pre_run_set}" \
    RUN_TAG="${pre_run_tag}" \
    JOB_NAME="${pre_job_name}" \
    MIX_CONFIG="${MIX_CONFIG}" \
    MAX_STEPS="${PT_MAX_STEPS}" \
    BATCH="${PT_BATCH}" \
    N_SURF="${PT_N_SURF}" \
    N_QRY="${PT_N_QRY}" \
    TOKEN_QA_LAYOUT="${layout}" \
    PM_PC_NORM="${PT_PM_PC_NORM}" \
    PM_SCALE_TRANSLATE="${PT_PM_SCALE_TRANSLATE}" \
    PM_TRANSFORM_ANSWERS="${PT_PM_TRANSFORM_ANSWERS}" \
    LOSS_TARGET_MODE="${PT_LOSS_TARGET_MODE}" \
    PATCH_LOCAL_ENCODER="${PT_PATCH_LOCAL_ENCODER}" \
    bash "${PRETRAIN_SUBMIT}")"
  echo "${pre_out}" | tee "${LOG_DIR}/${ltag}.pretrain.submit.log"

  pre_jid="$(echo "${pre_out}" | awk '/\[submitted\]/{print $2; exit}')"
  if [[ -z "${pre_jid}" ]]; then
    echo "[error] failed to parse pretrain job id for layout=${layout}"
    exit 3
  fi

  cpac_tag="${RUN_ROOT}_cpac_${ltag}"
  cpac_job_name="cpok_${ltag}"
  cpac_pbs_log="${LOG_DIR}/${ltag}.cpac.pbs.log"
  cpac_vars="WORKDIR=${WORKDIR},RUN_TAG=${cpac_tag},CKPT=${ckpt},DATA_ROOT=${CPAC_DATA_ROOT},LOG_ROOT=${WORKDIR}/logs/patch_nepa_cpac/${RUN_ROOT},RESULTS_ROOT=${WORKDIR}/results,HEAD_TRAIN_SPLIT=${CPAC_HEAD_TRAIN_SPLIT},EVAL_SPLIT=${CPAC_EVAL_SPLIT},MINI_CPAC=1,N_CTX_POINTS=${CPAC_N_CTX},N_QUERY=${CPAC_N_QUERY},CHUNK_N_QUERY=${CPAC_CHUNK_N_QUERY},MAX_TRAIN_SHAPES=${CPAC_MAX_TRAIN_SHAPES},MAX_EVAL_SHAPES=${CPAC_MAX_EVAL_SHAPES},REP_SOURCE=${CPAC_REP_SOURCE},RIDGE_ALPHA=${CPAC_RIDGE_ALPHA},TAU=${CPAC_TAU},SEED=${CPAC_SEED}"

  echo "[submit][mini-cpac] layout=${layout} depend=afterok:${pre_jid}"
  cpac_jid="$(qsub \
    -l "rt_QF=${CPAC_RT_QF}" \
    -l "walltime=${CPAC_WALLTIME}" \
    -W "group_list=${GROUP_LIST}" \
    -W "depend=afterok:${pre_jid}" \
    -N "${cpac_job_name}" \
    -o "${cpac_pbs_log}" \
    -v "${cpac_vars}" \
    "${CPAC_JOB_SCRIPT}")"
  echo "[submitted] ${cpac_jid}" | tee "${LOG_DIR}/${ltag}.cpac.submit.log"

  echo -e "${layout}\t${pre_jid}\t${pre_run_set}\t${cpac_jid}\t${cpac_tag}\t${ckpt}\t${CPAC_DATA_ROOT}" >> "${RESULTS_TSV}"
done

echo "[done] ${RESULTS_TSV}"
cat "${RESULTS_TSV}"
