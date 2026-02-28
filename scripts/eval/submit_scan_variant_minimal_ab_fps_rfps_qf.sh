#!/bin/bash
set -euo pipefail

# Minimal protocol-correct ScanObjectNN rerun:
#   families: fps + rfps pretrain checkpoints
#   runs: A/B only
#   variants: obj_bg,obj_only,pb_t50_rs
#   ablations: base only (default)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_VARIANTS="${SCRIPT_DIR}/submit_sotafair_variants_llrd_droppath_ablation_qf.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -x "${SUBMIT_VARIANTS}" ]]; then
  echo "[error] missing executable: ${SUBMIT_VARIANTS}"
  exit 1
fi

RUN_SET_BASE="${RUN_SET_BASE:-variant_min_ab_fpsrfps_$(date +%Y%m%d_%H%M%S)}"
FAMILIES="${FAMILIES:-fps,rfps}"
VARIANTS="${VARIANTS:-obj_bg,obj_only,pb_t50_rs}"
RUN_IDS="${RUN_IDS:-A,B}"
ABLATIONS="${ABLATIONS:-base}"

RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-0}"
RUN_CPAC="${RUN_CPAC:-0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-file}"
AUG_EVAL="${AUG_EVAL:-1}"

CKPT_FPS_RUNA="${CKPT_FPS_RUNA:-${WORKDIR}/runs/pretrain_abcd_1024_runA/last.pt}"
CKPT_FPS_RUNB="${CKPT_FPS_RUNB:-${WORKDIR}/runs/pretrain_abcd_1024_runB/last.pt}"
CKPT_RFPS_RUNA="${CKPT_RFPS_RUNA:-${WORKDIR}/runs/pretrain_abcd_1024_rfps_20260225_142727_runA/last.pt}"
CKPT_RFPS_RUNB="${CKPT_RFPS_RUNB:-${WORKDIR}/runs/pretrain_abcd_1024_rfps_20260225_142727_runB/last.pt}"

IFS=',' read -r -a _fam_arr <<< "${FAMILIES}"
IFS=',' read -r -a _var_arr <<< "${VARIANTS}"
IFS=',' read -r -a _run_arr <<< "${RUN_IDS}"
IFS=',' read -r -a _abl_arr <<< "${ABLATIONS}"

for fam in "${_fam_arr[@]}"; do
  fam="$(echo "${fam}" | xargs)"
  case "${fam}" in
    fps)
      ckpt_runa="${CKPT_FPS_RUNA}"
      ckpt_runb="${CKPT_FPS_RUNB}"
      ;;
    rfps)
      ckpt_runa="${CKPT_RFPS_RUNA}"
      ckpt_runb="${CKPT_RFPS_RUNB}"
      ;;
    *)
      echo "[error] unknown family: ${fam} (use fps,rfps)"
      exit 2
      ;;
  esac

  echo "[submit-family] ${fam}"
  env \
    WORKDIR="${WORKDIR}" \
    RUN_SET_BASE_PREFIX="${RUN_SET_BASE}_${fam}" \
    VARIANTS="${VARIANTS}" \
    RUN_IDS="${RUN_IDS}" \
    ABLATIONS="${ABLATIONS}" \
    QSUB_DEPEND="${QSUB_DEPEND:-}" \
    CKPT_RUNA="${ckpt_runa}" \
    CKPT_RUNB="${ckpt_runb}" \
    RUN_SCAN="${RUN_SCAN}" \
    RUN_MODELNET="${RUN_MODELNET}" \
    RUN_CPAC="${RUN_CPAC}" \
    VAL_SPLIT_MODE="${VAL_SPLIT_MODE}" \
    AUG_EVAL="${AUG_EVAL}" \
    bash "${SUBMIT_VARIANTS}"
done

n_fam="${#_fam_arr[@]}"
n_var="${#_var_arr[@]}"
n_run="${#_run_arr[@]}"
n_abl="${#_abl_arr[@]}"
n_total=$((n_fam * n_var * n_run * n_abl))
echo "[summary] families=${n_fam} variants=${n_var} runs=${n_run} ablations=${n_abl} total_eval_jobs=${n_total}"
