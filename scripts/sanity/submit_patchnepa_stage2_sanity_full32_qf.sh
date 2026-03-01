#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${WORKDIR}" || exit 1

PRETRAIN_SUBMIT="${WORKDIR}/scripts/pretrain/submit_pretrain_patch_nepa_pointonly_qf.sh"
FT_SUBMIT="${WORKDIR}/scripts/sanity/submit_patchnepa_finetune_variants_qf.sh"

if [[ ! -x "${PRETRAIN_SUBMIT}" ]]; then
  echo "[error] missing pretrain submit script: ${PRETRAIN_SUBMIT}"
  exit 1
fi
if [[ ! -x "${FT_SUBMIT}" ]]; then
  echo "[error] missing ft submit script: ${FT_SUBMIT}"
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-patchnepa_stage2_sanity32_${TS}}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/patchnepa_stage2_sanity/${RUN_ROOT}}"
mkdir -p "${LOG_DIR}"

# Lightweight sanity settings
PT_EPOCHS="${PT_EPOCHS:-12}"
PT_WALLTIME="${PT_WALLTIME:-08:00:00}"
PT_BATCH="${PT_BATCH:-16}"
PT_RT_QF="${PT_RT_QF:-4}"

FT_EPOCHS="${FT_EPOCHS:-120}"
FT_VARIANTS="${FT_VARIANTS:-obj_only}"
FT_WALLTIME="${FT_WALLTIME:-08:00:00}"

COMMON_ENV=(
  "RT_QF=${PT_RT_QF}"
  "WALLTIME=${PT_WALLTIME}"
  "EPOCHS=${PT_EPOCHS}"
  "BATCH=${PT_BATCH}"
  "QA_TOKENS=1"
  "ENCDEC_ARCH=0"
  "DUAL_MASK_WINDOW=32"
  "PT_SAMPLE_MODE=rfps_cached"
  "PT_RFPS_KEY=pt_rfps_order_bank"
  "MIX_CONFIG=nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_onepass.yaml"
  "QA_FUSE=add"
  "NEPA2D_POS=1"
)

RESULTS_TSV="${LOG_DIR}/submitted_jobs.tsv"
echo -e "case\tlayout\tdual_mask\ttype_aware\ttype_specific_pos\tray\tpretrain_job\tpretrain_runset\tft_job\tft_runset\tckpt" > "${RESULTS_TSV}"

submit_case() {
  local layout="$1" dual="$2" typeaware="$3" typepos="$4" ray="$5"

  local sep="1"
  if [[ "${layout}" == "interleave" ]]; then
    sep="0"
  fi

  local dual_near="0.5"
  local dual_far="0.1"
  if [[ "${dual}" == "0" ]]; then
    dual_near="0.0"
    dual_far="0.0"
  fi

  local use_ray="1"
  local n_ray="1024"
  local require_ray="1"
  if [[ "${ray}" == "0" ]]; then
    use_ray="0"
    n_ray="0"
    require_ray="0"
  fi

  local lcode="S"
  if [[ "${layout}" == "interleave" ]]; then
    lcode="I"
  fi

  local case_name="l${lcode}_d${dual}_ta${typeaware}_tp${typepos}_r${ray}"
  local run_set="${RUN_ROOT}_${case_name}"
  local run_tag="run_${run_set}"

  local env_args=(
    "RUN_SET=${run_set}"
    "RUN_TAG=${run_tag}"
    "JOB_NAME=pnpa32_${case_name}"
    "QA_LAYOUT=${layout}"
    "QA_SEP_TOKEN=${sep}"
    "DUAL_MASK_NEAR=${dual_near}"
    "DUAL_MASK_FAR=${dual_far}"
    "DUAL_MASK_TYPE_AWARE=${typeaware}"
    "TYPE_SPECIFIC_POS=${typepos}"
    "USE_RAY_PATCH=${use_ray}"
    "N_RAY=${n_ray}"
    "STAGE2_REQUIRE_RAY=${require_ray}"
  )
  env_args+=("${COMMON_ENV[@]}")

  echo "[submit][pretrain] case=${case_name}"
  local out
  out=$(env "${env_args[@]}" bash "${PRETRAIN_SUBMIT}")
  echo "${out}" > "${LOG_DIR}/${case_name}.pretrain.submit.log"

  local pt_jid
  pt_jid=$(echo "${out}" | awk '/\[submitted\]/{print $2; exit}')
  if [[ -z "${pt_jid}" ]]; then
    echo "[error] failed to parse pretrain job id for ${case_name}"
    exit 2
  fi

  local ckpt="runs/patchnepa_rayqa/${run_set}/ckpt_latest.pt"
  local ft_run_set="${RUN_ROOT}_ft_${case_name}"
  echo "[submit][ft] case=${case_name} depend=afterok:${pt_jid}"

  local ft_out
  ft_out=$(env \
    CKPT="${ckpt}" \
    VARIANTS="${FT_VARIANTS}" \
    RUN_SET="${ft_run_set}" \
    EPOCHS="${FT_EPOCHS}" \
    WALLTIME="${FT_WALLTIME}" \
    QSUB_DEPEND="afterok:${pt_jid}" \
    MODEL_SOURCE=patchnepa \
    POOLING=cls \
    HEAD_MODE=linear \
    PATCHNEPA_CLS_TOKEN_SOURCE=last_q \
    PATCHNEPA_FREEZE_PATCH_EMBED=1 \
    LLRD_START=0.35 \
    LLRD_END=1.0 \
    LLRD_SCHEDULER=llrd_cosine_warmup \
    bash "${FT_SUBMIT}")

  echo "${ft_out}" > "${LOG_DIR}/${case_name}.ft.submit.log"

  local ft_jid
  ft_jid=$(echo "${ft_out}" | awk '/\[submitted\]/{print $2; exit}')
  if [[ -z "${ft_jid}" ]]; then
    ft_jid="(parse_failed)"
  fi

  echo -e "${case_name}\t${layout}\t${dual}\t${typeaware}\t${typepos}\t${ray}\t${pt_jid}\t${run_set}\t${ft_jid}\t${ft_run_set}\t${ckpt}" >> "${RESULTS_TSV}"
}

for layout in split_sep interleave; do
  for dual in 1 0; do
    for typeaware in 1 0; do
      for typepos in 0 1; do
        for ray in 1 0; do
          submit_case "${layout}" "${dual}" "${typeaware}" "${typepos}" "${ray}"
        done
      done
    done
  done
done

echo "[done] ${RESULTS_TSV}"
cat "${RESULTS_TSV}"
