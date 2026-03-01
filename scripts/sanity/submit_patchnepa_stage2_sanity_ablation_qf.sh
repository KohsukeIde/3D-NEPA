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
RUN_ROOT="${RUN_ROOT:-patchnepa_stage2_sanity_${TS}}"
LOG_DIR="${LOG_DIR:-${WORKDIR}/logs/sanity/patchnepa_stage2_sanity/${RUN_ROOT}}"
mkdir -p "${LOG_DIR}"

# Lightweight sanity settings (pretrain short + finetune short)
PT_EPOCHS="${PT_EPOCHS:-12}"
PT_WALLTIME="${PT_WALLTIME:-08:00:00}"
PT_BATCH="${PT_BATCH:-16}"
PT_RT_QF="${PT_RT_QF:-4}"

FT_EPOCHS="${FT_EPOCHS:-120}"
FT_VARIANTS="${FT_VARIANTS:-obj_only}"
FT_WALLTIME="${FT_WALLTIME:-08:00:00}"

# Common pretrain controls (baseline aligned)
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
  "N_RAY=1024"
  "USE_RAY_PATCH=1"
  "QA_LAYOUT=split_sep"
  "QA_SEP_TOKEN=1"
  "DUAL_MASK_NEAR=0.5"
  "DUAL_MASK_FAR=0.1"
  "DUAL_MASK_TYPE_AWARE=1"
  "TYPE_SPECIFIC_POS=0"
  "STAGE2_REQUIRE_RAY=1"
)

# name | overrides...
cases=(
  "base_split_dual_taware_ray"
  "layout_interleave QA_LAYOUT=interleave QA_SEP_TOKEN=1"
  "dualmask_off DUAL_MASK_NEAR=0.0 DUAL_MASK_FAR=0.0 DUAL_MASK_TYPE_AWARE=0"
  "typeaware_off DUAL_MASK_TYPE_AWARE=0"
  "typepos_on TYPE_SPECIFIC_POS=1"
  "ray_off USE_RAY_PATCH=0 N_RAY=0 STAGE2_REQUIRE_RAY=0"
)

RESULTS_TSV="${LOG_DIR}/submitted_jobs.tsv"
echo -e "case\tpretrain_job\tpretrain_runset\tft_job\tft_runset\tckpt" > "${RESULTS_TSV}"

submit_one_case() {
  local case_name="$1"
  shift || true
  local run_set="${RUN_ROOT}_${case_name}"
  local run_tag="run_${run_set}"

  local env_args=("RUN_SET=${run_set}" "RUN_TAG=${run_tag}" "JOB_NAME=pnpa_s2_${case_name}")
  env_args+=("${COMMON_ENV[@]}")
  while [[ $# -gt 0 ]]; do
    env_args+=("$1")
    shift
  done

  echo "[submit][pretrain] case=${case_name} run_set=${run_set}"
  local out
  out=$(env "${env_args[@]}" bash "${PRETRAIN_SUBMIT}")
  echo "${out}" | tee "${LOG_DIR}/${case_name}.pretrain.submit.log"

  local pt_jid
  pt_jid=$(echo "${out}" | awk '/\[submitted\]/{print $2; exit}')
  if [[ -z "${pt_jid}" ]]; then
    echo "[error] failed to parse pretrain job id for case=${case_name}"
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

  echo "${ft_out}" | tee "${LOG_DIR}/${case_name}.ft.submit.log"

  local ft_jid
  ft_jid=$(echo "${ft_out}" | awk '/\[submitted\]/{print $2; exit}')
  if [[ -z "${ft_jid}" ]]; then
    ft_jid="(parse_failed)"
  fi

  echo -e "${case_name}\t${pt_jid}\t${run_set}\t${ft_jid}\t${ft_run_set}\t${ckpt}" >> "${RESULTS_TSV}"
}

for row in "${cases[@]}"; do
  # shellcheck disable=SC2206
  parts=(${row})
  case_name="${parts[0]}"
  overrides=("${parts[@]:1}")
  submit_one_case "${case_name}" "${overrides[@]}"
done

echo "[done] ${RESULTS_TSV}"
cat "${RESULTS_TSV}"
