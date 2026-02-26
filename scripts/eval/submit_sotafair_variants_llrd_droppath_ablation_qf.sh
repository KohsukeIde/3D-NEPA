#!/bin/bash
set -euo pipefail

# Submit SOTA-fair ablations on ScanObjectNN protocol variants:
#   obj_bg / obj_only / pb_t50_rs
#
# Per variant, this wrapper calls:
#   scripts/eval/submit_sotafair_llrd_droppath_ablation_qf.sh
# which submits 4 runs (A/B/C/D) x N ablations.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_ABL="${SCRIPT_DIR}/submit_sotafair_llrd_droppath_ablation_qf.sh"

if [[ ! -x "${SUBMIT_ABL}" ]]; then
  echo "[error] missing executable submit script: ${SUBMIT_ABL}"
  exit 1
fi

VARIANTS="${VARIANTS:-obj_bg,obj_only,pb_t50_rs}"
RUN_SET_BASE_PREFIX="${RUN_SET_BASE_PREFIX:-$(date +%Y%m%d_%H%M%S)}"

SCAN_CACHE_ROOT_OBJ_BG="${SCAN_CACHE_ROOT_OBJ_BG:-data/scanobjectnn_obj_bg_v2}"
SCAN_CACHE_ROOT_OBJ_ONLY="${SCAN_CACHE_ROOT_OBJ_ONLY:-data/scanobjectnn_obj_only_v2}"
SCAN_CACHE_ROOT_PB_T50_RS="${SCAN_CACHE_ROOT_PB_T50_RS:-data/scanobjectnn_pb_t50_rs_v2}"

RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-0}"
RUN_CPAC="${RUN_CPAC:-0}"

IFS=',' read -r -a _ab_arr <<< "${ABLATIONS:-base,llrd,dp,llrd_dp}"
IFS=',' read -r -a _var_arr <<< "${VARIANTS}"

for variant in "${_var_arr[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  cache_root=""
  case "${variant}" in
    obj_bg)
      cache_root="${SCAN_CACHE_ROOT_OBJ_BG}"
      ;;
    obj_only)
      cache_root="${SCAN_CACHE_ROOT_OBJ_ONLY}"
      ;;
    pb_t50_rs)
      cache_root="${SCAN_CACHE_ROOT_PB_T50_RS}"
      ;;
    *)
      echo "[error] unknown variant: ${variant} (use obj_bg,obj_only,pb_t50_rs)"
      exit 2
      ;;
  esac

  if [[ ! -d "${cache_root}" ]]; then
    if [[ -n "${QSUB_DEPEND:-}" ]]; then
      echo "[warn] cache not found yet for ${variant}: ${cache_root} (allowed because QSUB_DEPEND is set)"
    else
      echo "[error] missing cache for ${variant}: ${cache_root}"
      exit 3
    fi
  fi

  run_set_base="${RUN_SET_BASE_PREFIX}_${variant}"
  echo "[submit-variant] variant=${variant} cache_root=${cache_root} run_set_base=${run_set_base}"

  env \
    RUN_SET_BASE="${run_set_base}" \
    SCAN_CACHE_ROOT="${cache_root}" \
    RUN_SCAN="${RUN_SCAN}" \
    RUN_MODELNET="${RUN_MODELNET}" \
    RUN_CPAC="${RUN_CPAC}" \
    bash "${SUBMIT_ABL}"
done

n_variants="${#_var_arr[@]}"
n_abl="${#_ab_arr[@]}"
n_eval_jobs=$((n_variants * 4 * n_abl))
echo "[summary] variants=${n_variants} ablations=${n_abl} eval_jobs=${n_eval_jobs}"
