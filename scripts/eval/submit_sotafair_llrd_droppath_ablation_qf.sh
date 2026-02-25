#!/bin/bash
set -euo pipefail

# Submit SOTA-fair classification/eval jobs with LLRD/drop_path ablations.
#
# Default matrix (4 combos):
#   base    : llrd=1.0,  drop_path=0.0
#   llrd    : llrd=0.75, drop_path=0.0
#   dp      : llrd=1.0,  drop_path=0.1
#   llrd_dp : llrd=0.75, drop_path=0.1
#
# This wrapper calls scripts/eval/submit_abcd_cls_cpac_qf.sh once per combo
# (=> 4 runs x N combos jobs).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBMIT_ABCD="${SCRIPT_DIR}/submit_abcd_cls_cpac_qf.sh"

if [[ ! -x "${SUBMIT_ABCD}" ]]; then
  echo "[error] missing executable submit script: ${SUBMIT_ABCD}"
  exit 1
fi

RUN_SET_BASE="${RUN_SET_BASE:-$(date +%Y%m%d_%H%M%S)}"
ABLATIONS="${ABLATIONS:-base,llrd,dp,llrd_dp}"

# SOTA-fair defaults.
PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS:-pc_xyz}"
PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS:-pt_dist_pool}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
N_RAY_CLS="${N_RAY_CLS:-0}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}"
RUN_CPAC="${RUN_CPAC:-0}"
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT:-data/scanobjectnn_main_split_v2}"
MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT:-data/modelnet40_cache_v2}"
UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}"
MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}"
AUG_EVAL="${AUG_EVAL:-1}"

IFS=',' read -r -a ablation_arr <<< "${ABLATIONS}"

for ab in "${ablation_arr[@]}"; do
  ab="$(echo "${ab}" | xargs)"
  llrd="1.0"
  drop_path="0.0"
  case "${ab}" in
    base)
      llrd="1.0"
      drop_path="0.0"
      ;;
    llrd)
      llrd="${LLRD_ONLY:-0.75}"
      drop_path="0.0"
      ;;
    dp)
      llrd="1.0"
      drop_path="${DROP_PATH_ONLY:-0.1}"
      ;;
    llrd_dp)
      llrd="${LLRD_WITH_DP:-0.75}"
      drop_path="${DROP_PATH_WITH_LLRD:-0.1}"
      ;;
    *)
      echo "[error] unknown ablation tag: ${ab}"
      exit 2
      ;;
  esac

  run_set="${RUN_SET_BASE}_${ab}"
  echo "[submit-ablation] ${ab} run_set=${run_set} llrd=${llrd} drop_path=${drop_path}"
  env \
    RUN_SET="${run_set}" \
    LLRD="${llrd}" \
    DROP_PATH="${drop_path}" \
    PT_XYZ_KEY_CLS="${PT_XYZ_KEY_CLS}" \
    PT_DIST_KEY_CLS="${PT_DIST_KEY_CLS}" \
    ABLATE_POINT_DIST="${ABLATE_POINT_DIST}" \
    N_RAY_CLS="${N_RAY_CLS}" \
    POINT_ORDER_MODE="${POINT_ORDER_MODE}" \
    VAL_SPLIT_MODE="${VAL_SPLIT_MODE}" \
    RUN_CPAC="${RUN_CPAC}" \
    RUN_SCAN="${RUN_SCAN}" \
    RUN_MODELNET="${RUN_MODELNET}" \
    SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT}" \
    MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT}" \
    UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT}" \
    SCAN_AUG_PRESET="${SCAN_AUG_PRESET}" \
    MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET}" \
    AUG_EVAL="${AUG_EVAL}" \
    bash "${SUBMIT_ABCD}"
done

echo "[done] submitted SOTA-fair LLRD/drop_path ablation matrix"
