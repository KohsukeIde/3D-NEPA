#!/bin/bash
set -euo pipefail

# Submit 18 eval jobs (9 variants x 2 protocols) for A256 query-rethink checkpoints.
# This script is eval-only (no pretrain submission) and is intended for
# classification-inclusive reruns; CPAC can be toggled independently.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EVAL_SCRIPT="${WORKDIR}/scripts/eval/nepa3d_eval_cls_cpac_qf.sh"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

SOURCE_RUN_SET="${SOURCE_RUN_SET:-a256_queryrethink_20260226_024537}"
RUN_SET="${RUN_SET:-a256_queryrethink_eval18_$(date +%Y%m%d_%H%M%S)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
QSUB_DEPEND="${QSUB_DEPEND:-}"

# classification-focused defaults
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
RUN_CPAC="${RUN_CPAC:-0}"

SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT:-}"
MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT:-data/modelnet40_cache_v2}"
UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"

BATCH_SCAN="${BATCH_SCAN:-96}"
BATCH_MODELNET="${BATCH_MODELNET:-128}"
EPOCHS_CLS="${EPOCHS_CLS:-100}"
LR_CLS="${LR_CLS:-1e-4}"
N_POINT_CLS="${N_POINT_CLS:-256}"
N_RAY_CLS="${N_RAY_CLS:-0}"
VAL_SPLIT_MODE="${VAL_SPLIT_MODE:-group_auto}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
PT_SAMPLE_MODE_TRAIN_CLS="${PT_SAMPLE_MODE_TRAIN_CLS:-fps}"
PT_SAMPLE_MODE_EVAL_CLS="${PT_SAMPLE_MODE_EVAL_CLS:-fps}"
PT_RFPS_M_CLS="${PT_RFPS_M_CLS:-4096}"

SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}"
MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}"
AUG_EVAL="${AUG_EVAL:-1}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-1}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"

USE_FC_NORM="${USE_FC_NORM:-0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
WEIGHT_DECAY_CLS="${WEIGHT_DECAY_CLS:-0.05}"
WEIGHT_DECAY_NORM="${WEIGHT_DECAY_NORM:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
LLRD="${LLRD:-1.0}"
LLRD_MODE="${LLRD_MODE:-exp}"
DROP_PATH="${DROP_PATH:-0.0}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"

# CPAC knobs (used only when RUN_CPAC=1)
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-256}"
CPAC_N_QUERY="${CPAC_N_QUERY:-256}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:-1300}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_MESH_EVAL="${CPAC_MESH_EVAL:-1}"
CPAC_MESH_EVAL_MAX_SHAPES="${CPAC_MESH_EVAL_MAX_SHAPES:-800}"
CPAC_MESH_GRID_RES="${CPAC_MESH_GRID_RES:-24}"
CPAC_MESH_CHUNK_N_QUERY="${CPAC_MESH_CHUNK_N_QUERY:-256}"
CPAC_MESH_MC_LEVEL="${CPAC_MESH_MC_LEVEL:-0.03}"
CPAC_MESH_NUM_SAMPLES="${CPAC_MESH_NUM_SAMPLES:-10000}"
CPAC_MESH_FSCORE_TAU="${CPAC_MESH_FSCORE_TAU:-0.01}"
CPAC_MESH_STORE_PER_SHAPE="${CPAC_MESH_STORE_PER_SHAPE:-0}"

if [[ "${RUN_SCAN}" == "1" ]]; then
  if [[ -z "${SCAN_CACHE_ROOT}" ]]; then
    echo "[error] SCAN_CACHE_ROOT is required when RUN_SCAN=1."
    echo "        Use variant cache roots (obj_bg/obj_only/pb_t50_rs)."
    exit 2
  fi
  if [[ "${SCAN_CACHE_ROOT}" == *"scanobjectnn_main_split_v2"* ]]; then
    echo "[error] SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT} is disallowed (main_split deprecated)."
    exit 2
  fi
fi

RESULTS_ROOT="${RESULTS_ROOT:-results/${RUN_SET}}"
EVAL_ROOT="${EVAL_ROOT:-runs/eval_${RUN_SET}}"
LOG_ROOT="${LOG_ROOT:-logs/eval/${RUN_SET}}"
mkdir -p "${WORKDIR}/${RESULTS_ROOT}" "${WORKDIR}/${EVAL_ROOT}" "${WORKDIR}/${LOG_ROOT}"

# rid|seed_offset
defs=(
  "b00_interleave_theta|0"
  "b01_split_theta|1"
  "b02_split_theta_typepos|2"
  "b03_split_viewraster_typepos|3"
  "b04_split_xanchor_morton_typepos|4"
  "b05_split_xanchor_fps_typepos|5"
  "b06_split_dirfps_typepos|6"
  "b07_event_xanchor_typepos|7"
  "b08_event_dirfps_typepos|8"
)

jobs_txt="${WORKDIR}/${LOG_ROOT}/submitted_jobs_${RUN_SET}.txt"
: > "${jobs_txt}"

echo "[submit] eval18 for ${#defs[@]} variants x 2 protocols"
for row in "${defs[@]}"; do
  IFS='|' read -r rid seed_off <<< "${row}"
  ckpt="${WORKDIR}/runs/pretrain_a256_queryrethink_${SOURCE_RUN_SET}_${rid}/last.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[warn] missing ckpt, skip: ${ckpt}"
    continue
  fi

  for proto in sotafair nepafull; do
    run_tag="${rid}_${proto}_eval18"
    seed=$((3600 + seed_off * 2))
    if [[ "${proto}" == "nepafull" ]]; then
      seed=$((seed + 1))
    fi

    if [[ "${proto}" == "sotafair" ]]; then
      pt_xyz_key="pc_xyz"
      ablate_point_dist="1"
      point_order_mode="morton"
    else
      pt_xyz_key="pt_xyz_pool"
      ablate_point_dist="0"
      point_order_mode="fps"
    fi

    vars="WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${seed},NUM_WORKERS=${NUM_WORKERS},NPROC_PER_NODE=${NPROC_PER_NODE},RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},SCAN_CACHE_ROOT=${SCAN_CACHE_ROOT},MODELNET_CACHE_ROOT=${MODELNET_CACHE_ROOT},UNPAIRED_CACHE_ROOT=${UNPAIRED_CACHE_ROOT},BATCH_SCAN=${BATCH_SCAN},BATCH_MODELNET=${BATCH_MODELNET},EPOCHS_CLS=${EPOCHS_CLS},LR_CLS=${LR_CLS},N_POINT_CLS=${N_POINT_CLS},N_RAY_CLS=${N_RAY_CLS},VAL_SPLIT_MODE=${VAL_SPLIT_MODE},CLS_POOLING=${CLS_POOLING},PT_XYZ_KEY_CLS=${pt_xyz_key},PT_DIST_KEY_CLS=pt_dist_pool,ABLATE_POINT_DIST=${ablate_point_dist},POINT_ORDER_MODE=${point_order_mode},PT_SAMPLE_MODE_TRAIN_CLS=${PT_SAMPLE_MODE_TRAIN_CLS},PT_SAMPLE_MODE_EVAL_CLS=${PT_SAMPLE_MODE_EVAL_CLS},PT_RFPS_M_CLS=${PT_RFPS_M_CLS},SCAN_AUG_PRESET=${SCAN_AUG_PRESET},MODELNET_AUG_PRESET=${MODELNET_AUG_PRESET},AUG_EVAL=${AUG_EVAL},AUG_RECOMPUTE_DIST=${AUG_RECOMPUTE_DIST},MC_EVAL_K_VAL=${MC_EVAL_K_VAL},MC_EVAL_K_TEST=${MC_EVAL_K_TEST},USE_FC_NORM=${USE_FC_NORM},LABEL_SMOOTHING=${LABEL_SMOOTHING},WEIGHT_DECAY_CLS=${WEIGHT_DECAY_CLS},WEIGHT_DECAY_NORM=${WEIGHT_DECAY_NORM},LR_SCHEDULER=${LR_SCHEDULER},WARMUP_EPOCHS=${WARMUP_EPOCHS},WARMUP_START_FACTOR=${WARMUP_START_FACTOR},MIN_LR=${MIN_LR},LLRD=${LLRD},LLRD_MODE=${LLRD_MODE},DROP_PATH=${DROP_PATH},GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS},MAX_GRAD_NORM=${MAX_GRAD_NORM},DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS},CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN},CPAC_MAX_SHAPES=${CPAC_MAX_SHAPES},CPAC_MESH_EVAL=${CPAC_MESH_EVAL},CPAC_MESH_EVAL_MAX_SHAPES=${CPAC_MESH_EVAL_MAX_SHAPES},CPAC_MESH_GRID_RES=${CPAC_MESH_GRID_RES},CPAC_MESH_CHUNK_N_QUERY=${CPAC_MESH_CHUNK_N_QUERY},CPAC_MESH_MC_LEVEL=${CPAC_MESH_MC_LEVEL},CPAC_MESH_NUM_SAMPLES=${CPAC_MESH_NUM_SAMPLES},CPAC_MESH_FSCORE_TAU=${CPAC_MESH_FSCORE_TAU},CPAC_MESH_STORE_PER_SHAPE=${CPAC_MESH_STORE_PER_SHAPE}"

    out="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
    err="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"
    cmd=(
      qsub
      -l "rt_QF=${RT_QF}"
      -l "walltime=${WALLTIME}"
      -W "group_list=${GROUP_LIST}"
      -N "ev_${rid}_${proto}"
      -o "${out}"
      -e "${err}"
      -v "${vars}"
    )
    if [[ -n "${QSUB_DEPEND}" ]]; then
      cmd+=( -W "depend=${QSUB_DEPEND}" )
    fi
    cmd+=( "${EVAL_SCRIPT}" )
    jid="$("${cmd[@]}")"
    echo "${run_tag} ${jid}" | tee -a "${jobs_txt}"
  done
done

echo "[done] run_set=${RUN_SET}"
echo "[done] log_root=${LOG_ROOT}"
echo "[done] results_root=${RESULTS_ROOT}"
echo "[done] jobs=${jobs_txt}"
