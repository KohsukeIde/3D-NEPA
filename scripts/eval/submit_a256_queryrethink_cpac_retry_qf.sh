#!/bin/bash
set -euo pipefail

# Retry CPAC(+mesh/chamfer) for the finished A256 query-rethink checkpoints.
# This re-runs only CPAC stage (no Scan/ModelNet classification).
#
# Default fix:
# - keep CPAC_N_CONTEXT=256, CPAC_N_QUERY=256, CPAC_MAX_LEN=1300
# - reduce mesh chunk query from 512 -> 256 to satisfy max_len precheck.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
EVAL_SCRIPT="${WORKDIR}/scripts/eval/nepa3d_eval_cls_cpac_qf.sh"

if ! command -v qsub >/dev/null 2>&1; then
  echo "[error] qsub not found"
  exit 1
fi

SOURCE_RUN_SET="${SOURCE_RUN_SET:-a256_queryrethink_20260226_024537}"
RUN_SET="${RUN_SET:-a256_queryrethink_cpac_retry_$(date +%Y%m%d_%H%M%S)}"

RT_QF="${RT_QF:-1}"
WALLTIME="${WALLTIME:-24:00:00}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# CPAC-only
RUN_SCAN="${RUN_SCAN:-0}"
RUN_MODELNET="${RUN_MODELNET:-0}"
RUN_CPAC="${RUN_CPAC:-1}"

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

echo "[submit] CPAC-only retry for ${#defs[@]} variants x 2 protocols (18 jobs)"
for row in "${defs[@]}"; do
  IFS='|' read -r rid seed_off <<< "${row}"
  ckpt="${WORKDIR}/runs/pretrain_a256_queryrethink_${SOURCE_RUN_SET}_${rid}/last.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[warn] missing ckpt, skip: ${ckpt}"
    continue
  fi

  for proto in sotafair nepafull; do
    run_tag="${rid}_${proto}_cpacfix"
    seed=$((3200 + seed_off * 2))
    if [[ "${proto}" == "nepafull" ]]; then
      seed=$((seed + 1))
    fi

    vars="WORKDIR=${WORKDIR},RUN_TAG=${run_tag},CKPT=${ckpt},EVAL_ROOT=${EVAL_ROOT},RESULTS_ROOT=${RESULTS_ROOT},LOG_ROOT=${LOG_ROOT},SEED=${seed},NUM_WORKERS=${NUM_WORKERS},NPROC_PER_NODE=${NPROC_PER_NODE},RUN_SCAN=${RUN_SCAN},RUN_MODELNET=${RUN_MODELNET},RUN_CPAC=${RUN_CPAC},CPAC_N_CONTEXT=${CPAC_N_CONTEXT},CPAC_N_QUERY=${CPAC_N_QUERY},CPAC_MAX_LEN=${CPAC_MAX_LEN},CPAC_MAX_SHAPES=${CPAC_MAX_SHAPES},CPAC_MESH_EVAL=${CPAC_MESH_EVAL},CPAC_MESH_EVAL_MAX_SHAPES=${CPAC_MESH_EVAL_MAX_SHAPES},CPAC_MESH_GRID_RES=${CPAC_MESH_GRID_RES},CPAC_MESH_CHUNK_N_QUERY=${CPAC_MESH_CHUNK_N_QUERY},CPAC_MESH_MC_LEVEL=${CPAC_MESH_MC_LEVEL},CPAC_MESH_NUM_SAMPLES=${CPAC_MESH_NUM_SAMPLES},CPAC_MESH_FSCORE_TAU=${CPAC_MESH_FSCORE_TAU},CPAC_MESH_STORE_PER_SHAPE=${CPAC_MESH_STORE_PER_SHAPE},UNPAIRED_CACHE_ROOT=data/shapenet_unpaired_cache_v1"

    out="${WORKDIR}/${LOG_ROOT}/${run_tag}.out"
    err="${WORKDIR}/${LOG_ROOT}/${run_tag}.err"
    jid="$(qsub -l "rt_QF=${RT_QF}" -l "walltime=${WALLTIME}" -W "group_list=${GROUP_LIST}" -N "cpac_${rid}_${proto}" -o "${out}" -e "${err}" -v "${vars}" "${EVAL_SCRIPT}")"
    echo "${run_tag} ${jid}" | tee -a "${jobs_txt}"
  done
done

echo "[done] run_set=${RUN_SET}"
echo "[done] log_root=${LOG_ROOT}"
echo "[done] results_root=${RESULTS_ROOT}"
echo "[done] jobs=${jobs_txt}"
