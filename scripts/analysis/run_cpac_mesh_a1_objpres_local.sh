#!/usr/bin/env bash
set -euo pipefail

# Objective-preserving CPAC mesh-eval matrix for A-1 query design.
#
# Default model line:
#   runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt
#
# This script compares grid query samplers:
#   - uniform
#   - near_surface
#   - coarse_to_fine (16->32->64)
#
# Environment overrides:
#   CKPT
#   CACHE_ROOT
#   SPLIT
#   GPU_ID
#   MAX_SHAPES
#   HEAD_TRAIN_MAX_SHAPES
#   MESH_EVAL_MAX_SHAPES
#   MESH_GRID_RES
#   MESH_CHUNK_N_QUERY
#   MESH_NUM_SAMPLES
#   MESH_FSCORE_TAU
#   EVAL_SEED

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

CKPT="${CKPT:-runs/eccv_upmix_nepa_qa_dualmask_s0/ckpt_ep049.pt}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"
SPLIT="${SPLIT:-eval}"
GPU_ID="${GPU_ID:-0}"
MAX_SHAPES="${MAX_SHAPES:-800}"
HEAD_TRAIN_MAX_SHAPES="${HEAD_TRAIN_MAX_SHAPES:-4000}"
MESH_EVAL_MAX_SHAPES="${MESH_EVAL_MAX_SHAPES:-80}"
MESH_GRID_RES="${MESH_GRID_RES:-24}"
MESH_CHUNK_N_QUERY="${MESH_CHUNK_N_QUERY:-512}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"
MESH_FSCORE_TAU="${MESH_FSCORE_TAU:-0.01}"
EVAL_SEED="${EVAL_SEED:-0}"

RUN_NAME="$(basename "$(dirname "$CKPT")")"
OUT_DIR="results"
LOG_DIR="logs/analysis/cpac_mesh_a1_objpres"
mkdir -p "$OUT_DIR" "$LOG_DIR"

run_one() {
  local mode="$1"
  local out_json="$2"
  echo "[$(date +"%F %T")] start mode=${mode} out=${out_json}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "$PYTHON_BIN" -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${CACHE_ROOT}" \
    --split "${SPLIT}" \
    --ckpt "${CKPT}" \
    --context_backend pointcloud_noray \
    --head_train_split train_udf \
    --head_train_backend udfgrid \
    --head_train_max_shapes "${HEAD_TRAIN_MAX_SHAPES}" \
    --n_context 256 --n_query 256 \
    --disjoint_context_query 1 \
    --context_mode_train normal --context_mode_test normal \
    --rep_source h --query_source grid \
    --grid_sample_mode "${mode}" \
    --grid_near_tau 0.05 --grid_near_frac 0.8 \
    --grid_res_schedule 16,32,64 --grid_c2f_expand 1 \
    --max_shapes "${MAX_SHAPES}" --head_train_ratio 0.2 \
    --ridge_lambda 1e-3 --tau 0.03 --eval_seed "${EVAL_SEED}" \
    --mesh_eval 1 \
    --mesh_eval_max_shapes "${MESH_EVAL_MAX_SHAPES}" \
    --mesh_grid_res "${MESH_GRID_RES}" \
    --mesh_chunk_n_query "${MESH_CHUNK_N_QUERY}" \
    --mesh_num_samples "${MESH_NUM_SAMPLES}" \
    --mesh_fscore_tau "${MESH_FSCORE_TAU}" \
    --out_json "${out_json}" \
    > "${LOG_DIR}/$(basename "${out_json%.json}").log" 2>&1
  echo "[$(date +"%F %T")] done mode=${mode}"
}

run_one "uniform" "${OUT_DIR}/cpac_${RUN_NAME}_pc2udf_800_grid_uniform_mesh_eval_seed${EVAL_SEED}.json"
run_one "near_surface" "${OUT_DIR}/cpac_${RUN_NAME}_pc2udf_800_grid_near08_mesh_eval_seed${EVAL_SEED}.json"
run_one "coarse_to_fine" "${OUT_DIR}/cpac_${RUN_NAME}_pc2udf_800_grid_c2f163264_mesh_eval_seed${EVAL_SEED}.json"

echo "[$(date +"%F %T")] all done"
