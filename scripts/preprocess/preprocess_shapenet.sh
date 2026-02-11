#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SHAPENET_ROOT="${SHAPENET_ROOT:-data/ShapeNetCore.v2}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_cache_v0}"
SPLIT="${SPLIT:-all}"
MESH_GLOB="${MESH_GLOB:-*/*/models/model_normalized.obj}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SPLIT_SEED="${SPLIT_SEED:-0}"

PC_POINTS="${PC_POINTS:-2048}"
PT_POOL="${PT_POOL:-20000}"
RAY_POOL="${RAY_POOL:-8000}"
N_VIEWS="${N_VIEWS:-20}"
RAYS_PER_VIEW="${RAYS_PER_VIEW:-400}"
PC_GRID="${PC_GRID:-64}"
PC_DILATE="${PC_DILATE:-1}"
PC_MAX_STEPS="${PC_MAX_STEPS:-0}"
DF_GRID="${DF_GRID:-64}"
DF_DILATE="${DF_DILATE:-1}"
PT_SURFACE_RATIO="${PT_SURFACE_RATIO:-0.5}"
PT_SURFACE_SIGMA="${PT_SURFACE_SIGMA:-0.02}"
PT_QUERY_CHUNK="${PT_QUERY_CHUNK:-2048}"
RAY_QUERY_CHUNK="${RAY_QUERY_CHUNK:-2048}"
PT_DIST_MODE="${PT_DIST_MODE:-kdtree}"
DIST_REF_POINTS="${DIST_REF_POINTS:-8192}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-1}"
MAX_TASKS_PER_CHILD="${MAX_TASKS_PER_CHILD:-2}"
OVERWRITE="${OVERWRITE:-0}"
NO_PC_RAYS="${NO_PC_RAYS:-0}"
NO_UDF="${NO_UDF:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${SHAPENET_ROOT}" ]; then
  echo "[error] ShapeNet root not found: ${SHAPENET_ROOT}"
  echo "        set SHAPENET_ROOT=/path/to/ShapeNetCore.v2"
  exit 1
fi

EXTRA_FLAGS=""
if [ "${OVERWRITE}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --overwrite"
fi
if [ "${NO_PC_RAYS}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --no_pc_rays"
fi
if [ "${NO_UDF}" = "1" ]; then
  EXTRA_FLAGS="${EXTRA_FLAGS} --no_udf"
fi

set -x
"${PYTHON_BIN}" -m nepa3d.data.preprocess_shapenet \
  --shapenet_root "${SHAPENET_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --split "${SPLIT}" \
  --mesh_glob "${MESH_GLOB}" \
  --test_ratio "${TEST_RATIO}" \
  --split_seed "${SPLIT_SEED}" \
  --pc_points "${PC_POINTS}" \
  --pt_pool "${PT_POOL}" \
  --ray_pool "${RAY_POOL}" \
  --n_views "${N_VIEWS}" \
  --rays_per_view "${RAYS_PER_VIEW}" \
  --pc_grid "${PC_GRID}" \
  --pc_dilate "${PC_DILATE}" \
  --pc_max_steps "${PC_MAX_STEPS}" \
  --df_grid "${DF_GRID}" \
  --df_dilate "${DF_DILATE}" \
  --pt_surface_ratio "${PT_SURFACE_RATIO}" \
  --pt_surface_sigma "${PT_SURFACE_SIGMA}" \
  --pt_query_chunk "${PT_QUERY_CHUNK}" \
  --ray_query_chunk "${RAY_QUERY_CHUNK}" \
  --pt_dist_mode "${PT_DIST_MODE}" \
  --dist_ref_points "${DIST_REF_POINTS}" \
  --seed "${SEED}" \
  --workers "${WORKERS}" \
  --chunk_size "${CHUNK_SIZE}" \
  --max_tasks_per_child "${MAX_TASKS_PER_CHILD}" \
  ${EXTRA_FLAGS}

