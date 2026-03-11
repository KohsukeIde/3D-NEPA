#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SHAPENET_ROOT="${SHAPENET_ROOT:-data/ShapeNetCore.v2}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_cache_v2}"
SYNSETS="${SYNSETS:-}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-32}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MISSING_ONLY="${MISSING_ONLY:-0}"

# Sizes
N_SURF="${N_SURF:-8192}"
N_MESH_QRY="${N_MESH_QRY:-2048}"
N_UDF_QRY="${N_UDF_QRY:-8192}"
N_PC="${N_PC:-2048}"
N_PC_QRY="${N_PC_QRY:-1024}"
N_RAYS="${N_RAYS:-4096}"

# Point-cloud degradation (scan-like)
PC_VIEW_CROP="${PC_VIEW_CROP:-0.5}"
PC_NOISE_STD="${PC_NOISE_STD:-0.005}"
PC_DROPOUT="${PC_DROPOUT:-0.1}"

# UDF query mix
UDF_NEAR_RATIO="${UDF_NEAR_RATIO:-0.5}"
UDF_NEAR_STD="${UDF_NEAR_STD:-0.05}"

# KNN and ray params
CURVATURE_KNN="${CURVATURE_KNN:-20}"
PCA_KNN="${PCA_KNN:-20}"
RAY_RADIUS="${RAY_RADIUS:-2.5}"
RAY_JITTER_STD="${RAY_JITTER_STD:-0.05}"
PC_CTX_BANK="${PC_CTX_BANK:-4}"
UDF_PROBE_DELTAS="${UDF_PROBE_DELTAS:-0.01,0.02,0.05}"
MESH_VIS_N_DIRS="${MESH_VIS_N_DIRS:-8}"
MESH_VIS_MAX_T="${MESH_VIS_MAX_T:-2.5}"
MESH_VIS_EPS="${MESH_VIS_EPS:-1e-4}"
STRICT_UDF_SURFACE="${STRICT_UDF_SURFACE:-1}"
SURF_UDF_GRID="${SURF_UDF_GRID:-128}"
SURF_UDF_DILATE="${SURF_UDF_DILATE:-1}"
SURF_UDF_MAX_T="${SURF_UDF_MAX_T:-2.0}"
SURF_UDF_EPS="${SURF_UDF_EPS:-1e-4}"
SURF_UDF_STEPS="${SURF_UDF_STEPS:-64}"
SURF_UDF_TOL="${SURF_UDF_TOL:-1e-4}"
SURF_UDF_MIN_STEP="${SURF_UDF_MIN_STEP:-1e-4}"
AUGMENT_EXISTING="${AUGMENT_EXISTING:-0}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [[ ! -d "${SHAPENET_ROOT}" ]]; then
  echo "[error] ShapeNet root not found: ${SHAPENET_ROOT}"
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${SYNSETS}" ]]; then
  EXTRA_ARGS+=( --synsets "${SYNSETS}" )
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=( --skip_existing )
fi
if [[ "${MISSING_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=( --missing_only )
fi
if [[ "${AUGMENT_EXISTING}" == "1" ]]; then
  EXTRA_ARGS+=( --augment_existing )
fi

set -x
"${PYTHON_BIN}" -m nepa3d.data.preprocess_shapenet_v2 \
  --shapenet_root "${SHAPENET_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --train_ratio "${TRAIN_RATIO}" \
  --seed "${SEED}" \
  --num_workers "${WORKERS}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --n_surf "${N_SURF}" \
  --n_mesh_qry "${N_MESH_QRY}" \
  --n_udf_qry "${N_UDF_QRY}" \
  --n_pc "${N_PC}" \
  --n_pc_qry "${N_PC_QRY}" \
  --n_rays "${N_RAYS}" \
  --pc_ctx_bank "${PC_CTX_BANK}" \
  --pc_view_crop "${PC_VIEW_CROP}" \
  --pc_noise_std "${PC_NOISE_STD}" \
  --pc_dropout "${PC_DROPOUT}" \
  --udf_near_ratio "${UDF_NEAR_RATIO}" \
  --udf_near_std "${UDF_NEAR_STD}" \
  --udf_probe_deltas "${UDF_PROBE_DELTAS}" \
  --curvature_knn "${CURVATURE_KNN}" \
  --pca_knn "${PCA_KNN}" \
  --ray_radius "${RAY_RADIUS}" \
  --ray_jitter_std "${RAY_JITTER_STD}" \
  --mesh_vis_n_dirs "${MESH_VIS_N_DIRS}" \
  --mesh_vis_max_t "${MESH_VIS_MAX_T}" \
  --mesh_vis_eps "${MESH_VIS_EPS}" \
  --strict_udf_surface "${STRICT_UDF_SURFACE}" \
  --surf_udf_grid "${SURF_UDF_GRID}" \
  --surf_udf_dilate "${SURF_UDF_DILATE}" \
  --surf_udf_max_t "${SURF_UDF_MAX_T}" \
  --surf_udf_eps "${SURF_UDF_EPS}" \
  --surf_udf_steps "${SURF_UDF_STEPS}" \
  --surf_udf_tol "${SURF_UDF_TOL}" \
  --surf_udf_min_step "${SURF_UDF_MIN_STEP}" \
  "${EXTRA_ARGS[@]}"
