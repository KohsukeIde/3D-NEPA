#!/usr/bin/env bash
set -eu

# Simple baseline: replace pretrain corpus from ModelNet40 to ShapeNet cache only.
# This runs mesh-only pretraining with NEPA and MAE objectives.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/shapenet_cache_v0}"
SEED="${SEED:-0}"

BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"

MASK_RATIO="${MASK_RATIO:-0.4}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

SAVE_NEPA="${SAVE_NEPA:-runs/shapenet_mesh_nepa_s0}"
SAVE_MAE="${SAVE_MAE:-runs/shapenet_mesh_mae_s0}"
LOG_NEPA="${LOG_NEPA:-logs/pretrain/shapenet_mesh/nepa_s0.log}"
LOG_MAE="${LOG_MAE:-logs/pretrain/shapenet_mesh/mae_s0.log}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${CACHE_ROOT}/train" ]; then
  echo "[error] cache not found: ${CACHE_ROOT}/train"
  echo "        run ShapeNet preprocess first."
  exit 1
fi

mkdir -p "$(dirname "${LOG_NEPA}")" "$(dirname "${LOG_MAE}")" "${SAVE_NEPA}" "${SAVE_MAE}"

nohup env CUDA_VISIBLE_DEVICES="${GPU0}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --backend mesh \
  --objective nepa \
  --batch "${BATCH}" --epochs "${EPOCHS}" --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
  --save_dir "${SAVE_NEPA}" \
  > "${LOG_NEPA}" 2>&1 &
echo $! > "${LOG_NEPA}.pid"

nohup env CUDA_VISIBLE_DEVICES="${GPU1}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
  --cache_root "${CACHE_ROOT}" \
  --backend mesh \
  --objective mae \
  --mask_ratio "${MASK_RATIO}" \
  --batch "${BATCH}" --epochs "${EPOCHS}" --lr "${LR}" \
  --n_point "${N_POINT}" --n_ray "${N_RAY}" \
  --d_model "${D_MODEL}" --layers "${LAYERS}" --heads "${HEADS}" \
  --num_workers "${NUM_WORKERS}" --seed "${SEED}" \
  --save_dir "${SAVE_MAE}" \
  > "${LOG_MAE}" 2>&1 &
echo $! > "${LOG_MAE}.pid"

echo "nepa_pid=$(cat "${LOG_NEPA}.pid")"
echo "mae_pid=$(cat "${LOG_MAE}.pid")"
