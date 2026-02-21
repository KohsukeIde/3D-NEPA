#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -W group_list=qgah50055
#PBS -N nepa3d_eval_cls_cpac

set -euo pipefail

WORKDIR="${WORKDIR:-/groups/qgah50055/ide/VGI/3D-NEPA}"
VENV_ACTIVATE="${VENV_ACTIVATE:-.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

RUN_TAG="${RUN_TAG:?set RUN_TAG}"
CKPT="${CKPT:?set CKPT}"

SCAN_CACHE_ROOT="${SCAN_CACHE_ROOT:-data/scanobjectnn_main_split_v2}"
MODELNET_CACHE_ROOT="${MODELNET_CACHE_ROOT:-data/modelnet40_cache_v2}"
UNPAIRED_CACHE_ROOT="${UNPAIRED_CACHE_ROOT:-data/shapenet_unpaired_cache_v1}"

EVAL_ROOT="${EVAL_ROOT:-runs/eval_abcd_1024}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
LOG_ROOT="${LOG_ROOT:-logs/eval/abcd_cls_cpac}"

SEED="${SEED:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SCAN="${BATCH_SCAN:-96}"
BATCH_MODELNET="${BATCH_MODELNET:-128}"
EPOCHS_CLS="${EPOCHS_CLS:-100}"
LR_CLS="${LR_CLS:-1e-4}"
N_POINT_CLS="${N_POINT_CLS:-1024}"
N_RAY_CLS="${N_RAY_CLS:-0}"
ACCELERATE_LAUNCH_MODULE="${ACCELERATE_LAUNCH_MODULE:-accelerate.commands.launch}"

CPAC_SPLIT="${CPAC_SPLIT:-eval}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_HEAD_TRAIN_SPLIT="${CPAC_HEAD_TRAIN_SPLIT:-train_udf}"
CPAC_HEAD_TRAIN_BACKEND="${CPAC_HEAD_TRAIN_BACKEND:-udfgrid}"
CPAC_HEAD_TRAIN_RATIO="${CPAC_HEAD_TRAIN_RATIO:-0.2}"
CPAC_RIDGE_LAMBDA="${CPAC_RIDGE_LAMBDA:-1e-3}"
CPAC_TAU="${CPAC_TAU:-0.03}"
CPAC_EVAL_SEED="${CPAC_EVAL_SEED:-0}"

cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}" "${RESULTS_ROOT}" "${EVAL_ROOT}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== EVAL JOB INFO ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "run_tag=${RUN_TAG}"
echo "ckpt=${CKPT}"
echo "python=$(which python)"
python -V || true
echo

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing ckpt: ${CKPT}"
  exit 2
fi

if [[ ! -d "${SCAN_CACHE_ROOT}" ]]; then
  echo "[error] missing ScanObjectNN cache: ${SCAN_CACHE_ROOT}"
  exit 2
fi

if [[ ! -d "${UNPAIRED_CACHE_ROOT}" ]]; then
  echo "[error] missing unpaired cache for CPAC: ${UNPAIRED_CACHE_ROOT}"
  exit 2
fi

SCAN_SAVE_DIR="${EVAL_ROOT}/classification_scan/${RUN_TAG}"
SCAN_LOG="${LOG_ROOT}/${RUN_TAG}_classification_scan.log"
mkdir -p "${SCAN_SAVE_DIR}"

echo "=== CLASSIFICATION: ScanObjectNN ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m "${ACCELERATE_LAUNCH_MODULE}" \
  --num_processes "${NPROC_PER_NODE}" \
  --num_machines 1 \
  --mixed_precision no \
  -m nepa3d.train.finetune_cls \
  --cache_root "${SCAN_CACHE_ROOT}" \
  --backend pointcloud_noray \
  --ckpt "${CKPT}" \
  --batch "${BATCH_SCAN}" \
  --epochs "${EPOCHS_CLS}" \
  --lr "${LR_CLS}" \
  --n_point "${N_POINT_CLS}" \
  --n_ray "${N_RAY_CLS}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --val_ratio 0.1 \
  --val_seed 0 \
  --eval_seed 0 \
  --mc_eval_k_val 1 \
  --mc_eval_k_test 10 \
  --cls_is_causal 0 \
  --cls_pooling mean_pts \
  --pt_xyz_key pc_xyz \
  --pt_dist_key pt_dist_pool \
  --ablate_point_dist \
  --pt_sample_mode_train fps \
  --pt_sample_mode_eval fps \
  --pt_fps_key auto \
  --pt_rfps_m 4096 \
  --aug_preset scanobjectnn \
  --save_dir "${SCAN_SAVE_DIR}" \
  2>&1 | tee "${SCAN_LOG}"

MODELNET_SAVE_DIR="${EVAL_ROOT}/classification_modelnet/${RUN_TAG}"
MODELNET_LOG="${LOG_ROOT}/${RUN_TAG}_classification_modelnet.log"
if [[ -d "${MODELNET_CACHE_ROOT}" ]]; then
  mkdir -p "${MODELNET_SAVE_DIR}"
  echo "=== CLASSIFICATION: ModelNet40 ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m "${ACCELERATE_LAUNCH_MODULE}" \
    --num_processes "${NPROC_PER_NODE}" \
    --num_machines 1 \
    --mixed_precision no \
    -m nepa3d.train.finetune_cls \
    --cache_root "${MODELNET_CACHE_ROOT}" \
    --backend pointcloud_noray \
    --ckpt "${CKPT}" \
    --batch "${BATCH_MODELNET}" \
    --epochs "${EPOCHS_CLS}" \
    --lr "${LR_CLS}" \
    --n_point "${N_POINT_CLS}" \
    --n_ray "${N_RAY_CLS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --val_ratio 0.1 \
    --val_seed 0 \
    --eval_seed 0 \
    --mc_eval_k_val 1 \
    --mc_eval_k_test 10 \
    --cls_is_causal 0 \
    --cls_pooling mean_pts \
    --pt_xyz_key pc_xyz \
    --pt_dist_key pt_dist_pool \
    --ablate_point_dist \
    --pt_sample_mode_train fps \
    --pt_sample_mode_eval fps \
    --pt_fps_key auto \
    --pt_rfps_m 4096 \
    --aug_preset modelnet40 \
    --save_dir "${MODELNET_SAVE_DIR}" \
    2>&1 | tee "${MODELNET_LOG}"
else
  echo "[skip] ModelNet cache not found: ${MODELNET_CACHE_ROOT}"
fi

CPAC_JSON="${RESULTS_ROOT}/cpac_abcd_1024_${RUN_TAG}.json"
CPAC_LOG="${LOG_ROOT}/${RUN_TAG}_cpac.log"

echo "=== CPAC ==="
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u -m nepa3d.analysis.completion_cpac_udf \
  --cache_root "${UNPAIRED_CACHE_ROOT}" \
  --split "${CPAC_SPLIT}" \
  --ckpt "${CKPT}" \
  --context_backend pointcloud_noray \
  --n_context "${CPAC_N_CONTEXT}" \
  --n_query "${CPAC_N_QUERY}" \
  --max_shapes "${CPAC_MAX_SHAPES}" \
  --head_train_split "${CPAC_HEAD_TRAIN_SPLIT}" \
  --head_train_backend "${CPAC_HEAD_TRAIN_BACKEND}" \
  --head_train_ratio "${CPAC_HEAD_TRAIN_RATIO}" \
  --ridge_lambda "${CPAC_RIDGE_LAMBDA}" \
  --tau "${CPAC_TAU}" \
  --eval_seed "${CPAC_EVAL_SEED}" \
  --disjoint_context_query 1 \
  --context_mode_test normal \
  --rep_source h \
  --query_source pool \
  --out_json "${CPAC_JSON}" \
  2>&1 | tee "${CPAC_LOG}"

echo "=== DONE ==="
echo "scan_cls_last=${SCAN_SAVE_DIR}/last.pt"
if [[ -d "${MODELNET_SAVE_DIR}" ]]; then
  echo "modelnet_cls_last=${MODELNET_SAVE_DIR}/last.pt"
fi
echo "cpac_json=${CPAC_JSON}"
