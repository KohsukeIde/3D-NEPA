#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N nepa3d_eval_cls_cpac

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
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
NUM_PROCESSES="${NUM_PROCESSES:-${NPROC_PER_NODE}}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
BATCH_SCAN="${BATCH_SCAN:-96}"
BATCH_MODELNET="${BATCH_MODELNET:-128}"
EPOCHS_CLS="${EPOCHS_CLS:-100}"
LR_CLS="${LR_CLS:-1e-4}"
N_POINT_CLS="${N_POINT_CLS:-1024}"
N_RAY_CLS="${N_RAY_CLS:-0}"
ACCELERATE_LAUNCH_MODULE="${ACCELERATE_LAUNCH_MODULE:-accelerate.commands.launch}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
CLS_POOLING="${CLS_POOLING:-mean_q}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-1}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
WARMUP_START_FACTOR="${WARMUP_START_FACTOR:-0.1}"
MIN_LR="${MIN_LR:-1e-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SCAN_AUG_PRESET="${SCAN_AUG_PRESET:-scanobjectnn}"
MODELNET_AUG_PRESET="${MODELNET_AUG_PRESET:-modelnet40}"
RUN_SCAN="${RUN_SCAN:-1}"
RUN_MODELNET="${RUN_MODELNET:-1}"
RUN_CPAC="${RUN_CPAC:-1}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-10}"

CPAC_SPLIT="${CPAC_SPLIT:-eval}"
CPAC_N_CONTEXT="${CPAC_N_CONTEXT:-1024}"
CPAC_N_QUERY="${CPAC_N_QUERY:-1024}"
CPAC_MAX_SHAPES="${CPAC_MAX_SHAPES:-800}"
CPAC_MAX_LEN="${CPAC_MAX_LEN:--1}"
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

if [[ "${NUM_MACHINES}" -gt 1 ]]; then
  echo "=== MULTI-NODE DDP ==="
  echo "num_processes=${NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK}"
  echo "main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT}"
  echo
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing ckpt: ${CKPT}"
  exit 2
fi

if [[ "${RUN_SCAN}" == "1" ]] && [[ ! -d "${SCAN_CACHE_ROOT}" ]]; then
  echo "[error] missing ScanObjectNN cache: ${SCAN_CACHE_ROOT}"
  exit 2
fi

if [[ "${RUN_CPAC}" == "1" ]] && [[ ! -d "${UNPAIRED_CACHE_ROOT}" ]]; then
  echo "[error] missing unpaired cache for CPAC: ${UNPAIRED_CACHE_ROOT}"
  exit 2
fi

# Only rank0 performs CPAC in multi-node mode.
RUN_CPAC_LOCAL="${RUN_CPAC}"
if [[ "${NUM_MACHINES}" -gt 1 ]] && [[ "${MACHINE_RANK}" != "0" ]]; then
  RUN_CPAC_LOCAL="0"
fi

if [[ "${RUN_CPAC_LOCAL}" == "1" ]]; then
  # Fail fast when CPAC token length is incompatible with checkpoint max_len.
  python - "${CKPT}" "${CPAC_N_CONTEXT}" "${CPAC_N_QUERY}" "${CPAC_MAX_LEN}" <<'PY'
import sys
import torch

ckpt_path = sys.argv[1]
n_context = int(sys.argv[2])
n_query = int(sys.argv[3])
max_len_override = int(sys.argv[4])
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model"]
pre_args = ckpt.get("args", {})
ckpt_n_types = int(state["type_emb.weight"].shape[0])
qa_tokens = bool(pre_args.get("qa_tokens", ckpt_n_types >= 9))
add_eos = bool(pre_args.get("add_eos", ckpt_n_types >= 5))
ckpt_max_len = int(state["pos_emb"].shape[1])
max_len = ckpt_max_len if max_len_override < 0 else max_len_override
required = 1 + (2 if qa_tokens else 1) * (n_context + n_query) + (1 if add_eos else 0)
if required > max_len:
    raise SystemExit(
        f"[error] CPAC precheck failed: required_seq_len={required} > effective_max_len={max_len} "
        f"(ckpt_max_len={ckpt_max_len}, qa_tokens={int(qa_tokens)}, add_eos={int(add_eos)}, "
        f"n_context={n_context}, n_query={n_query}, max_len_override={max_len_override})."
    )
print(
    f"[ok] CPAC precheck: required_seq_len={required} <= effective_max_len={max_len} "
    f"(ckpt_max_len={ckpt_max_len}, qa_tokens={int(qa_tokens)}, add_eos={int(add_eos)})"
)
PY
else
  echo "[skip] CPAC precheck disabled (RUN_CPAC=${RUN_CPAC}, RUN_CPAC_LOCAL=${RUN_CPAC_LOCAL})"
fi

ABLATE_POINT_DIST_FLAG=()
if [[ "${ABLATE_POINT_DIST}" == "1" ]]; then
  ABLATE_POINT_DIST_FLAG+=(--ablate_point_dist)
fi

ACCELERATE_DDP_ARGS=()
if [[ "${NUM_MACHINES}" -gt 1 ]]; then
  ACCELERATE_DDP_ARGS=(
    --multi_gpu
    --num_processes "${NUM_PROCESSES}"
    --num_machines "${NUM_MACHINES}"
    --machine_rank "${MACHINE_RANK}"
    --main_process_ip "${MAIN_PROCESS_IP}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    --mixed_precision "${MIXED_PRECISION}"
  )
else
  ACCELERATE_DDP_ARGS=(
    --num_processes "${NPROC_PER_NODE}"
    --num_machines 1
    --mixed_precision "${MIXED_PRECISION}"
  )
fi

SCAN_SAVE_DIR="${EVAL_ROOT}/classification_scan/${RUN_TAG}"
if [[ "${NUM_MACHINES}" -gt 1 ]] && [[ "${MACHINE_RANK}" != "0" ]]; then
  SCAN_LOG="${LOG_ROOT}/${RUN_TAG}_classification_scan.rank${MACHINE_RANK}.log"
else
  SCAN_LOG="${LOG_ROOT}/${RUN_TAG}_classification_scan.log"
fi
if [[ "${RUN_SCAN}" == "1" ]]; then
  mkdir -p "${SCAN_SAVE_DIR}"
  echo "=== CLASSIFICATION: ScanObjectNN ==="
  echo "cls_pooling=${CLS_POOLING} ablate_point_dist=${ABLATE_POINT_DIST} point_order_mode=${POINT_ORDER_MODE} aug_preset=${SCAN_AUG_PRESET}"
  echo "lr_scheduler=${LR_SCHEDULER} warmup_epochs=${WARMUP_EPOCHS} min_lr=${MIN_LR} grad_accum_steps=${GRAD_ACCUM_STEPS} max_grad_norm=${MAX_GRAD_NORM}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m "${ACCELERATE_LAUNCH_MODULE}" \
    "${ACCELERATE_DDP_ARGS[@]}" \
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
    --mc_eval_k_val "${MC_EVAL_K_VAL}" \
    --mc_eval_k_test "${MC_EVAL_K_TEST}" \
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
    --cls_is_causal 0 \
    --cls_pooling "${CLS_POOLING}" \
    --pt_xyz_key pc_xyz \
    --pt_dist_key pt_dist_pool \
    "${ABLATE_POINT_DIST_FLAG[@]}" \
    --pt_sample_mode_train fps \
    --pt_sample_mode_eval fps \
    --pt_fps_key auto \
    --pt_rfps_m 4096 \
    --point_order_mode "${POINT_ORDER_MODE}" \
    --lr_scheduler "${LR_SCHEDULER}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --warmup_start_factor "${WARMUP_START_FACTOR}" \
    --min_lr "${MIN_LR}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --aug_preset "${SCAN_AUG_PRESET}" \
    --save_dir "${SCAN_SAVE_DIR}" \
    2>&1 | tee "${SCAN_LOG}"
else
  echo "[skip] ScanObjectNN classification disabled (RUN_SCAN=${RUN_SCAN})"
fi

MODELNET_SAVE_DIR="${EVAL_ROOT}/classification_modelnet/${RUN_TAG}"
if [[ "${NUM_MACHINES}" -gt 1 ]] && [[ "${MACHINE_RANK}" != "0" ]]; then
  MODELNET_LOG="${LOG_ROOT}/${RUN_TAG}_classification_modelnet.rank${MACHINE_RANK}.log"
else
  MODELNET_LOG="${LOG_ROOT}/${RUN_TAG}_classification_modelnet.log"
fi
if [[ "${RUN_MODELNET}" == "1" ]] && [[ -d "${MODELNET_CACHE_ROOT}" ]]; then
  mkdir -p "${MODELNET_SAVE_DIR}"
  echo "=== CLASSIFICATION: ModelNet40 ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m "${ACCELERATE_LAUNCH_MODULE}" \
    "${ACCELERATE_DDP_ARGS[@]}" \
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
    --mc_eval_k_val "${MC_EVAL_K_VAL}" \
    --mc_eval_k_test "${MC_EVAL_K_TEST}" \
    --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
    --cls_is_causal 0 \
    --cls_pooling "${CLS_POOLING}" \
    --pt_xyz_key pc_xyz \
    --pt_dist_key pt_dist_pool \
    "${ABLATE_POINT_DIST_FLAG[@]}" \
    --pt_sample_mode_train fps \
    --pt_sample_mode_eval fps \
    --pt_fps_key auto \
    --pt_rfps_m 4096 \
    --point_order_mode "${POINT_ORDER_MODE}" \
    --lr_scheduler "${LR_SCHEDULER}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --warmup_start_factor "${WARMUP_START_FACTOR}" \
    --min_lr "${MIN_LR}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --aug_preset "${MODELNET_AUG_PRESET}" \
    --save_dir "${MODELNET_SAVE_DIR}" \
    2>&1 | tee "${MODELNET_LOG}"
else
  if [[ "${RUN_MODELNET}" != "1" ]]; then
    echo "[skip] ModelNet40 classification disabled (RUN_MODELNET=${RUN_MODELNET})"
  else
    echo "[skip] ModelNet cache not found: ${MODELNET_CACHE_ROOT}"
  fi
fi

CPAC_JSON="${RESULTS_ROOT}/cpac_abcd_1024_${RUN_TAG}.json"
CPAC_LOG="${LOG_ROOT}/${RUN_TAG}_cpac.log"

if [[ "${RUN_CPAC_LOCAL}" == "1" ]]; then
  echo "=== CPAC ==="
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m nepa3d.analysis.completion_cpac_udf \
    --cache_root "${UNPAIRED_CACHE_ROOT}" \
    --split "${CPAC_SPLIT}" \
    --ckpt "${CKPT}" \
    --context_backend pointcloud_noray \
    --n_context "${CPAC_N_CONTEXT}" \
    --n_query "${CPAC_N_QUERY}" \
    --max_len "${CPAC_MAX_LEN}" \
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
else
  echo "[skip] CPAC disabled (RUN_CPAC=${RUN_CPAC}, RUN_CPAC_LOCAL=${RUN_CPAC_LOCAL})"
fi

echo "=== DONE ==="
if [[ "${RUN_SCAN}" == "1" ]]; then
  echo "scan_cls_last=${SCAN_SAVE_DIR}/last.pt"
fi
if [[ "${RUN_MODELNET}" == "1" ]] && [[ -d "${MODELNET_SAVE_DIR}" ]]; then
  echo "modelnet_cls_last=${MODELNET_SAVE_DIR}/last.pt"
fi
if [[ "${RUN_CPAC_LOCAL}" == "1" ]]; then
  echo "cpac_json=${CPAC_JSON}"
fi
