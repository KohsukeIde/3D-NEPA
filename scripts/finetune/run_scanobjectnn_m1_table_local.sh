#!/usr/bin/env bash
set -eu

# M1 main table on ScanObjectNN:
#   - scratch
#   - shapenet_nepa               (mesh-only pretrain)
#   - shapenet_mesh_udf_nepa      (mesh+UDF pretrain)
#   - shapenet_mix_nepa           (mesh+UDF+realPC mixed pretrain)
#   - shapenet_mix_mae            (same mixed data with MAE objective)
# Seeds: 0,1,2
# K-shot: 0,1,5,10,20

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
# Paper-safe default: use split-specific cache generated from h5_files/main_split.
# Legacy/internal tables may still use data/scanobjectnn_cache_v2 explicitly.
CACHE_ROOT="${CACHE_ROOT:-data/scanobjectnn_main_split_v2}"
BACKEND="${BACKEND:-pointcloud_noray}"

BATCH="${BATCH:-128}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
MC_EVAL_K="${MC_EVAL_K:-4}"
MC_EVAL_K_VAL="${MC_EVAL_K_VAL:-1}"
MC_EVAL_K_TEST="${MC_EVAL_K_TEST:-${MC_EVAL_K}}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

LOG_ROOT="${LOG_ROOT:-logs/finetune/scan_m1_table/jobs}"
RUN_ROOT="${RUN_ROOT:-runs}"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

SHAPENET_NEPA_CKPT="${SHAPENET_NEPA_CKPT:-runs/shapenet_mesh_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MESH_UDF_NEPA_CKPT="${SHAPENET_MESH_UDF_NEPA_CKPT:-runs/shapenet_mesh_udf_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MIX_NEPA_CKPT="${SHAPENET_MIX_NEPA_CKPT:-runs/shapenet_mix_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MIX_MAE_CKPT="${SHAPENET_MIX_MAE_CKPT:-runs/shapenet_mix_mae_s0/ckpt_ep049.pt}"
METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"

SEEDS="${SEEDS:-0 1 2}"
K_LIST="${K_LIST:-0 1 5 10 20}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

if [ ! -d "${CACHE_ROOT}" ]; then
  echo "[error] missing cache root: ${CACHE_ROOT}"
  exit 1
fi

has_method() {
  local target="$1"
  for m in ${METHODS}; do
    if [ "${m}" = "${target}" ]; then
      return 0
    fi
  done
  return 1
}

if has_method "shapenet_nepa" && [ ! -f "${SHAPENET_NEPA_CKPT}" ]; then
  echo "[error] missing ckpt: ${SHAPENET_NEPA_CKPT}"
  exit 1
fi
if has_method "shapenet_mesh_udf_nepa" && [ ! -f "${SHAPENET_MESH_UDF_NEPA_CKPT}" ]; then
  echo "[error] missing ckpt: ${SHAPENET_MESH_UDF_NEPA_CKPT}"
  exit 1
fi
if has_method "shapenet_mix_nepa" && [ ! -f "${SHAPENET_MIX_NEPA_CKPT}" ]; then
  echo "[error] missing ckpt: ${SHAPENET_MIX_NEPA_CKPT}"
  exit 1
fi
if has_method "shapenet_mix_mae" && [ ! -f "${SHAPENET_MIX_MAE_CKPT}" ]; then
  echo "[error] missing ckpt: ${SHAPENET_MIX_MAE_CKPT}"
  exit 1
fi

make_scratch_ckpt() {
  local seed="$1"
  local out_dir="${RUN_ROOT}/scratch_seed${seed}_init"
  local ckpt="${out_dir}/ckpt_ep000.pt"
  mkdir -p "${out_dir}"
  if [ -f "${ckpt}" ]; then
    echo "${ckpt}"
    return 0
  fi
  "${PYTHON_BIN}" - "${seed}" "${ckpt}" <<'PY'
import os
import sys
import torch
from nepa3d.models.query_nepa import QueryNepa
from nepa3d.utils.seed import set_seed

seed = int(sys.argv[1])
ckpt_path = sys.argv[2]

set_seed(seed)
n_point = 256
n_ray = 256
d_model = 384
layers = 8
heads = 6
add_eos = True
n_types = 5
t = 1 + n_point + n_ray + (1 if add_eos else 0)

model = QueryNepa(
    feat_dim=15,
    d_model=d_model,
    n_types=n_types,
    nhead=heads,
    num_layers=layers,
    max_len=t,
)
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
torch.save(
    {
        "model": model.state_dict(),
        "args": {
            "n_point": n_point,
            "n_ray": n_ray,
            "d_model": d_model,
            "layers": layers,
            "heads": heads,
            "add_eos": int(add_eos),
            "n_types": n_types,
            "backend": "scratch",
            "seed": seed,
        },
        "epoch": 0,
    },
    ckpt_path,
)
print(ckpt_path)
PY
}

declare -a JOBS
add_job() {
  local method="$1"
  local seed="$2"
  local k="$3"
  local ckpt="$4"
  local fewshot_seed="$5"
  local save_dir="${RUN_ROOT}/scan_${method}_k${k}_s${seed}"
  local job_log="${LOG_ROOT}/${method}_k${k}_s${seed}.log"
  JOBS+=("${method}|${seed}|${k}|${ckpt}|${fewshot_seed}|${save_dir}|${job_log}")
}

for seed in ${SEEDS}; do
  scratch_ckpt="$(make_scratch_ckpt "${seed}")"
  for k in ${K_LIST}; do
    fs="${seed}"
    if [ "${k}" = "0" ]; then
      fs="0"
    fi
    if has_method "scratch"; then
      add_job "scratch" "${seed}" "${k}" "${scratch_ckpt}" "${fs}"
    fi
    if has_method "shapenet_nepa"; then
      add_job "shapenet_nepa" "${seed}" "${k}" "${SHAPENET_NEPA_CKPT}" "${fs}"
    fi
    if has_method "shapenet_mesh_udf_nepa"; then
      add_job "shapenet_mesh_udf_nepa" "${seed}" "${k}" "${SHAPENET_MESH_UDF_NEPA_CKPT}" "${fs}"
    fi
    if has_method "shapenet_mix_nepa"; then
      add_job "shapenet_mix_nepa" "${seed}" "${k}" "${SHAPENET_MIX_NEPA_CKPT}" "${fs}"
    fi
    if has_method "shapenet_mix_mae"; then
      add_job "shapenet_mix_mae" "${seed}" "${k}" "${SHAPENET_MIX_MAE_CKPT}" "${fs}"
    fi
  done
done

run_one() {
  local gpu="$1"
  local spec="$2"
  IFS="|" read -r method seed k ckpt fewshot_seed save_dir job_log <<<"${spec}"
  mkdir -p "${save_dir}" "$(dirname "${job_log}")"

  if [ -f "${save_dir}/last.pt" ]; then
    echo "[skip][gpu${gpu}] ${method} k=${k} seed=${seed} (last.pt exists)" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
    return 0
  fi

  echo "[start][gpu${gpu}] $(date '+%F %T') ${method} k=${k} seed=${seed}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.finetune_cls \
    --cache_root "${CACHE_ROOT}" \
    --backend "${BACKEND}" \
    --ckpt "${ckpt}" \
    --batch "${BATCH}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --n_point "${N_POINT}" \
    --n_ray "${N_RAY}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --fewshot_k "${k}" \
    --fewshot_seed "${fewshot_seed}" \
    --val_ratio "${VAL_RATIO}" \
    --val_seed "${VAL_SEED}" \
    --eval_seed "${EVAL_SEED}" \
    --mc_eval_k "${MC_EVAL_K}" \
    --mc_eval_k_val "${MC_EVAL_K_VAL}" \
    --mc_eval_k_test "${MC_EVAL_K_TEST}" \
    --save_dir "${save_dir}" \
    > "${job_log}" 2>&1
  rc=$?

  if [ "${rc}" -eq 0 ]; then
    echo "[done ][gpu${gpu}] $(date '+%F %T') ${method} k=${k} seed=${seed}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  else
    echo "[fail ][gpu${gpu}] $(date '+%F %T') ${method} k=${k} seed=${seed} rc=${rc}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  fi
  return "${rc}"
}

worker() {
  local gpu="$1"
  local start_idx="$2"
  local i="${start_idx}"
  local total="${#JOBS[@]}"
  while [ "${i}" -lt "${total}" ]; do
    run_one "${gpu}" "${JOBS[${i}]}"
    i=$((i + 2))
  done
}

echo "[info] total_jobs=${#JOBS[@]}"
echo "[info] gpu0=${GPU0} gpu1=${GPU1}"
echo "[info] logs=${LOG_ROOT}"
echo "[info] batch=${BATCH} workers=${NUM_WORKERS}"
echo "[info] methods=${METHODS}"

worker "${GPU0}" 0 &
pid0=$!
worker "${GPU1}" 1 &
pid1=$!

echo "[info] worker_pid_gpu${GPU0}=${pid0}"
echo "[info] worker_pid_gpu${GPU1}=${pid1}"

wait "${pid0}"
rc0=$?
wait "${pid1}"
rc1=$?

echo "[info] completed rc_gpu${GPU0}=${rc0} rc_gpu${GPU1}=${rc1}"
if [ "${rc0}" -ne 0 ] || [ "${rc1}" -ne 0 ]; then
  exit 1
fi
