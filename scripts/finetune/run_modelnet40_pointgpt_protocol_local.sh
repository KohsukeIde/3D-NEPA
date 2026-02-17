#!/usr/bin/env bash
set -eu

# ModelNet40 protocol runner (resume-safe)
#
# Stage A: full fine-tune (all classes)
# Stage B: few-shot episodic (N-way M-shot, trial-averaged)
#
# Defaults are chosen to be reviewer-friendly and easy to resume.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
CACHE_ROOT="${CACHE_ROOT:-data/modelnet40_cache_v2}"
BACKEND="${BACKEND:-pointcloud_noray}"

N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-}"
if [ -z "${N_RAY}" ]; then
  if [ "${BACKEND}" = "pointcloud_noray" ]; then
    N_RAY=0
  else
    N_RAY=256
  fi
fi

BATCH="${BATCH:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
CLS_IS_CAUSAL="${CLS_IS_CAUSAL:-0}"
CLS_POOLING="${CLS_POOLING:-mean_a}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-0}"

# Full fine-tune defaults (PointGPT-like full table side)
FULL_SEEDS="${FULL_SEEDS:-0 1 2}"
FULL_EPOCHS="${FULL_EPOCHS:-100}"
FULL_LR="${FULL_LR:-1e-4}"
FULL_FREEZE_BACKBONE="${FULL_FREEZE_BACKBONE:-0}"
FULL_MC_EVAL_K_VAL="${FULL_MC_EVAL_K_VAL:-1}"
FULL_MC_EVAL_K_TEST="${FULL_MC_EVAL_K_TEST:-10}"

# Few-shot episodic defaults (mainly linear-probe side)
# Format: "N:K" entries separated by spaces.
FEWSHOT_SETTINGS="${FEWSHOT_SETTINGS:-5:10 5:20 10:10 10:20}"
FEWSHOT_TRIALS="${FEWSHOT_TRIALS:-10}"
FEWSHOT_SEED_BASE="${FEWSHOT_SEED_BASE:-0}"
FEWSHOT_EPOCHS="${FEWSHOT_EPOCHS:-100}"
FEWSHOT_LR="${FEWSHOT_LR:-1e-4}"
FEWSHOT_FREEZE_BACKBONE="${FEWSHOT_FREEZE_BACKBONE:-1}"
FEWSHOT_MC_EVAL_K_VAL="${FEWSHOT_MC_EVAL_K_VAL:-1}"
FEWSHOT_MC_EVAL_K_TEST="${FEWSHOT_MC_EVAL_K_TEST:-1}"

METHODS="${METHODS:-scratch shapenet_nepa shapenet_mesh_udf_nepa shapenet_mix_nepa shapenet_mix_mae}"

RUN_ROOT="${RUN_ROOT:-runs/modelnet40_pointgpt_protocol}"
LOG_ROOT="${LOG_ROOT:-logs/finetune/modelnet40_pointgpt_protocol/jobs}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

SHAPENET_NEPA_CKPT="${SHAPENET_NEPA_CKPT:-runs/shapenet_mesh_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MESH_UDF_NEPA_CKPT="${SHAPENET_MESH_UDF_NEPA_CKPT:-runs/shapenet_mesh_udf_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MIX_NEPA_CKPT="${SHAPENET_MIX_NEPA_CKPT:-runs/shapenet_mix_nepa_s0/ckpt_ep049.pt}"
SHAPENET_MIX_MAE_CKPT="${SHAPENET_MIX_MAE_CKPT:-runs/shapenet_mix_mae_s0/ckpt_ep049.pt}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi
if [ ! -d "${CACHE_ROOT}" ]; then
  echo "[error] missing cache root: ${CACHE_ROOT}"
  exit 1
fi

if [ "${N_POINT}" -gt 256 ]; then
  echo "[warn] N_POINT=${N_POINT} requested."
  echo "[warn] Existing pretrained checkpoints in this repo were mostly trained with n_point=256."
  echo "[warn] finetune_cls will clamp n_point down to checkpoint pretrain value for those methods."
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
  local out_dir="${RUN_ROOT}/scratch_seed${seed}_np${N_POINT}_nr${N_RAY}_init"
  local ckpt="${out_dir}/ckpt_ep000.pt"
  mkdir -p "${out_dir}"
  if [ -f "${ckpt}" ]; then
    echo "${ckpt}"
    return 0
  fi
  "${PYTHON_BIN}" - "${seed}" "${ckpt}" "${N_POINT}" "${N_RAY}" <<'PY'
import os
import sys
import torch
from nepa3d.models.query_nepa import QueryNepa
from nepa3d.utils.seed import set_seed

seed = int(sys.argv[1])
ckpt_path = sys.argv[2]
n_point = int(sys.argv[3])
n_ray = int(sys.argv[4])

set_seed(seed)
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

ckpt_for_method() {
  local method="$1"
  local seed="$2"
  case "${method}" in
    scratch)
      make_scratch_ckpt "${seed}"
      ;;
    shapenet_nepa)
      echo "${SHAPENET_NEPA_CKPT}"
      ;;
    shapenet_mesh_udf_nepa)
      echo "${SHAPENET_MESH_UDF_NEPA_CKPT}"
      ;;
    shapenet_mix_nepa)
      echo "${SHAPENET_MIX_NEPA_CKPT}"
      ;;
    shapenet_mix_mae)
      echo "${SHAPENET_MIX_MAE_CKPT}"
      ;;
    *)
      echo "[error] unknown method: ${method}" >&2
      exit 1
      ;;
  esac
}

declare -a JOBS
add_job() {
  local phase="$1"       # full|fewshot
  local method="$2"
  local seed="$3"
  local n_way="$4"
  local k_shot="$5"
  local few_seed="$6"
  local way_seed="$7"
  local ckpt="$8"
  local epochs="$9"
  local lr="${10}"
  local mc_val="${11}"
  local mc_test="${12}"
  local freeze="${13}"

  local suffix=""
  if [ "${freeze}" = "1" ]; then
    suffix="_lp"
  fi

  local save_dir job_log
  if [ "${phase}" = "full" ]; then
    save_dir="${RUN_ROOT}/full/scan_${method}${suffix}_s${seed}"
    job_log="${LOG_ROOT}/full_${method}${suffix}_s${seed}.log"
  else
    save_dir="${RUN_ROOT}/fewshot/scan_${method}${suffix}_N${n_way}_K${k_shot}_t${few_seed}"
    job_log="${LOG_ROOT}/fewshot_${method}${suffix}_N${n_way}_K${k_shot}_t${few_seed}.log"
  fi
  JOBS+=("${phase}|${method}|${seed}|${n_way}|${k_shot}|${few_seed}|${way_seed}|${ckpt}|${save_dir}|${job_log}|${epochs}|${lr}|${mc_val}|${mc_test}|${freeze}")
}

# Build full jobs
for method in ${METHODS}; do
  for seed in ${FULL_SEEDS}; do
    ckpt="$(ckpt_for_method "${method}" "${seed}")"
    add_job "full" "${method}" "${seed}" 0 0 0 0 "${ckpt}" "${FULL_EPOCHS}" "${FULL_LR}" "${FULL_MC_EVAL_K_VAL}" "${FULL_MC_EVAL_K_TEST}" "${FULL_FREEZE_BACKBONE}"
  done
done

# Build few-shot episodic jobs
for method in ${METHODS}; do
  for setting in ${FEWSHOT_SETTINGS}; do
    n_way="${setting%%:*}"
    k_shot="${setting##*:}"
    for ((t = 0; t < FEWSHOT_TRIALS; t++)); do
      trial_seed=$((FEWSHOT_SEED_BASE + t))
      ckpt="$(ckpt_for_method "${method}" "${trial_seed}")"
      add_job "fewshot" "${method}" "${trial_seed}" "${n_way}" "${k_shot}" "${trial_seed}" "${trial_seed}" "${ckpt}" "${FEWSHOT_EPOCHS}" "${FEWSHOT_LR}" "${FEWSHOT_MC_EVAL_K_VAL}" "${FEWSHOT_MC_EVAL_K_TEST}" "${FEWSHOT_FREEZE_BACKBONE}"
    done
  done
done

run_one() {
  local gpu="$1"
  local spec="$2"
  IFS="|" read -r phase method seed n_way k_shot few_seed way_seed ckpt save_dir job_log epochs lr mc_val mc_test freeze <<<"${spec}"

  mkdir -p "${save_dir}" "$(dirname "${job_log}")"

  if [ -f "${save_dir}/last.pt" ]; then
    echo "[skip][gpu${gpu}] ${phase} ${method} seed=${seed} N=${n_way} K=${k_shot} (last.pt exists)" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
    return 0
  fi

  echo "[start][gpu${gpu}] $(date '+%F %T') ${phase} ${method} seed=${seed} N=${n_way} K=${k_shot}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"

  local extra_args=()
  if [ "${freeze}" = "1" ]; then
    extra_args+=(--freeze_backbone)
  fi
  if [ "${ABLATE_POINT_DIST}" = "1" ]; then
    extra_args+=(--ablate_point_dist)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.finetune_cls \
    --cache_root "${CACHE_ROOT}" \
    --backend "${BACKEND}" \
    --ckpt "${ckpt}" \
    --batch "${BATCH}" \
    --epochs "${epochs}" \
    --lr "${lr}" \
    --n_point "${N_POINT}" \
    --n_ray "${N_RAY}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${seed}" \
    --fewshot_n_way "${n_way}" \
    --fewshot_k "${k_shot}" \
    --fewshot_seed "${few_seed}" \
    --fewshot_way_seed "${way_seed}" \
    --val_ratio "${VAL_RATIO}" \
    --val_seed "${VAL_SEED}" \
    --eval_seed "${EVAL_SEED}" \
    --mc_eval_k_val "${mc_val}" \
    --mc_eval_k_test "${mc_test}" \
    --cls_is_causal "${CLS_IS_CAUSAL}" \
    --cls_pooling "${CLS_POOLING}" \
    "${extra_args[@]}" \
    --save_dir "${save_dir}" \
    > "${job_log}" 2>&1
  rc=$?

  if [ "${rc}" -eq 0 ]; then
    echo "[done ][gpu${gpu}] $(date '+%F %T') ${phase} ${method} seed=${seed} N=${n_way} K=${k_shot}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  else
    echo "[fail ][gpu${gpu}] $(date '+%F %T') ${phase} ${method} seed=${seed} N=${n_way} K=${k_shot} rc=${rc}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  fi
  return "${rc}"
}

worker_static() {
  local gpu="$1"
  local start_idx="$2"
  local i="${start_idx}"
  local total="${#JOBS[@]}"
  while [ "${i}" -lt "${total}" ]; do
    run_one "${gpu}" "${JOBS[${i}]}"
    i=$((i + 2))
  done
}

next_job_index() {
  local total="$1"
  local out=""
  exec 9>>"${JOB_LOCK_FILE}"
  flock 9
  local cur
  cur="$(cat "${JOB_INDEX_FILE}")"
  if [ "${cur}" -lt "${total}" ]; then
    out="${cur}"
    echo $((cur + 1)) > "${JOB_INDEX_FILE}"
  fi
  flock -u 9
  exec 9>&-
  echo "${out}"
}

worker_dynamic() {
  local gpu="$1"
  local total="${#JOBS[@]}"
  while true; do
    local idx
    idx="$(next_job_index "${total}")"
    if [ -z "${idx}" ]; then
      break
    fi
    run_one "${gpu}" "${JOBS[${idx}]}"
  done
}

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

echo "[info] total_jobs=${#JOBS[@]}"
echo "[info] cache_root=${CACHE_ROOT} backend=${BACKEND} n_point=${N_POINT} n_ray=${N_RAY}"
echo "[info] cls_is_causal=${CLS_IS_CAUSAL}"
echo "[info] cls_pooling=${CLS_POOLING}"
echo "[info] ablate_point_dist=${ABLATE_POINT_DIST}"
echo "[info] methods=${METHODS}"
echo "[info] full_seeds=${FULL_SEEDS} full_freeze=${FULL_FREEZE_BACKBONE}"
echo "[info] fewshot_settings=${FEWSHOT_SETTINGS} trials=${FEWSHOT_TRIALS} few_freeze=${FEWSHOT_FREEZE_BACKBONE}"
echo "[info] logs=${LOG_ROOT}"

JOB_INDEX_FILE="$(mktemp)"
JOB_LOCK_FILE="$(mktemp)"
echo "0" > "${JOB_INDEX_FILE}"
cleanup_scheduler() {
  rm -f "${JOB_INDEX_FILE}" "${JOB_LOCK_FILE}"
}
trap cleanup_scheduler EXIT

if ! command -v flock >/dev/null 2>&1; then
  echo "[warn] flock not found; fallback to static split scheduling"
  worker_static "${GPU0}" 0 &
  pid0=$!
  worker_static "${GPU1}" 1 &
  pid1=$!
else
  worker_dynamic "${GPU0}" &
  pid0=$!
  worker_dynamic "${GPU1}" &
  pid1=$!
fi

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
