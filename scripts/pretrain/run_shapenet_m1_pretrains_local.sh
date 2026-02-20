#!/usr/bin/env bash
set -eu

# M1 pretrains on 2 GPUs:
#   1) mesh+UDF (NEPA)
#   2) mesh+UDF+realPC (NEPA)
#   3) mesh+UDF+realPC (MAE)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
# 2D NEPA style linear LR scaling:
#   LEARNING_RATE = BASE_LEARNING_RATE * TOTAL_BATCH_SIZE / 256
# Single-GPU local runner uses TOTAL_BATCH_SIZE=BATCH by default.
LR_SCALE_ENABLE="${LR_SCALE_ENABLE:-1}"
LR_SCALE_REF_BATCH="${LR_SCALE_REF_BATCH:-256}"
LR_BASE_TOTAL_BATCH="${LR_BASE_TOTAL_BATCH:-32}"
BASE_LEARNING_RATE="${BASE_LEARNING_RATE:-}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-${BATCH}}"
N_POINT="${N_POINT:-256}"
N_RAY="${N_RAY:-256}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-6}"
NUM_WORKERS="${NUM_WORKERS:-6}"
SEED="${SEED:-0}"

MIX_NUM_SAMPLES="${MIX_NUM_SAMPLES:-200000}"
MIX_SEED="${MIX_SEED:-0}"
MASK_RATIO="${MASK_RATIO:-0.4}"
DROP_RAY_PROB="${DROP_RAY_PROB:-0.3}"
SAVE_EVERY="${SAVE_EVERY:-10}"
SAVE_LAST="${SAVE_LAST:-1}"
AUTO_RESUME="${AUTO_RESUME:-1}"

CFG_MESH_UDF="${CFG_MESH_UDF:-nepa3d/configs/pretrain_mixed_shapenet_mesh_udf.yaml}"
CFG_MIX="${CFG_MIX:-nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan.yaml}"

RUN_ROOT="${RUN_ROOT:-runs}"
LOG_ROOT="${LOG_ROOT:-logs/pretrain/m1}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi

for f in "${CFG_MESH_UDF}" "${CFG_MIX}"; do
  if [ ! -f "${f}" ]; then
    echo "[error] missing mix config: ${f}"
    exit 1
  fi
done

if [ "${LR_SCALE_ENABLE}" = "1" ]; then
  if [ -z "${BASE_LEARNING_RATE}" ]; then
    BASE_LEARNING_RATE="$("${PYTHON_BIN}" -c "print(float('${LR}') * float('${LR_BASE_TOTAL_BATCH}') / 256.0)")"
  fi
  LR="$("${PYTHON_BIN}" -c "print(float('${BASE_LEARNING_RATE}') * float('${TOTAL_BATCH_SIZE}') / float('${LR_SCALE_REF_BATCH}'))")"
  echo "[lr-scale] enabled: base_lr=${BASE_LEARNING_RATE} total_batch=${TOTAL_BATCH_SIZE} ref_batch=${LR_SCALE_REF_BATCH} lr=${LR}"
else
  echo "[lr-scale] disabled: lr=${LR}"
fi

declare -a JOBS
add_job() {
  local name="$1"
  local objective="$2"
  local mix_cfg="$3"
  local drop_ray="$4"
  local save_dir="$5"
  local log_file="$6"
  JOBS+=("${name}|${objective}|${mix_cfg}|${drop_ray}|${save_dir}|${log_file}")
}

add_job \
  "shapenet_mesh_udf_nepa_s0" "nepa" "${CFG_MESH_UDF}" "${DROP_RAY_PROB}" \
  "${RUN_ROOT}/shapenet_mesh_udf_nepa_s0" "${LOG_ROOT}/shapenet_mesh_udf_nepa_s0.log"

add_job \
  "shapenet_mix_nepa_s0" "nepa" "${CFG_MIX}" "${DROP_RAY_PROB}" \
  "${RUN_ROOT}/shapenet_mix_nepa_s0" "${LOG_ROOT}/shapenet_mix_nepa_s0.log"

add_job \
  "shapenet_mix_mae_s0" "mae" "${CFG_MIX}" "0.0" \
  "${RUN_ROOT}/shapenet_mix_mae_s0" "${LOG_ROOT}/shapenet_mix_mae_s0.log"

run_one() {
  local gpu="$1"
  local spec="$2"
  IFS="|" read -r name objective mix_cfg drop_ray save_dir log_file <<<"${spec}"
  mkdir -p "${save_dir}" "$(dirname "${log_file}")"

  local final_ep
  final_ep="$(printf '%03d' $((EPOCHS - 1)))"
  local final_ckpt="${save_dir}/ckpt_ep${final_ep}.pt"

  if [ -f "${final_ckpt}" ]; then
    echo "[skip][gpu${gpu}] ${name} (${final_ckpt} exists)" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
    return 0
  fi

  echo "[start][gpu${gpu}] $(date '+%F %T') ${name}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"

  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
    --mix_config "${mix_cfg}" \
    --mix_num_samples "${MIX_NUM_SAMPLES}" \
    --mix_seed "${MIX_SEED}" \
    --objective "${objective}" \
    --mask_ratio "${MASK_RATIO}" \
    --drop_ray_prob "${drop_ray}" \
    --batch "${BATCH}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --n_point "${N_POINT}" \
    --n_ray "${N_RAY}" \
    --d_model "${D_MODEL}" \
    --layers "${LAYERS}" \
    --heads "${HEADS}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --save_every "${SAVE_EVERY}" \
    --save_last "${SAVE_LAST}" \
    --auto_resume "${AUTO_RESUME}" \
    --resume "${save_dir}/last.pt" \
    --save_dir "${save_dir}" \
    > "${log_file}" 2>&1
  rc=$?

  if [ "${rc}" -eq 0 ]; then
    echo "[done ][gpu${gpu}] $(date '+%F %T') ${name}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  else
    echo "[fail ][gpu${gpu}] $(date '+%F %T') ${name} rc=${rc}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
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

echo "[info] total_jobs=${#JOBS[@]}"
echo "[info] gpu0=${GPU0} gpu1=${GPU1}"
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
  worker "${GPU0}" 0 &
  pid0=$!
  worker "${GPU1}" 1 &
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
