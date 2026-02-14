#!/usr/bin/env bash
set -eu

# Paper-safe mix pretrains on 2 GPUs:
#   1) shapenet_mix_nepa_mainsplit_s0
#   2) shapenet_mix_mae_mainsplit_s0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

BATCH="${BATCH:-32}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-3e-4}"
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

CFG_MIX="${CFG_MIX:-nepa3d/configs/pretrain_mixed_shapenet_mesh_udf_scan_mainsplit.yaml}"

RUN_ROOT="${RUN_ROOT:-runs}"
LOG_ROOT="${LOG_ROOT:-logs/pretrain/mix_mainsplit}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[error] python not found: ${PYTHON_BIN}"
  exit 1
fi
if [ ! -f "${CFG_MIX}" ]; then
  echo "[error] missing mix config: ${CFG_MIX}"
  exit 1
fi

run_one() {
  local gpu="$1"
  local name="$2"
  local objective="$3"
  local drop_ray="$4"
  local save_dir="$5"
  local log_file="$6"

  mkdir -p "${save_dir}" "$(dirname "${log_file}")"
  local final_ep final_ckpt
  final_ep="$(printf '%03d' $((EPOCHS - 1)))"
  final_ckpt="${save_dir}/ckpt_ep${final_ep}.pt"

  if [ -f "${final_ckpt}" ]; then
    echo "[skip][gpu${gpu}] ${name} (${final_ckpt} exists)" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
    return 0
  fi

  echo "[start][gpu${gpu}] $(date '+%F %T') ${name}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${PYTHON_BIN}" -u -m nepa3d.train.pretrain \
    --mix_config "${CFG_MIX}" \
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
  local rc=$?

  if [ "${rc}" -eq 0 ]; then
    echo "[done ][gpu${gpu}] $(date '+%F %T') ${name}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  else
    echo "[fail ][gpu${gpu}] $(date '+%F %T') ${name} rc=${rc}" | tee -a "${LOG_ROOT}/runner_gpu${gpu}.log"
  fi
  return "${rc}"
}

echo "[info] cfg_mix=${CFG_MIX}"
echo "[info] gpu0=${GPU0} gpu1=${GPU1}"
echo "[info] log_root=${LOG_ROOT}"

run_one "${GPU0}" "shapenet_mix_nepa_mainsplit_s0" "nepa" "${DROP_RAY_PROB}" \
  "${RUN_ROOT}/shapenet_mix_nepa_mainsplit_s0" "${LOG_ROOT}/shapenet_mix_nepa_mainsplit_s0.log" &
pid0=$!
run_one "${GPU1}" "shapenet_mix_mae_mainsplit_s0" "mae" "0.0" \
  "${RUN_ROOT}/shapenet_mix_mae_mainsplit_s0" "${LOG_ROOT}/shapenet_mix_mae_mainsplit_s0.log" &
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

