#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -N pointmae_scan_test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTMAE_ROOT="${POINTMAE_ROOT:-${WORKDIR}/Point-MAE}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
REQUIRE_POINT_EXT="${REQUIRE_POINT_EXT:-1}"  # 1: require pointnet2_ops + knn_cuda imports
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-}"

VARIANT="${VARIANT:-pb_t50_rs}"   # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_test_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
CFG_PROFILE="${CFG_PROFILE:-standard}"  # standard|sanity
CKPT_PATH="${CKPT_PATH:-}"
CKPT_RUN_TAG="${CKPT_RUN_TAG:-}"
CKPT_PICK="${CKPT_PICK:-best}"   # best | last
CONFIG_PATH_OVERRIDE="${CONFIG_PATH_OVERRIDE:-}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae_scratch_tests}"
if [[ "${LOG_ROOT}" != /* ]]; then
  LOG_ROOT="${WORKDIR}/${LOG_ROOT}"
fi

if [[ "${CKPT_PICK}" != "best" && "${CKPT_PICK}" != "last" ]]; then
  echo "[error] CKPT_PICK must be one of: best | last (got: ${CKPT_PICK})"
  exit 2
fi

if [[ -n "${CKPT_PATH}" ]]; then
  if [[ "${CKPT_PATH}" != /* ]]; then
    CKPT_PATH="${WORKDIR}/${CKPT_PATH}"
  fi
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "[error] CKPT_PATH not found: ${CKPT_PATH}"
    exit 2
  fi
else
  if [[ -z "${CKPT_RUN_TAG}" ]]; then
    echo "[error] either CKPT_PATH or CKPT_RUN_TAG is required"
    exit 2
  fi
  CKPT_FILE="ckpt-${CKPT_PICK}.pth"
  mapfile -t _cand < <(find "${POINTMAE_ROOT}/experiments" -type f -path "*/${CKPT_RUN_TAG}/${CKPT_FILE}" 2>/dev/null | sort)
  if [[ "${#_cand[@]}" -eq 0 ]]; then
    echo "[error] could not resolve checkpoint by CKPT_RUN_TAG=${CKPT_RUN_TAG} (wanted ${CKPT_FILE})"
    exit 2
  fi
  if [[ "${#_cand[@]}" -gt 1 ]]; then
    echo "[warn] multiple checkpoints matched CKPT_RUN_TAG=${CKPT_RUN_TAG}; using first:"
    printf '  %s\n' "${_cand[@]}"
  fi
  CKPT_PATH="${_cand[0]}"
fi

if [[ ! -d "${POINTMAE_ROOT}" ]]; then
  echo "[error] Point-MAE root not found: ${POINTMAE_ROOT}"
  exit 2
fi

mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi
if [[ -z "${TORCH_EXTENSIONS_DIR}" ]]; then
  if [[ -n "${PBS_JOBID:-}" ]]; then
    TORCH_EXTENSIONS_DIR="/local/${PBS_JOBID}/torch_extensions"
  else
    TORCH_EXTENSIONS_DIR="${WORKDIR}/.cache/torch_extensions"
  fi
fi
mkdir -p "${TORCH_EXTENSIONS_DIR}"
export TORCH_CUDA_ARCH_LIST
export TORCH_EXTENSIONS_DIR
if [[ "${REQUIRE_POINT_EXT}" == "1" ]]; then
  python - <<'PY'
try:
    from pointnet2_ops import pointnet2_utils  # noqa: F401
except Exception as e:
    print(f"[error] pointnet2_ops unavailable: {e}")
    raise SystemExit(2)
try:
    from knn_cuda import KNN  # noqa: F401
except Exception as e:
    print(f"[error] knn_cuda unavailable: {e}")
    raise SystemExit(2)
print("[env-check] pointnet2_ops + knn_cuda available")
PY
fi

# Point-MAE expected ScanObjectNN layout.
mkdir -p "${POINTMAE_ROOT}/data/ScanObjectNN"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split"
ln -sfn "${WORKDIR}/data/ScanObjectNN/h5_files/main_split_nobg" "${POINTMAE_ROOT}/data/ScanObjectNN/main_split_nobg"

case "${CFG_PROFILE}" in
  standard) CFG_SUFFIX="" ;;
  sanity) CFG_SUFFIX="_sanity" ;;
  *)
    echo "[error] unsupported CFG_PROFILE=${CFG_PROFILE} (standard|sanity)"
    exit 2
    ;;
esac

case "${VARIANT}" in
  pb_t50_rs|hardest) CONFIG_PATH="cfgs/finetune_scan_hardest${CFG_SUFFIX}.yaml" ;;
  obj_bg) CONFIG_PATH="cfgs/finetune_scan_objbg${CFG_SUFFIX}.yaml" ;;
  obj_only) CONFIG_PATH="cfgs/finetune_scan_objonly${CFG_SUFFIX}.yaml" ;;
  *)
    echo "[error] unsupported VARIANT=${VARIANT} (pb_t50_rs|obj_bg|obj_only)"
    exit 2
    ;;
esac

CONFIG_PATH_EXEC="${POINTMAE_ROOT}/${CONFIG_PATH}"
if [[ -n "${CONFIG_PATH_OVERRIDE}" ]]; then
  if [[ "${CONFIG_PATH_OVERRIDE}" != /* ]]; then
    CONFIG_PATH_EXEC="${WORKDIR}/${CONFIG_PATH_OVERRIDE}"
  else
    CONFIG_PATH_EXEC="${CONFIG_PATH_OVERRIDE}"
  fi
elif [[ -n "${CKPT_RUN_TAG}" ]]; then
  CKPT_CFG_CAND="$(dirname "${CKPT_PATH}")/config.yaml"
  if [[ -f "${CKPT_CFG_CAND}" ]]; then
    CONFIG_PATH_EXEC="${CKPT_CFG_CAND}"
  fi
fi
if [[ ! -f "${CONFIG_PATH_EXEC}" ]]; then
  echo "[error] config not found: ${CONFIG_PATH_EXEC}"
  exit 2
fi

OUT_LOG="${LOG_ROOT}/${RUN_TAG}.log"

echo "=== POINT-MAE TEST (from ckpt) ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "variant=${VARIANT}"
echo "run_tag=${RUN_TAG}"
echo "venv_activate=${VENV_ACTIVATE}"
echo "config=${CONFIG_PATH_EXEC}"
echo "ckpt=${CKPT_PATH}"
echo "require_point_ext=${REQUIRE_POINT_EXT}"
echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
if [[ -n "${CKPT_RUN_TAG}" ]]; then
  echo "ckpt_run_tag=${CKPT_RUN_TAG}"
fi
echo "log=${OUT_LOG}"
echo

cd "${POINTMAE_ROOT}"
python main.py \
  --test \
  --config "${CONFIG_PATH_EXEC}" \
  --exp_name "${RUN_TAG}" \
  --ckpts "${CKPT_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  2>&1 | tee "${OUT_LOG}"

TEST_LINE="$(grep -n "\\[TEST\\] acc =" "${OUT_LOG}" | tail -n 1 || true)"
if [[ -n "${TEST_LINE}" ]]; then
  echo "[summary] ${TEST_LINE}"
else
  echo "[warn] could not find '[TEST] acc' in ${OUT_LOG}"
fi
