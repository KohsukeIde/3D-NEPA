#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N pointmae_scan_scratch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
POINTMAE_ROOT="${POINTMAE_ROOT:-${WORKDIR}/Point-MAE}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

VARIANT="${VARIANT:-pb_t50_rs}"  # pb_t50_rs|obj_bg|obj_only
RUN_TAG="${RUN_TAG:-pointmae_${VARIANT}_scratch_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CFG_PROFILE="${CFG_PROFILE:-standard}"  # standard|sanity
TOTAL_BS_OVERRIDE="${TOTAL_BS_OVERRIDE:-}"
NPOINT_OVERRIDE="${NPOINT_OVERRIDE:-}"
NUM_GROUP_OVERRIDE="${NUM_GROUP_OVERRIDE:-}"
GROUP_SIZE_OVERRIDE="${GROUP_SIZE_OVERRIDE:-}"
NO_TEST_AS_VAL="${NO_TEST_AS_VAL:-1}"   # 1: split train->(train/val), 0: legacy Point-MAE test-as-val
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_SEED="${VAL_SEED:-0}"

if [[ "${NO_TEST_AS_VAL}" != "0" && "${NO_TEST_AS_VAL}" != "1" ]]; then
  echo "[error] NO_TEST_AS_VAL must be 0 or 1 (got: ${NO_TEST_AS_VAL})"
  exit 2
fi

LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae_scratch}"
if [[ "${LOG_ROOT}" != /* ]]; then
  LOG_ROOT="${WORKDIR}/${LOG_ROOT}"
fi
mkdir -p "${LOG_ROOT}"

if [[ ! -d "${POINTMAE_ROOT}" ]]; then
  echo "[error] Point-MAE root not found: ${POINTMAE_ROOT}"
  exit 2
fi

cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

python -V
python -m pip install -q easydict tensorboardX termcolor timm==0.4.5 transforms3d matplotlib torchvision h5py

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
if [[ -n "${TOTAL_BS_OVERRIDE}" ]]; then
  TMP_CFG_DIR="${WORKDIR}/tmp/pointmae_scratch_cfg_override"
  mkdir -p "${TMP_CFG_DIR}"
  TMP_CFG="${TMP_CFG_DIR}/${RUN_TAG}.yaml"
  cp "${CONFIG_PATH_EXEC}" "${TMP_CFG}"
  sed -E -i "s/^(total_bs[[:space:]]*:[[:space:]]*).*/\\1${TOTAL_BS_OVERRIDE}/" "${TMP_CFG}"
  CONFIG_PATH_EXEC="${TMP_CFG}"
fi
if [[ -n "${NPOINT_OVERRIDE}" || -n "${NUM_GROUP_OVERRIDE}" || -n "${GROUP_SIZE_OVERRIDE}" ]]; then
  if [[ -z "${TMP_CFG_DIR:-}" ]]; then
    TMP_CFG_DIR="${WORKDIR}/tmp/pointmae_scratch_cfg_override"
    mkdir -p "${TMP_CFG_DIR}"
    TMP_CFG="${TMP_CFG_DIR}/${RUN_TAG}.yaml"
    cp "${CONFIG_PATH_EXEC}" "${TMP_CFG}"
    CONFIG_PATH_EXEC="${TMP_CFG}"
  fi
  python - "${CONFIG_PATH_EXEC}" "${NPOINT_OVERRIDE}" "${NUM_GROUP_OVERRIDE}" "${GROUP_SIZE_OVERRIDE}" <<'PY'
import sys
import yaml

cfg_path, npoints, num_group, group_size = sys.argv[1:5]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

if npoints:
    cfg["npoints"] = int(npoints)
if num_group:
    cfg.setdefault("model", {})
    cfg["model"]["num_group"] = int(num_group)
if group_size:
    cfg.setdefault("model", {})
    cfg["model"]["group_size"] = int(group_size)

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
fi

# Default policy: do not use test-as-val. Build val split from train set.
if [[ "${NO_TEST_AS_VAL}" == "1" ]]; then
  if [[ -z "${TMP_CFG_DIR:-}" ]]; then
    TMP_CFG_DIR="${WORKDIR}/tmp/pointmae_scratch_cfg_override"
    mkdir -p "${TMP_CFG_DIR}"
    TMP_CFG="${TMP_CFG_DIR}/${RUN_TAG}.yaml"
    cp "${CONFIG_PATH_EXEC}" "${TMP_CFG}"
    CONFIG_PATH_EXEC="${TMP_CFG}"
  fi
  python - "${CONFIG_PATH_EXEC}" "${VAL_RATIO}" "${VAL_SEED}" <<'PY'
import sys
import yaml

cfg_path = sys.argv[1]
val_ratio = float(sys.argv[2])
val_seed = int(sys.argv[3])
if not (0.0 < val_ratio < 1.0):
    raise SystemExit(f"[error] VAL_RATIO must be in (0,1), got {val_ratio}")

with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

ds = cfg.get("dataset", {})
for split in ("train", "val"):
    others = ds.get(split, {}).get("others", {})
    others["val_ratio"] = val_ratio
    others["val_seed"] = val_seed
    ds[split]["others"] = others

ds["val"]["others"]["subset"] = "val"
cfg["dataset"] = ds

with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
fi

OUT_LOG="${LOG_ROOT}/${RUN_TAG}.log"

echo "=== POINT-MAE SCRATCH ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "variant=${VARIANT}"
echo "run_tag=${RUN_TAG}"
echo "config=${CONFIG_PATH_EXEC}"
echo "nproc_per_node=${NPROC_PER_NODE}"
if [[ -n "${TOTAL_BS_OVERRIDE}" ]]; then
  echo "total_bs_override=${TOTAL_BS_OVERRIDE}"
fi
if [[ -n "${NPOINT_OVERRIDE}" ]]; then
  echo "npoints_override=${NPOINT_OVERRIDE}"
fi
if [[ -n "${NUM_GROUP_OVERRIDE}" ]]; then
  echo "num_group_override=${NUM_GROUP_OVERRIDE}"
fi
if [[ -n "${GROUP_SIZE_OVERRIDE}" ]]; then
  echo "group_size_override=${GROUP_SIZE_OVERRIDE}"
fi
echo "no_test_as_val=${NO_TEST_AS_VAL}"
if [[ "${NO_TEST_AS_VAL}" == "1" ]]; then
  echo "policy=STRICT(no test-as-val)"
  echo "val_ratio=${VAL_RATIO}"
  echo "val_seed=${VAL_SEED}"
else
  echo "policy=LEGACY(test-as-val, explicitly requested)"
fi
echo "log=${OUT_LOG}"
echo

cd "${POINTMAE_ROOT}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" main.py \
    --launcher pytorch \
    --scratch_model \
    --config "${CONFIG_PATH_EXEC}" \
    --exp_name "${RUN_TAG}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUT_LOG}"
else
  python main.py \
    --launcher none \
    --scratch_model \
    --config "${CONFIG_PATH_EXEC}" \
    --exp_name "${RUN_TAG}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUT_LOG}"
fi

VAL_LINE="$(grep -n \"\\[Validation\\]\" "${OUT_LOG}" | tail -n 1 || true)"
if [[ -n "${VAL_LINE}" ]]; then
  echo "[summary] ${VAL_LINE}"
else
  echo "[warn] could not find '[Validation]' in ${OUT_LOG}"
fi
