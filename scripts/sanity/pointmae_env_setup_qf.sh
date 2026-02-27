#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -N pointmae_env_setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CREATE_VENV_IF_MISSING="${CREATE_VENV_IF_MISSING:-0}"  # 1: create venv if missing
INSTALL_TORCH_IF_MISSING="${INSTALL_TORCH_IF_MISSING:-0}"  # 1: install torch/cu128 if absent
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/sanity/pointmae_env}"

mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  if [[ "${CREATE_VENV_IF_MISSING}" != "1" ]]; then
    echo "[error] venv activate script not found: ${VENV_ACTIVATE}"
    echo "        set CREATE_VENV_IF_MISSING=1 to auto-create a venv."
    exit 2
  fi
  VENV_DIR="$(dirname "$(dirname "${VENV_ACTIVATE}")")"
  mkdir -p "${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"

source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi
export TORCH_CUDA_ARCH_LIST

echo "=== POINT-MAE ENV SETUP ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "workdir=${WORKDIR}"
echo "venv_activate=${VENV_ACTIVATE}"
echo "cuda_module=${CUDA_MODULE}"
echo "torch_cuda_arch_list=${TORCH_CUDA_ARCH_LIST}"
echo

python -V
python -m pip install -q --upgrade pip wheel setuptools ninja packaging
python -m pip install -q easydict tensorboardX termcolor timm==0.4.5 transforms3d matplotlib torchvision h5py

if ! python - <<'PY'
try:
    import torch  # noqa: F401
    print("[check] torch present")
except Exception:
    raise SystemExit(1)
PY
then
  if [[ "${INSTALL_TORCH_IF_MISSING}" != "1" ]]; then
    echo "[error] torch is missing in venv. set INSTALL_TORCH_IF_MISSING=1 to install it."
    exit 2
  fi
  python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio
fi

# Install pointnet2_ops from source with patched TORCH_CUDA_ARCH_LIST.
TMP_BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf "${TMP_BUILD_ROOT}"' EXIT
git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git "${TMP_BUILD_ROOT}/Pointnet2_PyTorch"
python - "${TMP_BUILD_ROOT}/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py" "${TORCH_CUDA_ARCH_LIST}" <<'PY'
import pathlib
import re
import sys

setup_py = pathlib.Path(sys.argv[1])
arch = sys.argv[2]
src = setup_py.read_text()
src_new = re.sub(
    r'os\.environ\["TORCH_CUDA_ARCH_LIST"\]\s*=\s*".*?"',
    f'os.environ["TORCH_CUDA_ARCH_LIST"] = "{arch}"',
    src,
)
if src == src_new:
    raise SystemExit("[error] failed to patch TORCH_CUDA_ARCH_LIST in setup.py")
setup_py.write_text(src_new)
print(f"[patch] setup.py TORCH_CUDA_ARCH_LIST -> {arch}")
PY
python -m pip install -v --no-cache-dir --no-build-isolation \
  "${TMP_BUILD_ROOT}/Pointnet2_PyTorch/pointnet2_ops_lib"
python -m pip install -v --no-cache-dir --upgrade \
  "https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops import: OK", pointnet2_utils.__name__)
from knn_cuda import KNN
print("knn_cuda import: OK", KNN.__name__)
PY

echo "[done] pointnet2_ops setup completed."
