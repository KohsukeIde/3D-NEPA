#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PCP_ROOT="${PCP_ROOT:-${ROOT_DIR}/PCP-MAE}"
GEOPCP_ENV_NAME="${GEOPCP_ENV_NAME:-geopcp-pcpmae-cu118}"
PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/envs/${GEOPCP_ENV_NAME}/bin/python}"
CONDA_PREFIX_FROM_PYTHON="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
GEOPCP_ARTIFACT_ROOT="${GEOPCP_ARTIFACT_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/repo_artifacts}"
PCPMAE_EXPERIMENTS_ROOT="${PCPMAE_EXPERIMENTS_ROOT:-${GEOPCP_ARTIFACT_ROOT}/pcpmae_experiments}"
PCPMAE_SEG_LOG_ROOT="${PCPMAE_SEG_LOG_ROOT:-${GEOPCP_ARTIFACT_ROOT}/pcpmae_segmentation_logs}"

export ROOT_DIR PCP_ROOT PYTHON_BIN
export CONDA_PREFIX="${CONDA_PREFIX_FROM_PYTHON}"
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PCP_ROOT}:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GEOPCP_ARTIFACT_ROOT PCPMAE_EXPERIMENTS_ROOT PCPMAE_SEG_LOG_ROOT

mkdir -p "${PCPMAE_EXPERIMENTS_ROOT}" "${PCPMAE_SEG_LOG_ROOT}" "${GEOPCP_ARTIFACT_ROOT}"

geopcp_die() {
  echo "[error] $*" >&2
  exit 1
}

geopcp_require_python() {
  [[ -x "${PYTHON_BIN}" ]] || geopcp_die "python not found: ${PYTHON_BIN}"
}

geopcp_require_compiled_backends() {
  geopcp_require_python
  (
    cd "${PCP_ROOT}"
    "${PYTHON_BIN}" - <<'PY'
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import require_compiled_backend as require_chamfer

pointnet2_utils.require_compiled_backend()
require_chamfer()
print(f"POINTNET2_BACKEND={pointnet2_utils.backend_name()}")
from extensions.chamfer_dist import backend_name as chamfer_backend_name
print(f"CHAMFER_BACKEND={chamfer_backend_name()}")
PY
  )
}

geopcp_require_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || geopcp_die "nvidia-smi not found"
}
