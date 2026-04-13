#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PCP_ROOT="${ROOT_DIR}/PCP-MAE"
ENV_NAME="${GEOPCP_ENV_NAME:-geopcp-pcpmae-cu118}"
CONDA_BIN="${CONDA_BIN:-/home/minesawa/anaconda3/bin/conda}"

[[ -x "${CONDA_BIN}" ]] || { echo "[error] conda not found: ${CONDA_BIN}" >&2; exit 1; }

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.10
fi

"${CONDA_BIN}" install -y -n "${ENV_NAME}" \
  -c pytorch -c nvidia -c conda-forge \
  pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 \
  cuda-nvcc=11.8.89 cuda-toolkit=11.8.0 \
  gcc_linux-64=10.4.0 gxx_linux-64=10.4.0 \
  "numpy<2" "setuptools<81" wheel

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade "numpy<2" "setuptools<81" wheel
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install \
  "numpy<2" \
  "setuptools<81" \
  timm==0.4.5 \
  easydict \
  accelerate \
  einops \
  tensorboardX \
  tensorboard \
  transforms3d \
  ninja \
  pyyaml \
  scipy \
  h5py \
  tqdm \
  termcolor \
  matplotlib \
  opencv-python \
  pandas \
  scikit-learn \
  trimesh \
  wandb
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade "numpy<2" "setuptools<81" wheel

"${CONDA_BIN}" run -n "${ENV_NAME}" bash -lc "
  set -euo pipefail
  export CUDA_HOME=\"\$CONDA_PREFIX\"
  export CC=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc\"
  export CXX=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\"
  export CPATH=\"\$CONDA_PREFIX/targets/x86_64-linux/include:\$CONDA_PREFIX/include:\${CPATH:-}\"
  export CPLUS_INCLUDE_PATH=\"\$CONDA_PREFIX/targets/x86_64-linux/include:\$CONDA_PREFIX/include:\${CPLUS_INCLUDE_PATH:-}\"
  export TORCH_CUDA_ARCH_LIST=\"8.0\"
  export MAX_JOBS=\"\${MAX_JOBS:-8}\"
  cd '${PCP_ROOT}/extensions/chamfer_dist'
  python setup.py install
"

"${CONDA_BIN}" run -n "${ENV_NAME}" bash -lc "
  set -euo pipefail
  export CUDA_HOME=\"\$CONDA_PREFIX\"
  export CC=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc\"
  export CXX=\"\$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++\"
  export CPATH=\"\$CONDA_PREFIX/targets/x86_64-linux/include:\$CONDA_PREFIX/include:\${CPATH:-}\"
  export CPLUS_INCLUDE_PATH=\"\$CONDA_PREFIX/targets/x86_64-linux/include:\$CONDA_PREFIX/include:\${CPLUS_INCLUDE_PATH:-}\"
  export TORCH_CUDA_ARCH_LIST=\"8.0\"
  export MAX_JOBS=\"\${MAX_JOBS:-8}\"
  pip install --no-build-isolation 'git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib'
"

"${CONDA_BIN}" run -n "${ENV_NAME}" bash -lc "
  set -euo pipefail
  cd '${PCP_ROOT}'
  python - <<'PY'
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import backend_name as chamfer_backend_name
print(f'POINTNET2_BACKEND={pointnet2_utils.backend_name()}')
print(f'CHAMFER_BACKEND={chamfer_backend_name()}')
if pointnet2_utils.backend_name() != 'compiled':
    raise SystemExit('pointnet2_ops did not resolve to compiled backend')
if chamfer_backend_name() != 'native':
    raise SystemExit('chamfer_dist did not resolve to native backend')
PY
"

echo "[done] env=${ENV_NAME}"
