#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -N ptgpt_sp_str
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
ROOT="${WORKDIR}/data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}/logs/sanity" "${OUT_DIR}"

source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6/12.6.2 2>/dev/null || true
  module load gcc/11.4.1 2>/dev/null || true
fi
if command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME_AUTO="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_AUTO}}"
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${POINTGPT_DIR}:${PYTHONPATH:-}"

cd "${POINTGPT_DIR}/segmentation"

echo "=== ShapeNetPart support stress ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "root=${ROOT}"
echo

python eval_shapenetpart_support_stress.py \
  --ckpt "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth" \
  --root "${ROOT}" \
  --seed 0 \
  --batch_size 16 \
  --num_workers 8 \
  --output_json "${OUT_DIR}/ptgpt_shapenetpart_official_support_stress.json" \
  --output_md "${OUT_DIR}/ptgpt_shapenetpart_official_support_stress.md"

python eval_shapenetpart_support_stress.py \
  --ckpt "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_e300/checkpoints/best_model.pth" \
  --root "${ROOT}" \
  --seed 0 \
  --batch_size 16 \
  --num_workers 8 \
  --output_json "${OUT_DIR}/ptgpt_shapenetpart_nomask_support_stress.json" \
  --output_md "${OUT_DIR}/ptgpt_shapenetpart_nomask_support_stress.md"

echo "[done] ShapeNetPart support stress"
