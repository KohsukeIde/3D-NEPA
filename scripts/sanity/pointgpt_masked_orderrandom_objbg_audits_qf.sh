#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=00:45:00
#PBS -j oe
#PBS -N ptgpt_mord_aud
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"

mkdir -p "${WORKDIR}/logs/sanity" "${WORKDIR}/results"

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

CONFIG_PATH="cfgs/PointGPT-S/finetune_scan_objbg.yaml"
CKPT_PATH="${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_masked_ordrand_objbg_e300/ckpt-best.pth"
OUT_PREFIX="${WORKDIR}/results/ptgpt_masked_ordrand_objbg"

cd "${POINTGPT_DIR}"

echo "=== PointGPT-S mask0.7 order-randomized obj_bg audits ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "config=${CONFIG_PATH}"
echo "ckpt=${CKPT_PATH}"
echo "out_prefix=${OUT_PREFIX}"
echo

python tools/eval_scanobjectnn_readout_audit.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --output_json "${OUT_PREFIX}_readout_full.json" \
  --output_md "${OUT_PREFIX}_readout_full.md"

python tools/eval_scanobjectnn_support_stress.py \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --output_json "${OUT_PREFIX}_stress_full.json" \
  --output_md "${OUT_PREFIX}_stress_full.md"

echo "[done] PointGPT-S mask0.7 order-randomized obj_bg audits"
