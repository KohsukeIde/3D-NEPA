#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -N ptgpt_obj_str
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
cd "${POINTGPT_DIR}"

echo "=== PointGPT-S ScanObjectNN support-stress severity ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"

run_row() {
  local name="$1"
  local ckpt="$2"
  echo "[row] ${name}"
  python tools/eval_scanobjectnn_support_stress.py \
    --config "${CONFIG_PATH}" \
    --ckpt "${ckpt}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --output_json "${WORKDIR}/results/ptgpt_stress_${name}_objbg_severity.json" \
    --output_md "${WORKDIR}/results/ptgpt_stress_${name}_objbg_severity.md"
}

run_row "official" "checkpoints/official/pointgpt_s_scan_objbg_official.pth"
run_row "nomask" "${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_objbg_e300/ckpt-best.pth"
run_row "nomask_ordrand" "${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300/ckpt-best.pth"
run_row "masked_ordrand" "${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_masked_ordrand_objbg_e300/ckpt-best.pth"

echo "[done] PointGPT-S ScanObjectNN support-stress severity"
