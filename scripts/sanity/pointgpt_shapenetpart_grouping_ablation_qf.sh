#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -N ptgpt_sp_grp
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

echo "=== ShapeNetPart grouping ablation ==="
echo "date=$(date -Is)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "root=${ROOT}"
echo

run_row() {
  local name="$1"
  local ckpt="$2"
  echo "[row] ${name}"
  python eval_shapenetpart_grouping_ablation.py \
    --ckpt "${ckpt}" \
    --root "${ROOT}" \
    --seed 0 \
    --batch_size "${BATCH_SIZE:-16}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --group_modes "${GROUP_MODES:-fps_knn,random_center_knn,voxel_center_knn,radius_fps,random_group}" \
    --conditions "${CONDITIONS:-clean,random_keep20,structured_keep20,part_drop_largest,part_keep20_per_part,xyz_zero}" \
    --output_json "${OUT_DIR}/ptgpt_shapenetpart_${name}_grouping_ablation.json" \
    --output_csv "${OUT_DIR}/ptgpt_shapenetpart_${name}_grouping_ablation.csv" \
    --output_md "${OUT_DIR}/ptgpt_shapenetpart_${name}_grouping_ablation.md"
}

run_row "official" "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth"
run_row "nomask" "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_e300/checkpoints/best_model.pth"

echo "[done] ShapeNetPart grouping ablation"
