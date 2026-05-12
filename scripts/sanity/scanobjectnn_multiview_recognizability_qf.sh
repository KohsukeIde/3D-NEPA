#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=04:00:00
#PBS -N mv2d_recog
#PBS -j oe
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

cd /groups/qgah50055/ide/concerto-shortcut-mvp
source 3D-NEPA/.venv-pointgpt/bin/activate
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

echo "=== ScanObjectNN multi-view recognizability ==="
date --iso-8601=seconds
hostname
echo "pbs_jobid=${PBS_JOBID:-none}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
PY

TRAIN_CONDITIONS_VALUE="${TRAIN_CONDITIONS:-clean}"
TRAIN_TAG="$(echo "${TRAIN_CONDITIONS_VALUE}" | tr ',/' '__')"
OUT="3D-NEPA/results/multiview_2d/scanobjectnn_recognizability_${TRAIN_TAG}_v${VIEWS:-10}_s${IMAGE_SIZE:-96}_e${EPOCHS:-120}"
python 3D-NEPA/multiview_2d/eval_scanobjectnn_multiview_recognizability.py \
  --image_size "${IMAGE_SIZE:-96}" \
  --views "${VIEWS:-10}" \
  --epochs "${EPOCHS:-120}" \
  --batch_size "${BATCH_SIZE:-48}" \
  --train_conditions "${TRAIN_CONDITIONS_VALUE}" \
  --local_noise_sigma "${LOCAL_NOISE_SIGMA:-0.08}" \
  --num_examples "${NUM_EXAMPLES:-15}" \
  --output_dir "${OUT}"

echo "[done] ${OUT}"
