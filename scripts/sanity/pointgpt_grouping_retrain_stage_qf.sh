#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=10:00:00
#PBS -j oe
#PBS -N ptgpt_grp
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
STAGE="${STAGE:-pretrain}"
GROUP_VARIANT="${GROUP_VARIANT:-random_center_knn}"

case "${GROUP_VARIANT}" in
  random_center_knn|random_group) ;;
  *)
    echo "[error] unsupported GROUP_VARIANT=${GROUP_VARIANT}"
    exit 2
    ;;
esac

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
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export USE_WANDB="${USE_WANDB:-0}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTHONPATH="${POINTGPT_DIR}:${POINTGPT_DIR}/segmentation/models${PYTHONPATH:+:${PYTHONPATH}}"

pretrain_cfg="cfgs/PointGPT-S/pretrain_group_${GROUP_VARIANT}.yaml"
objbg_cfg="cfgs/PointGPT-S/finetune_scan_objbg_group_${GROUP_VARIANT}.yaml"
pretrain_exp="pgpt_s_group_${GROUP_VARIANT}_e300"
pretrain_ckpt="${POINTGPT_DIR}/experiments/pretrain_group_${GROUP_VARIANT}/PointGPT-S/${pretrain_exp}/ckpt-last.pth"
objbg_exp="pgpt_s_group_${GROUP_VARIANT}_objbg_e300"
objbg_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg_group_${GROUP_VARIANT}/PointGPT-S/${objbg_exp}/ckpt-best.pth"
partseg_exp="pgpt_s_shapenetpart_group_${GROUP_VARIANT}_e300"
partseg_ckpt="${POINTGPT_DIR}/segmentation/log/part_seg/${partseg_exp}/checkpoints/best_model.pth"

echo "=== PointGPT grouping retrain stage ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "stage=${STAGE}"
echo "group_variant=${GROUP_VARIANT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo

python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
from pointnet2_ops import pointnet2_utils
print("pointnet2_ops import: OK", pointnet2_utils.__name__)
from knn_cuda import KNN
print("knn_cuda import: OK", KNN.__name__)
PY

cd "${POINTGPT_DIR}"

case "${STAGE}" in
  pretrain)
    export CONFIG_PATH="${pretrain_cfg}"
    export EXP_NAME="${pretrain_exp}"
    export WANDB_GROUP="pointgpt_grouping_pretrain"
    export WANDB_RUN_NAME="${EXP_NAME}"
    bash ../scripts/local/pointgpt_train_local_ddp.sh
    ;;

  objbg_ft)
    if [[ ! -f "${pretrain_ckpt}" ]]; then
      echo "[error] missing pretrain checkpoint: ${pretrain_ckpt}"
      exit 2
    fi
    export CONFIG_PATH="${objbg_cfg}"
    export EXP_NAME="${objbg_exp}"
    export CKPT_PATH="${pretrain_ckpt}"
    export VAL_FREQ="${VAL_FREQ:-1}"
    export SAVE_LAST_EVERY_EPOCH="${SAVE_LAST_EVERY_EPOCH:-0}"
    export EXTRA_ARGS="${EXTRA_ARGS:---seed 0}"
    export WANDB_GROUP="pointgpt_grouping_objbg_ft"
    export WANDB_RUN_NAME="${EXP_NAME}"
    bash ../scripts/local/pointgpt_finetune_local_ddp.sh
    ;;

  partseg_ft)
    if [[ ! -f "${pretrain_ckpt}" ]]; then
      echo "[error] missing pretrain checkpoint: ${pretrain_ckpt}"
      exit 2
    fi
    export CKPT_PATH="${pretrain_ckpt}"
    export RUN_NAME="${partseg_exp}"
    export GROUP_MODE="${GROUP_VARIANT}"
    export SEED="${SEED:-0}"
    export EPOCH="${EPOCH:-300}"
    export BATCH_SIZE="${BATCH_SIZE:-16}"
    bash ../scripts/local/pointgpt_s_shapenetpart_ft.sh
    ;;

  objbg_audit)
    if [[ ! -f "${objbg_ckpt}" ]]; then
      echo "[error] missing obj_bg checkpoint: ${objbg_ckpt}"
      exit 2
    fi
    out_prefix="${WORKDIR}/results/ptgpt_group_${GROUP_VARIANT}_objbg"
    python tools/eval_scanobjectnn_readout_audit.py \
      --config "${objbg_cfg}" \
      --ckpt "${objbg_ckpt}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --output_json "${out_prefix}_readout.json" \
      --output_md "${out_prefix}_readout.md"
    python tools/eval_scanobjectnn_support_stress.py \
      --config "${objbg_cfg}" \
      --ckpt "${objbg_ckpt}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --output_json "${out_prefix}_stress.json" \
      --output_md "${out_prefix}_stress.md"
    python tools/eval_scanobjectnn_grouping_ablation.py \
      --config "${objbg_cfg}" \
      --ckpt "${objbg_ckpt}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --num_workers "${NUM_WORKERS:-4}" \
      --group-modes "fps_knn,random_center_knn,voxel_center_knn,radius_fps,random_group" \
      --support-conditions "clean,random_keep20,structured_keep20" \
      --output_json "${out_prefix}_grouping.json" \
      --output_csv "${out_prefix}_grouping.csv" \
      --output_md "${out_prefix}_grouping.md"
    ;;

  partseg_audit)
    if [[ ! -f "${partseg_ckpt}" ]]; then
      echo "[error] missing ShapeNetPart checkpoint: ${partseg_ckpt}"
      exit 2
    fi
    cd "${POINTGPT_DIR}/segmentation"
    out_prefix="${WORKDIR}/results/ptgpt_shapenetpart_group_${GROUP_VARIANT}"
    root="${WORKDIR}/data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
    python eval_shapenetpart_support_stress.py \
      --ckpt "${partseg_ckpt}" \
      --root "${root}" \
      --group_mode "${GROUP_VARIANT}" \
      --batch_size "${BATCH_SIZE:-16}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --output_json "${out_prefix}_stress.json" \
      --output_md "${out_prefix}_stress.md"
    python eval_shapenetpart_grouping_ablation.py \
      --ckpt "${partseg_ckpt}" \
      --root "${root}" \
      --batch_size "${BATCH_SIZE:-16}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --group_modes "fps_knn,random_center_knn,voxel_center_knn,radius_fps,random_group" \
      --conditions "clean,random_keep20,structured_keep20,part_drop_largest,part_keep20_per_part,xyz_zero" \
      --output_json "${out_prefix}_grouping.json" \
      --output_csv "${out_prefix}_grouping.csv" \
      --output_md "${out_prefix}_grouping.md"
    ;;

  *)
    echo "[error] unsupported STAGE=${STAGE}"
    exit 2
    ;;
esac

echo "[done] stage=${STAGE} group_variant=${GROUP_VARIANT}"
