#!/bin/bash
#PBS -P qgah50055
#PBS -q abciq
#PBS -W group_list=qgah50055
#PBS -l rt_QF=1
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -N ptgpt_lnoise2d
#PBS -o /groups/qgah50055/ide/concerto-shortcut-mvp/3D-NEPA/logs/sanity/

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/groups/qgah50055/ide/concerto-shortcut-mvp}"
WORKDIR="${WORKDIR:-${REPO_ROOT}/3D-NEPA}"
POINTGPT_DIR="${POINTGPT_DIR:-${WORKDIR}/PointGPT}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv-pointgpt/bin/activate}"
STAGE="${STAGE:-objbg_localnoise}"

mkdir -p "${WORKDIR}/logs/sanity" "${WORKDIR}/results" "${WORKDIR}/results/multiview_2d"

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
export PYTHONPATH="${POINTGPT_DIR}:${POINTGPT_DIR}/segmentation/models${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== PointGPT local-noise / multi-view 2D stage ==="
echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "pbs_jobid=${PBS_JOBID:-}"
echo "stage=${STAGE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
PY

cd "${POINTGPT_DIR}"

obj_cfg="cfgs/PointGPT-S/finetune_scan_objbg.yaml"
obj_group_center_cfg="cfgs/PointGPT-S/finetune_scan_objbg_group_random_center_knn.yaml"
obj_group_random_cfg="cfgs/PointGPT-S/finetune_scan_objbg_group_random_group.yaml"

official_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_official_objbg_e300_seed1/ckpt-best.pth"
nomask_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_objbg_e300/ckpt-best.pth"
masked_ordrand_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_masked_ordrand_objbg_e300/ckpt-best.pth"
nomask_ordrand_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg/PointGPT-S/pgpt_s_nomask_ordrand_objbg_e300/ckpt-best.pth"
group_center_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg_group_random_center_knn/PointGPT-S/pgpt_s_group_random_center_knn_objbg_e300/ckpt-best.pth"
group_random_obj_ckpt="${POINTGPT_DIR}/experiments/finetune_scan_objbg_group_random_group/PointGPT-S/pgpt_s_group_random_group_objbg_e300/ckpt-best.pth"

run_obj_stress () {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local out_prefix="${WORKDIR}/results/${name}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] missing ckpt for ${name}: ${ckpt}"
    return 0
  fi
  python tools/eval_scanobjectnn_support_stress.py \
    --config "${cfg}" \
    --ckpt "${ckpt}" \
    --batch_size "${BATCH_SIZE:-32}" \
    --num_workers "${NUM_WORKERS:-4}" \
    --local_noise_sigma "${LOCAL_NOISE_SIGMA:-0.08}" \
    --output_json "${out_prefix}.json" \
    --output_md "${out_prefix}.md"
}

run_part_stress () {
  local name="$1"
  local ckpt="$2"
  local group_mode="$3"
  local out_prefix="${WORKDIR}/results/${name}"
  local root="${WORKDIR}/data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] missing ckpt for ${name}: ${ckpt}"
    return 0
  fi
  cd "${POINTGPT_DIR}/segmentation"
  python eval_shapenetpart_support_stress.py \
    --ckpt "${ckpt}" \
    --root "${root}" \
    --group_mode "${group_mode}" \
    --batch_size "${PART_BATCH_SIZE:-16}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --local_noise_sigma "${LOCAL_NOISE_SIGMA:-0.08}" \
    --output_json "${out_prefix}.json" \
    --output_md "${out_prefix}.md"
  cd "${POINTGPT_DIR}"
}

case "${STAGE}" in
  objbg_localnoise)
    run_obj_stress "ptgpt_stress_official_objbg_localnoise" "${obj_cfg}" "${official_obj_ckpt}"
    run_obj_stress "ptgpt_stress_nomask_objbg_localnoise" "${obj_cfg}" "${nomask_obj_ckpt}"
    run_obj_stress "ptgpt_stress_masked_ordrand_objbg_localnoise" "${obj_cfg}" "${masked_ordrand_obj_ckpt}"
    run_obj_stress "ptgpt_stress_nomask_ordrand_objbg_localnoise" "${obj_cfg}" "${nomask_ordrand_obj_ckpt}"
    run_obj_stress "ptgpt_group_random_center_knn_objbg_localnoise" "${obj_group_center_cfg}" "${group_center_obj_ckpt}"
    run_obj_stress "ptgpt_group_random_group_objbg_localnoise" "${obj_group_random_cfg}" "${group_random_obj_ckpt}"
    ;;

  shapenetpart_localnoise)
    run_part_stress "ptgpt_shapenetpart_official_support_stress_localnoise" \
      "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_official_e300/checkpoints/best_model.pth" "fps_knn"
    run_part_stress "ptgpt_shapenetpart_nomask_support_stress_localnoise" \
      "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_nomask_e300/checkpoints/best_model.pth" "fps_knn"
    run_part_stress "ptgpt_shapenetpart_group_random_center_knn_stress_localnoise" \
      "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_group_random_center_knn_e300/checkpoints/best_model.pth" "random_center_knn"
    run_part_stress "ptgpt_shapenetpart_group_random_group_stress_localnoise" \
      "${POINTGPT_DIR}/segmentation/log/part_seg/pgpt_s_shapenetpart_group_random_group_e300/checkpoints/best_model.pth" "random_group"
    ;;

  multiview_2d)
    cd "${REPO_ROOT}"
    python 3D-NEPA/multiview_2d/eval_scanobjectnn_multiview_support.py \
      --root "${POINTGPT_DIR}/data/ScanObjectNN/h5_files/main_split" \
      --image_size "${IMAGE_SIZE:-64}" \
      --epochs "${EPOCHS:-40}" \
      --batch_size "${BATCH_SIZE:-64}" \
      --local_noise_sigma "${LOCAL_NOISE_SIGMA:-0.08}" \
      --output_dir "${WORKDIR}/results/multiview_2d/scanobjectnn_objbg_e${EPOCHS:-40}_s${LOCAL_NOISE_SIGMA:-0.08}" \
      --save_examples 1
    ;;

  *)
    echo "[error] unsupported STAGE=${STAGE}"
    exit 2
    ;;
esac

echo "[done] stage=${STAGE}"
