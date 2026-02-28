#!/bin/bash
#PBS -l rt_QF=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -N patchnepa_ptonly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"

MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only_onepass.yaml}"
RUN_TAG="${RUN_TAG:-patchnepa_ptonly_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-runs/patchnepa_pointonly/${RUN_TAG}}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/patch_nepa_pretrain}"

N_POINT="${N_POINT:-1024}"
N_RAY="${N_RAY:-0}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-3e-4}"
SEED="${SEED:-0}"

PATCH_EMBED="${PATCH_EMBED:-fps_knn}"
GROUP_SIZE="${GROUP_SIZE:-32}"
NUM_GROUPS="${NUM_GROUPS:-64}"
SERIAL_ORDER="${SERIAL_ORDER:-morton}"
SERIAL_BITS="${SERIAL_BITS:-10}"
SERIAL_SHUFFLE_WITHIN_PATCH="${SERIAL_SHUFFLE_WITHIN_PATCH:-0}"

PT_XYZ_KEY="${PT_XYZ_KEY:-pc_xyz}"
PT_DIST_KEY="${PT_DIST_KEY:-pt_dist_pool}"
ABLATE_POINT_DIST="${ABLATE_POINT_DIST:-0}"
PT_SAMPLE_MODE="${PT_SAMPLE_MODE:-rfps_cached}"
PT_FPS_KEY="${PT_FPS_KEY:-auto}"
PT_RFPS_KEY="${PT_RFPS_KEY:-auto}"
PT_RFPS_M="${PT_RFPS_M:-4096}"
POINT_ORDER_MODE="${POINT_ORDER_MODE:-morton}"

USE_RAY_PATCH="${USE_RAY_PATCH:-0}"
RAY_POOL_MODE="${RAY_POOL_MODE:-mean}"
RAY_FUSE="${RAY_FUSE:-add}"
RAY_MISS_T="${RAY_MISS_T:-4.0}"
RAY_HIT_THRESHOLD="${RAY_HIT_THRESHOLD:-0.5}"

D_MODEL="${D_MODEL:-384}"
N_LAYERS="${N_LAYERS:-12}"
N_HEADS="${N_HEADS:-6}"
MLP_RATIO="${MLP_RATIO:-4.0}"
BACKBONE_MODE="${BACKBONE_MODE:-nepa2d}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.0}"
QK_NORM="${QK_NORM:-1}"
QK_NORM_AFFINE="${QK_NORM_AFFINE:-0}"
QK_NORM_BIAS="${QK_NORM_BIAS:-0}"
LAYERSCALE_VALUE="${LAYERSCALE_VALUE:-1e-5}"
USE_GATED_MLP="${USE_GATED_MLP:-0}"
HIDDEN_ACT="${HIDDEN_ACT:-gelu}"
ROPE_THETA="${ROPE_THETA:-100.0}"
QA_TOKENS="${QA_TOKENS:-1}"
QA_LAYOUT="${QA_LAYOUT:-split_sep}"
QA_SEP_TOKEN="${QA_SEP_TOKEN:-1}"
QA_FUSE="${QA_FUSE:-add}"
USE_PT_DIST="${USE_PT_DIST:-1}"
USE_PT_GRAD="${USE_PT_GRAD:-0}"
ANSWER_MLP_LAYERS="${ANSWER_MLP_LAYERS:-2}"
ANSWER_POOL="${ANSWER_POOL:-max}"
NEPA2D_POS="${NEPA2D_POS:-1}"
TYPE_SPECIFIC_POS="${TYPE_SPECIFIC_POS:-0}"
TYPE_POS_MAX_LEN="${TYPE_POS_MAX_LEN:-4096}"
MAX_LEN="${MAX_LEN:-4096}"
DUAL_MASK_NEAR="${DUAL_MASK_NEAR:-0.0}"
DUAL_MASK_FAR="${DUAL_MASK_FAR:-0.0}"
DUAL_MASK_WINDOW="${DUAL_MASK_WINDOW:-32}"
DUAL_MASK_TYPE_AWARE="${DUAL_MASK_TYPE_AWARE:-0}"
DUAL_MASK_WARMUP_FRAC="${DUAL_MASK_WARMUP_FRAC:-0.05}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
MIN_LR="${MIN_LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"   # none|cosine (Query parity: none)
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_OPTIMIZER="${RESUME_OPTIMIZER:-1}"
RESUME="${RESUME:-}"
# Pretrain augmentation parity knobs (defaults off, same semantics as Query-NEPA script).
AUG_ROTATE_Z="${AUG_ROTATE_Z:-0}"
AUG_SCALE_MIN="${AUG_SCALE_MIN:-1.0}"
AUG_SCALE_MAX="${AUG_SCALE_MAX:-1.0}"
AUG_TRANSLATE="${AUG_TRANSLATE:-0.0}"
AUG_JITTER_SIGMA="${AUG_JITTER_SIGMA:-0.0}"
AUG_JITTER_CLIP="${AUG_JITTER_CLIP:-0.0}"
AUG_RECOMPUTE_DIST="${AUG_RECOMPUTE_DIST:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
MIXED_PRECISION="${MIXED_PRECISION:-auto}"   # auto/no/fp16/bf16
ACCELERATE_PYTHON="${ACCELERATE_PYTHON:-python}"
ACCELERATE_LAUNCH_MODULE="${ACCELERATE_LAUNCH_MODULE:-accelerate.commands.launch}"

LOG_SUFFIX=""
if [[ "${NUM_PROCESSES}" -gt 1 || "${NUM_MACHINES}" -gt 1 ]]; then
  LOG_SUFFIX=".mr${MACHINE_RANK}"
fi
LOG_PATH="${LOG_ROOT}/${RUN_TAG}${LOG_SUFFIX}.log"

mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${SAVE_DIR}")"
cd "${WORKDIR}"

if [[ -f "${VENV_ACTIVATE}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${CUDA_MODULE}" 2>/dev/null || true
fi

echo "=== PATCH-NEPA PRETRAIN (POINT-ONLY) ===" | tee "${LOG_PATH}"
echo "date=$(date -Is)" | tee -a "${LOG_PATH}"
echo "host=$(hostname)" | tee -a "${LOG_PATH}"
echo "pbs_jobid=${PBS_JOBID:-}" | tee -a "${LOG_PATH}"
echo "workdir=${WORKDIR}" | tee -a "${LOG_PATH}"
echo "run_tag=${RUN_TAG}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CONFIG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "n_point=${N_POINT} n_ray=${N_RAY} use_ray_patch=${USE_RAY_PATCH}" | tee -a "${LOG_PATH}"
echo "patch_embed=${PATCH_EMBED} group_size=${GROUP_SIZE} num_groups=${NUM_GROUPS} serial_order=${SERIAL_ORDER}" | tee -a "${LOG_PATH}"
echo "pt_xyz_key=${PT_XYZ_KEY} pt_dist_key=${PT_DIST_KEY} ablate_point_dist=${ABLATE_POINT_DIST} pt_sample_mode=${PT_SAMPLE_MODE} pt_fps_key=${PT_FPS_KEY} pt_rfps_key=${PT_RFPS_KEY} pt_rfps_m=${PT_RFPS_M} point_order_mode=${POINT_ORDER_MODE}" | tee -a "${LOG_PATH}"
echo "qa: tokens=${QA_TOKENS} layout=${QA_LAYOUT} sep=${QA_SEP_TOKEN} fuse=${QA_FUSE} use_pt_dist=${USE_PT_DIST} use_pt_grad=${USE_PT_GRAD}" | tee -a "${LOG_PATH}"
echo "dual_mask: near=${DUAL_MASK_NEAR} far=${DUAL_MASK_FAR} window=${DUAL_MASK_WINDOW} type_aware=${DUAL_MASK_TYPE_AWARE} warmup_frac=${DUAL_MASK_WARMUP_FRAC}" | tee -a "${LOG_PATH}"
echo "epochs=${EPOCHS} batch=${BATCH} lr=${LR}" | tee -a "${LOG_PATH}"
echo "backbone_mode=${BACKBONE_MODE} qk_norm=${QK_NORM} qk_norm_affine=${QK_NORM_AFFINE} qk_norm_bias=${QK_NORM_BIAS} layerscale=${LAYERSCALE_VALUE} rope_theta=${ROPE_THETA}" | tee -a "${LOG_PATH}"
echo "optimizer: weight_decay=${WEIGHT_DECAY} max_grad_norm=${MAX_GRAD_NORM} lr_scheduler=${LR_SCHEDULER} warmup_epochs=${WARMUP_EPOCHS} min_lr=${MIN_LR}" | tee -a "${LOG_PATH}"
echo "resume: auto_resume=${AUTO_RESUME} resume_optimizer=${RESUME_OPTIMIZER} resume=${RESUME}" | tee -a "${LOG_PATH}"
echo "aug: rotate_z=${AUG_ROTATE_Z} scale=[${AUG_SCALE_MIN},${AUG_SCALE_MAX}] translate=${AUG_TRANSLATE} jitter_sigma=${AUG_JITTER_SIGMA} jitter_clip=${AUG_JITTER_CLIP} recompute_dist=${AUG_RECOMPUTE_DIST}" | tee -a "${LOG_PATH}"
echo "ddp: num_processes=${NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK}" | tee -a "${LOG_PATH}"
echo "ddp: main_process_ip=${MAIN_PROCESS_IP} main_process_port=${MAIN_PROCESS_PORT}" | tee -a "${LOG_PATH}"
echo "effective_global_batch=$((BATCH * NUM_PROCESSES * GRAD_ACCUM)) (batch * num_processes * grad_accum)" | tee -a "${LOG_PATH}"
echo | tee -a "${LOG_PATH}"

TRAIN_ARGS=(
  --mix_config_path "${MIX_CONFIG}"
  --save_dir "$(dirname "${SAVE_DIR}")"
  --run_name "$(basename "${SAVE_DIR}")"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH}"
  --n_point "${N_POINT}"
  --n_ray "${N_RAY}"
  --num_workers "${NUM_WORKERS}"
  --pt_xyz_key "${PT_XYZ_KEY}"
  --pt_dist_key "${PT_DIST_KEY}"
  --ablate_point_dist "${ABLATE_POINT_DIST}"
  --pt_sample_mode "${PT_SAMPLE_MODE}"
  --pt_fps_key "${PT_FPS_KEY}"
  --pt_rfps_key "${PT_RFPS_KEY}"
  --pt_rfps_m "${PT_RFPS_M}"
  --point_order_mode "${POINT_ORDER_MODE}"
  --qa_tokens "${QA_TOKENS}"
  --qa_layout "${QA_LAYOUT}"
  --qa_sep_token "${QA_SEP_TOKEN}"
  --qa_fuse "${QA_FUSE}"
  --use_pt_dist "${USE_PT_DIST}"
  --use_pt_grad "${USE_PT_GRAD}"
  --answer_mlp_layers "${ANSWER_MLP_LAYERS}"
  --answer_pool "${ANSWER_POOL}"
  --nepa2d_pos "${NEPA2D_POS}"
  --type_specific_pos "${TYPE_SPECIFIC_POS}"
  --type_pos_max_len "${TYPE_POS_MAX_LEN}"
  --max_len "${MAX_LEN}"
  --patch_embed "${PATCH_EMBED}"
  --group_size "${GROUP_SIZE}"
  --num_groups "${NUM_GROUPS}"
  --serial_order "${SERIAL_ORDER}"
  --serial_bits "${SERIAL_BITS}"
  --serial_shuffle_within_patch "${SERIAL_SHUFFLE_WITHIN_PATCH}"
  --use_ray_patch "${USE_RAY_PATCH}"
  --ray_pool_mode "${RAY_POOL_MODE}"
  --ray_fuse "${RAY_FUSE}"
  --ray_miss_t "${RAY_MISS_T}"
  --ray_hit_threshold "${RAY_HIT_THRESHOLD}"
  --d_model "${D_MODEL}"
  --n_layers "${N_LAYERS}"
  --n_heads "${N_HEADS}"
  --mlp_ratio "${MLP_RATIO}"
  --backbone_mode "${BACKBONE_MODE}"
  --rope_theta "${ROPE_THETA}"
  --drop_path_rate "${DROP_PATH_RATE}"
  --qk_norm "${QK_NORM}"
  --qk_norm_affine "${QK_NORM_AFFINE}"
  --qk_norm_bias "${QK_NORM_BIAS}"
  --layerscale_value "${LAYERSCALE_VALUE}"
  --use_gated_mlp "${USE_GATED_MLP}"
  --hidden_act "${HIDDEN_ACT}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --min_lr "${MIN_LR}"
  --lr_scheduler "${LR_SCHEDULER}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --grad_accum "${GRAD_ACCUM}"
  --auto_resume "${AUTO_RESUME}"
  --resume_optimizer "${RESUME_OPTIMIZER}"
  --resume "${RESUME}"
  --aug_rotate_z "${AUG_ROTATE_Z}"
  --aug_scale_min "${AUG_SCALE_MIN}"
  --aug_scale_max "${AUG_SCALE_MAX}"
  --aug_translate "${AUG_TRANSLATE}"
  --aug_jitter_sigma "${AUG_JITTER_SIGMA}"
  --aug_jitter_clip "${AUG_JITTER_CLIP}"
  --aug_recompute_dist "${AUG_RECOMPUTE_DIST}"
  --dual_mask_near "${DUAL_MASK_NEAR}"
  --dual_mask_far "${DUAL_MASK_FAR}"
  --dual_mask_window "${DUAL_MASK_WINDOW}"
  --dual_mask_type_aware "${DUAL_MASK_TYPE_AWARE}"
  --dual_mask_warmup_frac "${DUAL_MASK_WARMUP_FRAC}"
  --lr "${LR}"
  --seed "${SEED}"
)

LAUNCH_MIXED_PRECISION="${MIXED_PRECISION}"
if [[ "${LAUNCH_MIXED_PRECISION}" == "auto" ]]; then
  LAUNCH_MIXED_PRECISION="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    print("no")
elif hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    print("bf16")
else:
    print("fp16")
PY
)"
fi
echo "launch_mixed_precision=${LAUNCH_MIXED_PRECISION}" | tee -a "${LOG_PATH}"

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  "${ACCELERATE_PYTHON}" -m "${ACCELERATE_LAUNCH_MODULE}" \
    --multi_gpu \
    --num_processes "${NUM_PROCESSES}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --main_process_ip "${MAIN_PROCESS_IP}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    --mixed_precision "${LAUNCH_MIXED_PRECISION}" \
    -m nepa3d.train.pretrain_patch_nepa \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "${LOG_PATH}"
else
  python -u -m nepa3d.train.pretrain_patch_nepa \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "${LOG_PATH}"
fi

echo "[done] log=${LOG_PATH}" | tee -a "${LOG_PATH}"
