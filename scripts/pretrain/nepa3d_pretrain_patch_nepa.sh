#!/usr/bin/env bash
set -euo pipefail

# Patch-token NEPA pretrain (Stage-2, single-node helper).
# This launcher keeps defaults aligned with the active point-only baseline.

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}"

MIX_CFG="${1:-nepa3d/configs/pretrain_mixed_shapenet_pointcloud_only_onepass.yaml}"
SAVE_DIR="${2:-runs/patchnepa_pointonly}"
RUN_NAME="${3:-patchnepa_ptonly_manual_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/patch_nepa_pretrain/${RUN_NAME}}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.log"
mkdir -p "${LOG_ROOT}"

echo "=== PATCH-NEPA PRETRAIN (single-node helper) ===" | tee "${LOG_PATH}"
echo "root=${ROOT_DIR}" | tee -a "${LOG_PATH}"
echo "mix_config=${MIX_CFG}" | tee -a "${LOG_PATH}"
echo "save_dir=${SAVE_DIR}" | tee -a "${LOG_PATH}"
echo "run_name=${RUN_NAME}" | tee -a "${LOG_PATH}"

python -u -m nepa3d.train.pretrain_patch_nepa \
  --mix_config_path "${MIX_CFG}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --epochs "${EPOCHS:-100}" \
  --batch_size "${BATCH:-16}" \
  --n_point "${N_POINT:-1024}" \
  --n_ray "${N_RAY:-0}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --pt_xyz_key "${PT_XYZ_KEY:-pc_xyz}" \
  --pt_dist_key "${PT_DIST_KEY:-pt_dist_pool}" \
  --ablate_point_dist "${ABLATE_POINT_DIST:-1}" \
  --pt_sample_mode "${PT_SAMPLE_MODE:-rfps_cached}" \
  --pt_fps_key "${PT_FPS_KEY:-auto}" \
  --pt_rfps_key "${PT_RFPS_KEY:-auto}" \
  --pt_rfps_m "${PT_RFPS_M:-4096}" \
  --point_order_mode "${POINT_ORDER_MODE:-morton}" \
  --patch_embed "${PATCH_EMBED:-fps_knn}" \
  --group_size "${GROUP_SIZE:-32}" \
  --num_groups "${NUM_GROUPS:-64}" \
  --serial_order "${SERIAL_ORDER:-morton}" \
  --serial_bits "${SERIAL_BITS:-10}" \
  --serial_shuffle_within_patch "${SERIAL_SHUFFLE_WITHIN_PATCH:-0}" \
  --use_ray_patch "${USE_RAY_PATCH:-0}" \
  --ray_pool_mode "${RAY_POOL_MODE:-mean}" \
  --ray_fuse "${RAY_FUSE:-add}" \
  --ray_miss_t "${RAY_MISS_T:-4.0}" \
  --ray_hit_threshold "${RAY_HIT_THRESHOLD:-0.5}" \
  --backbone_mode "${BACKBONE_MODE:-nepa2d}" \
  --qk_norm "${QK_NORM:-1}" \
  --qk_norm_affine "${QK_NORM_AFFINE:-0}" \
  --qk_norm_bias "${QK_NORM_BIAS:-0}" \
  --layerscale_value "${LAYERSCALE_VALUE:-1e-5}" \
  --rope_theta "${ROPE_THETA:-100.0}" \
  --d_model "${D_MODEL:-384}" \
  --n_layers "${N_LAYERS:-12}" \
  --n_heads "${N_HEADS:-6}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --drop_path_rate "${DROP_PATH_RATE:-0.0}" \
  --lr "${LR:-3e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_epochs "${WARMUP_EPOCHS:-0}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --lr_scheduler "${LR_SCHEDULER:-none}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.0}" \
  --auto_resume "${AUTO_RESUME:-1}" \
  --resume_optimizer "${RESUME_OPTIMIZER:-1}" \
  --resume "${RESUME:-}" \
  --aug_rotate_z "${AUG_ROTATE_Z:-0}" \
  --aug_scale_min "${AUG_SCALE_MIN:-1.0}" \
  --aug_scale_max "${AUG_SCALE_MAX:-1.0}" \
  --aug_translate "${AUG_TRANSLATE:-0.0}" \
  --aug_jitter_sigma "${AUG_JITTER_SIGMA:-0.0}" \
  --aug_jitter_clip "${AUG_JITTER_CLIP:-0.0}" \
  --aug_recompute_dist "${AUG_RECOMPUTE_DIST:-0}" \
  --seed "${SEED:-0}" \
  2>&1 | tee -a "${LOG_PATH}"

echo "[done] log=${LOG_PATH}"
