#!/usr/bin/env bash
set -euo pipefail

# Patch-token NEPA pretrain (Stage-2).
# - Mixed pretrain cache loader with return_raw=1
# - Point patch tokens (+ optional ray-to-patch binding)
# - NEPA next-embedding prediction on patch sequence

cd "$(dirname "$0")/../.."
export PYTHONPATH="$(pwd)"

MIX_CFG=${1:-nepa3d/configs/shapenet_unpaired_mix.yaml}
SAVE_DIR=${2:-runs_patch_nepa}
RUN_NAME=${3:-patchnepa_serial_raybind}

python -m nepa3d.train.pretrain_patch_nepa \
  --mix_config_path "${MIX_CFG}" \
  --save_dir "${SAVE_DIR}" \
  --run_name "${RUN_NAME}" \
  --epochs "${EPOCHS:-50}" \
  --batch_size "${BATCH:-96}" \
  --n_point "${N_POINT:-1024}" \
  --n_ray "${N_RAY:-1024}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --patch_embed "${PATCH_EMBED:-fps_knn}" \
  --group_size "${GROUP_SIZE:-32}" \
  --num_groups "${NUM_GROUPS:-64}" \
  --serial_order "${SERIAL_ORDER:-morton}" \
  --serial_bits "${SERIAL_BITS:-10}" \
  --serial_shuffle_within_patch "${SERIAL_SHUFFLE_WITHIN_PATCH:-0}" \
  --use_ray_patch "${USE_RAY_PATCH:-1}" \
  --ray_pool_mode "${RAY_POOL_MODE:-mean}" \
  --ray_fuse "${RAY_FUSE:-add}" \
  --ray_miss_t "${RAY_MISS_T:-4.0}" \
  --ray_hit_threshold "${RAY_HIT_THRESHOLD:-0.5}" \
  --backbone_mode "${BACKBONE_MODE:-nepa2d}" \
  --rope_theta "${ROPE_THETA:-10000}" \
  --d_model "${D_MODEL:-384}" \
  --n_layers "${N_LAYERS:-12}" \
  --n_heads "${N_HEADS:-6}" \
  --mlp_ratio "${MLP_RATIO:-4.0}" \
  --lr "${LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-0.05}" \
  --warmup_epochs "${WARMUP_EPOCHS:-0}" \
  --min_lr "${MIN_LR:-1e-6}" \
  --seed "${SEED:-0}"
