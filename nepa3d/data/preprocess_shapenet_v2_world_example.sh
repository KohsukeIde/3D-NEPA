#!/usr/bin/env bash
set -euo pipefail

# Example launcher for the v2 ShapeNet world-package generator.
# This intentionally enables the richer extras so that the resulting NPZs
# can support future answer-schema / query-schema experiments without
# another full regeneration pass.
#
# Usage:
#   bash nepa3d/data/preprocess_shapenet_v2_world_example.sh \
#     /path/to/ShapeNetCore.v2 /path/to/out_root

SHAPENET_ROOT=${1:-}
OUT_ROOT=${2:-}
if [[ -z "${SHAPENET_ROOT}" || -z "${OUT_ROOT}" ]]; then
  echo "Usage: $0 SHAPENET_ROOT OUT_ROOT" >&2
  exit 1
fi

python -m nepa3d.data.preprocess_shapenet_v2 \
  --shapenet_root "${SHAPENET_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --train_ratio 0.9 \
  --seed 0 \
  --num_workers 8 \
  --n_surf 8192 \
  --n_mesh_qry 2048 \
  --n_udf_qry 8192 \
  --n_pc 2048 \
  --n_pc_qry 1024 \
  --n_rays 4096 \
  --pc_ctx_bank 4 \
  --pc_view_crop 0.5 \
  --pc_noise_std 0.005 \
  --pc_dropout 0.1 \
  --udf_near_ratio 0.5 \
  --udf_near_std 0.05 \
  --udf_probe_deltas 0.01,0.02,0.05 \
  --curvature_knn 20 \
  --pca_knn 20 \
  --mesh_vis_n_dirs 8 \
  --mesh_vis_eps 1e-4 \
  --mesh_vis_max_t 2.5 \
  --strict_udf_surface 1 \
  --surf_udf_grid 128 \
  --surf_udf_dilate 1 \
  --surf_udf_max_t 2.0 \
  --surf_udf_eps 1e-4 \
  --surf_udf_steps 64 \
  --surf_udf_tol 1e-4 \
  --surf_udf_min_step 1e-4
