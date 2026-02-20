#!/usr/bin/env bash
set -e

# CPAC UDF evaluation + optional mesh reconstruction (marching cubes) and Chamfer/F-score.
#
# Usage:
#   bash scripts/analysis/nepa3d_cpac_udf_mesh.sh /path/to/ckpt.pt
#
# NOTE: Mesh evaluation is significantly more expensive than pointwise MAE/RMSE.
#       Start with a small mesh_grid_res (e.g. 16/24) and mesh_eval_max_shapes.

CKPT=${1:-""}
if [ -z "$CKPT" ]; then
  echo "Usage: $0 /path/to/ckpt.pt"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

"$PYTHON_BIN" -m nepa3d.analysis.completion_cpac_udf \
  --ckpt "$CKPT" \
  --split "eval" \
  --eval_seed 0 \
  --n_context 256 \
  --n_query 256 \
  --query_source "pool" \
  --mesh_eval 1 \
  --mesh_eval_max_shapes 50 \
  --mesh_grid_res 24 \
  --mesh_chunk_n_query 512 \
  --mesh_mc_level 0.03 \
  --mesh_num_samples 10000 \
  --mesh_fscore_tau 0.01
