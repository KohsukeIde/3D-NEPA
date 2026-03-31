#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

# Paper-facing fast build:
# keep surf / pc_ctx_bank / udf-distance / thickness carriers,
# drop legacy mesh_qry / pc_qry / ray packs and skip base visibility.
export OUT_ROOT="${OUT_ROOT:-data/shapenet_cache_v2_geo_teacher_fast}"
export WORKERS="${WORKERS:-24}"
export TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC:-1800}"
export TASK_TIMEOUT_GRACE_SEC="${TASK_TIMEOUT_GRACE_SEC:-5}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export MISSING_ONLY="${MISSING_ONLY:-0}"

export N_SURF="${N_SURF:-4096}"
export N_MESH_QRY="${N_MESH_QRY:-0}"
export N_UDF_QRY="${N_UDF_QRY:-4096}"
export N_PC="${N_PC:-1024}"
export N_PC_QRY="${N_PC_QRY:-0}"
export N_RAYS="${N_RAYS:-0}"
export PC_CTX_BANK="${PC_CTX_BANK:-1}"
export MESH_VIS_N_DIRS="${MESH_VIS_N_DIRS:-0}"
export SURF_UDF_GRID="${SURF_UDF_GRID:-96}"
export SURF_UDF_STEPS="${SURF_UDF_STEPS:-48}"

exec bash "${ROOT_DIR}/scripts/preprocess/preprocess_shapenet_v2.sh"
