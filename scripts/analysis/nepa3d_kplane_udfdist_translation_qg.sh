#!/bin/bash
#PBS -l rt_QG=1
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -N kp_trn

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${WORKDIR}/.venv/bin/activate}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9}"
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then set -a; source "${ENV_FILE}"; set +a; fi
RUN_TAG="${RUN_TAG:-kplane_udfdist_translation_$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:?set CKPT=...}"
MIX_CONFIG="${MIX_CONFIG:-nepa3d/configs/shapenet_unpaired_mix_v2_cqa_udfdist.yaml}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/logs/kplane_udfdist_completion}"
OUT_JSON="${OUT_JSON:-${WORKDIR}/results/kplane_udfdist_completion/${RUN_TAG}.json}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
N_CTX="${N_CTX:-2048}"
N_QRY="${N_QRY:-64}"
BATCH="${BATCH:-4}"
MAX_SHAPES="${MAX_SHAPES:-64}"
SPLIT_OVERRIDE="${SPLIT_OVERRIDE:-eval}"
EVAL_SAMPLE_MODE="${EVAL_SAMPLE_MODE:-random}"
GRID_RES="${GRID_RES:-16}"
CHUNK_N_QUERY="${CHUNK_N_QUERY:-128}"
TAU_LIST="${TAU_LIST:-0.01,0.02,0.05}"
MESH_EVAL="${MESH_EVAL:-1}"
MC_LEVEL="${MC_LEVEL:-0.05}"
MESH_NUM_SAMPLES="${MESH_NUM_SAMPLES:-10000}"
EXPORT_ASSETS="${EXPORT_ASSETS:-1}"
ASSETS_ROOT="${ASSETS_ROOT:-}"
mkdir -p "${LOG_ROOT}" "$(dirname "${OUT_JSON}")"
LOG_PATH="${LOG_ROOT}/${RUN_TAG}.log"
cd "${WORKDIR}"
[[ -f "${VENV_ACTIVATE}" ]] && source "${VENV_ACTIVATE}"
source /etc/profile.d/modules.sh 2>/dev/null || true
command -v module >/dev/null 2>&1 && module load "${CUDA_MODULE}" 2>/dev/null || true
python -u -m nepa3d.tracks.kplane.analysis.completion_udfdist_worldv3 \
  --ckpt "${CKPT}" \
  --mix_config_path "${MIX_CONFIG}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --n_ctx "${N_CTX}" \
  --n_qry "${N_QRY}" \
  --batch_size "${BATCH}" \
  --max_shapes "${MAX_SHAPES}" \
  --split_override "${SPLIT_OVERRIDE}" \
  --eval_sample_mode "${EVAL_SAMPLE_MODE}" \
  --grid_res "${GRID_RES}" \
  --chunk_n_query "${CHUNK_N_QUERY}" \
  --tau_list "${TAU_LIST}" \
  --mesh_eval "${MESH_EVAL}" \
  --mc_level "${MC_LEVEL}" \
  --mesh_num_samples "${MESH_NUM_SAMPLES}" \
  --export_assets "${EXPORT_ASSETS}" \
  --assets_root "${ASSETS_ROOT}" \
  --output_json "${OUT_JSON}" \
  2>&1 | tee "${LOG_PATH}"
