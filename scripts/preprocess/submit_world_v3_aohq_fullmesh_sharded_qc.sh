#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SINGLE_SUBMIT="${SCRIPT_DIR}/submit_world_v3_mesh_aux_qc.sh"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ ! -x "${SINGLE_SUBMIT}" ]]; then
  echo "[error] missing submit helper: ${SINGLE_SUBMIT}"
  exit 1
fi

RUN_TAG_BASE="${RUN_TAG_BASE:-world_v3_aohq_fullmesh_$(date +%Y%m%d_%H%M%S)}"
SRC_CACHE_ROOT="${SRC_CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1}"
DST_CACHE_ROOT="${DST_CACHE_ROOT:-data/shapenet_unpaired_cache_v2_20260311_worldvis_drop1_aohq_fullmesh_sharded_r3}"
GROUP_LIST="${GROUP_LIST:-qgah50055}"
WALLTIME="${WALLTIME:-12:00:00}"
AO_RAYS="${AO_RAYS:-128}"
AO_EPS="${AO_EPS:-1e-4}"
AO_MAX_T="${AO_MAX_T:-2.5}"
AO_BATCH_SIZE="${AO_BATCH_SIZE:-64}"
REFRESH="${REFRESH:-0}"
OUT_DIR_BASE="${OUT_DIR_BASE:-${WORKDIR}/results/data_freeze/${RUN_TAG_BASE}}"
TRAIN_NUM_SHARDS="${TRAIN_NUM_SHARDS:-32}"
EVAL_NUM_SHARDS="${EVAL_NUM_SHARDS:-16}"

mkdir -p "${OUT_DIR_BASE}"

submit_split() {
  local split="$1"
  local num_shards="$2"
  local jobs=()
  for (( shard=0; shard<num_shards; shard++ )); do
    local run_tag="${RUN_TAG_BASE}_${split}_s$(printf '%02d' "${shard}")of$(printf '%02d' "${num_shards}")"
    local out_dir="${OUT_DIR_BASE}/${split}/shard_${shard}"
    mkdir -p "${out_dir}"
    local copy_meta="0"
    if [[ "${split}" == "train_mesh" && "${shard}" == "0" ]]; then
      copy_meta="1"
    fi
    local out
    out="$(
      env \
        WORKDIR="${WORKDIR}" \
        GROUP_LIST="${GROUP_LIST}" \
        WALLTIME="${WALLTIME}" \
        RUN_TAG="${run_tag}" \
        SRC_CACHE_ROOT="${SRC_CACHE_ROOT}" \
        DST_CACHE_ROOT="${DST_CACHE_ROOT}" \
        SPLIT="${split}" \
        LIMIT="0" \
        SAMPLE_SEED="0" \
        NUM_SHARDS="${num_shards}" \
        SHARD_INDEX="${shard}" \
        REFRESH="${REFRESH}" \
        COPY_META="${copy_meta}" \
        COMPUTE_AO_HQ="1" \
        AO_RAYS="${AO_RAYS}" \
        AO_EPS="${AO_EPS}" \
        AO_MAX_T="${AO_MAX_T}" \
        AO_BATCH_SIZE="${AO_BATCH_SIZE}" \
        COMPUTE_HKS="0" \
        OUT_DIR="${out_dir}" \
        OUTPUT_JSON="${out_dir}/mesh_aux_summary.json" \
        "${SINGLE_SUBMIT}"
    )"
    printf '%s\n' "${out}"
    local jid
    jid="$(printf '%s\n' "${out}" | awk '/^\[submitted\]/{print $2; exit}')"
    if [[ -z "${jid:-}" ]]; then
      echo "[error] failed to parse job id for split=${split} shard=${shard}" >&2
      exit 1
    fi
    jobs+=("${jid}")
  done
  printf '%s\n' "${jobs[*]}"
}

echo "[run_tag_base] ${RUN_TAG_BASE}"
echo "[dst_cache_root] ${DST_CACHE_ROOT}"

train_jobs="$(submit_split train_mesh "${TRAIN_NUM_SHARDS}" | tail -n 1)"
eval_jobs="$(submit_split eval "${EVAL_NUM_SHARDS}" | tail -n 1)"

echo "[train_mesh_jobs] ${train_jobs}"
echo "[eval_jobs] ${eval_jobs}"
echo "[all_jobs_colon] $(printf '%s %s' "${train_jobs}" "${eval_jobs}" | xargs | tr ' ' ':')"
