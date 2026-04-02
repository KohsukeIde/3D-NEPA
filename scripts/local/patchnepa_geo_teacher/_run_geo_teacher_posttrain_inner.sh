#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "${ROOT_DIR}"

LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/geo_teacher_posttrain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/geo_teacher_posttrain.pid}"
PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi/${PRETRAIN_RUN_TAG}}"
CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/ckpt_final.pt}"
PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"

mkdir -p "${LOG_ROOT}" "${FT_SAVE_ROOT:-${ROOT_DIR}/runs/patchcls_itachi}" \
  "${FT_LOG_ROOT:-${LOG_ROOT}/finetune}" \
  "${SUITE_LOG_ROOT:-${LOG_ROOT}/cqa_multitype}" \
  "${SUITE_RESULT_ROOT:-${ROOT_DIR}/results/cqa_multitype_itachi}" \
  "${COMPLETION_LOG_ROOT:-${LOG_ROOT}/cqa_completion}" \
  "${COMPLETION_RESULT_ROOT:-${ROOT_DIR}/results/cqa_completion_itachi}"

cleanup() {
  rc=$?
  printf "[launcher] %s exit_code=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${rc}" >> "${LOG_FILE}"
  rm -f "${PID_FILE}"
  exit "${rc}"
}

trap cleanup EXIT

echo "$$" > "${PID_FILE}"

log() {
  printf "[launcher] %s %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "${LOG_FILE}"
}

count_npz() {
  local root="$1"
  local split="$2"
  find "${root}/${split}" -type f -name '*.npz' 2>/dev/null | wc -l || true
}

variant_ready() {
  local root="$1"
  local tr_n te_n
  tr_n="$(count_npz "${root}" train)"
  te_n="$(count_npz "${root}" test)"
  [[ "${tr_n}" -gt 0 && "${te_n}" -gt 0 ]]
}

wait_for_pretrain() {
  if [[ "${WAIT_FOR_PRETRAIN:-1}" != "1" ]]; then
    return 0
  fi
  while true; do
    local status_output status
    status_output="$(
      env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u SAVE_ROOT -u SAVE_DIR -u TMUX_SESSION -u LAUNCH_MODE \
        RUN_TAG="${PRETRAIN_RUN_TAG}" \
        bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/status_geo_teacher_pretrain_local.sh" || true
    )"
    status="$(printf "%s\n" "${status_output}" | awk -F= '/^status=/{print $2; exit}')"
    if [[ "${status}" == "running" ]]; then
      log "waiting for pretrain run=${PRETRAIN_RUN_TAG} to finish"
      sleep "${PRETRAIN_POLL_SEC:-60}"
      continue
    fi
    break
  done
}

require_ckpt() {
  if [[ ! -f "${CKPT_PATH}" ]]; then
    log "missing checkpoint: ${CKPT_PATH}"
    exit 1
  fi
  log "using checkpoint=${CKPT_PATH}"
}

ensure_scanobjectnn_variants() {
  if [[ "${RUN_SCANOBJECTNN_PREP:-1}" != "1" ]]; then
    log "skip ScanObjectNN variant prep (RUN_SCANOBJECTNN_PREP=0)"
    return 0
  fi
  local obj_bg="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
  local obj_only="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
  local pb="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
  if variant_ready "${obj_bg}" && variant_ready "${obj_only}" && variant_ready "${pb}"; then
    log "ScanObjectNN variant caches already ready"
    return 0
  fi
  log "building ScanObjectNN variant caches before downstream FT"
  env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
    FOREGROUND=1 \
    bash "${ROOT_DIR}/scripts/local/patchnepa_data/run_scanobjectnn_variants_local.sh" \
    >> "${LOG_FILE}" 2>&1
}

launch_scan_ft() {
  local variant="$1"
  local gpu="$2"
  local cache_root="$3"
  local scan_variant="$4"
  local run_name="${PRETRAIN_RUN_TAG}__ft_${variant}_300ep"
  local job_log="${FT_LOG_ROOT}/${run_name}.log"
  printf "[launcher] %s launch ft variant=%s gpu=%s cache_root=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${variant}" "${gpu}" "${cache_root}" \
    | tee -a "${LOG_FILE}" >&2
  (
    set -euo pipefail
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export CKPT="${CKPT_PATH}"
    export CACHE_ROOT="${cache_root}"
    export SCAN_VARIANT="${scan_variant}"
    export RUN_NAME="${run_name}"
    export SAVE_DIR="${FT_SAVE_ROOT}"
    export EPOCHS="${FT_EPOCHS:-300}"
    export BATCH="${FT_BATCH:-64}"
    export NUM_WORKERS="${FT_NUM_WORKERS:-8}"
    export NPROC_PER_NODE=1
    export USE_WANDB="${FT_USE_WANDB:-1}"
    export WANDB_PROJECT="${FT_WANDB_PROJECT:-patchnepa-finetune}"
    export WANDB_ENTITY="${FT_WANDB_ENTITY:-}"
    export WANDB_GROUP="${FT_WANDB_GROUP:-itachi_geo_teacher_ft}"
    export WANDB_RUN_NAME="${run_name}"
    export WANDB_TAGS="local,itachi,geo_teacher,scanobjectnn,${variant}"
    export WANDB_MODE="${FT_WANDB_MODE:-offline}"
    export VAL_SPLIT_MODE="${FT_VAL_SPLIT_MODE:-pointmae}"
    export AUG_EVAL="${FT_AUG_EVAL:-1}"
    export MC_EVAL_K_TEST="${FT_MC_EVAL_K_TEST:-10}"
    bash "${ROOT_DIR}/scripts/finetune/patchnepa_scanobjectnn_finetune.sh"
  ) > "${job_log}" 2>&1 &
  LAST_LAUNCHED_PID="$!"
}

launch_route_b() {
  local gpu="$1"
  local run_stem="${PRETRAIN_RUN_TAG}__routeb"
  local orchestration_log="${LOG_ROOT}/${run_stem}.log"
  printf "[launcher] %s launch route-b eval gpu=%s\n" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${gpu}" \
    | tee -a "${LOG_FILE}" >&2
  (
    set -euo pipefail
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    export WORKDIR="${ROOT_DIR}"
    export VENV_ACTIVATE=""

    export RUN_TAG="${PRETRAIN_RUN_TAG}__multitype_same_offdiag"
    export CKPT="${CKPT_PATH}"
    export SAME_MIX_CONFIG="${SAME_MIX_CONFIG}"
    export OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG}"
    export LOG_ROOT="${SUITE_LOG_ROOT}"
    export OUT_JSON="${SUITE_RESULT_ROOT}/${RUN_TAG}.json"
    export OUT_CSV="${SUITE_RESULT_ROOT}/${RUN_TAG}.csv"
    export OUT_MD="${SUITE_RESULT_ROOT}/${RUN_TAG}.md"
    export DEVICE="cuda"
    export BATCH="${SUITE_BATCH:-16}"
    export NUM_WORKERS="${SUITE_NUM_WORKERS:-4}"
    export N_CTX="${SUITE_N_CTX:-2048}"
    export N_QRY="${SUITE_N_QRY:-64}"
    export MAX_SAMPLES="${SUITE_MAX_SAMPLES:-4995}"
    export SPLIT_OVERRIDE="${SUITE_SPLIT_OVERRIDE:-test}"
    export TASKS="${SUITE_TASKS:-udf_distance,mesh_normal_unsigned}"
    export CONTROLS="${SUITE_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
    export QUERY_ORDER="${SUITE_QUERY_ORDER:-sampled}"
    bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_multitype_suite_qg.sh"

    export CKPT="${CKPT_PATH}"
    export MIX_CONFIG="${SAME_MIX_CONFIG}"
    export LOG_ROOT="${COMPLETION_LOG_ROOT}"
    export RUN_TAG="${PRETRAIN_RUN_TAG}__completion_same"
    export OUT_JSON="${COMPLETION_RESULT_ROOT}/${RUN_TAG}.json"
    export DEVICE="cuda"
    export BATCH="${COMPLETION_BATCH:-4}"
    export MAX_SHAPES="${COMPLETION_MAX_SHAPES:-16}"
    export SPLIT_OVERRIDE="${COMPLETION_SPLIT_OVERRIDE:-test}"
    export N_CTX="${COMPLETION_N_CTX:-2048}"
    export N_QRY="${COMPLETION_N_QRY:-64}"
    export GRID_RES="${COMPLETION_GRID_RES:-12}"
    export CHUNK_N_QUERY="${COMPLETION_CHUNK_N_QUERY:-64}"
    export TAU_LIST="${COMPLETION_TAU_LIST:-0.01,0.02,0.05}"
    bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_udfdist_completion_qg.sh"

    export CKPT="${CKPT_PATH}"
    export MIX_CONFIG="${OFFDIAG_MIX_CONFIG}"
    export LOG_ROOT="${COMPLETION_LOG_ROOT}"
    export RUN_TAG="${PRETRAIN_RUN_TAG}__completion_offdiag"
    export OUT_JSON="${COMPLETION_RESULT_ROOT}/${RUN_TAG}.json"
    export DEVICE="cuda"
    export BATCH="${COMPLETION_BATCH:-4}"
    export MAX_SHAPES="${COMPLETION_MAX_SHAPES:-16}"
    export SPLIT_OVERRIDE="${COMPLETION_SPLIT_OVERRIDE:-test}"
    export N_CTX="${COMPLETION_N_CTX:-2048}"
    export N_QRY="${COMPLETION_N_QRY:-64}"
    export GRID_RES="${COMPLETION_GRID_RES:-12}"
    export CHUNK_N_QUERY="${COMPLETION_CHUNK_N_QUERY:-64}"
    export TAU_LIST="${COMPLETION_TAU_LIST:-0.01,0.02,0.05}"
    bash "${ROOT_DIR}/scripts/analysis/nepa3d_cqa_udfdist_completion_qg.sh"
  ) > "${orchestration_log}" 2>&1 &
  LAST_LAUNCHED_PID="$!"
}

log "start geo-teacher posttrain chain run=${PRETRAIN_RUN_TAG}"
wait_for_pretrain
require_ckpt
ensure_scanobjectnn_variants

declare -a child_pids=()
declare -a child_names=()

if [[ "${RUN_ROUTE_A:-1}" == "1" ]]; then
  IFS=',' read -r -a ft_gpus <<< "${FT_GPU_IDS:-0,1,2}"
  if [[ "${#ft_gpus[@]}" -lt 3 ]]; then
    log "FT_GPU_IDS must contain 3 GPU ids for obj_bg,obj_only,pb_t50_rs"
    exit 1
  fi
  launch_scan_ft obj_bg "${ft_gpus[0]}" "${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}" obj_bg
  child_pids+=("${LAST_LAUNCHED_PID}")
  child_names+=("ft_obj_bg")
  launch_scan_ft obj_only "${ft_gpus[1]}" "${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}" obj_only
  child_pids+=("${LAST_LAUNCHED_PID}")
  child_names+=("ft_obj_only")
  launch_scan_ft pb_t50_rs "${ft_gpus[2]}" "${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}" pb_t50_rs
  child_pids+=("${LAST_LAUNCHED_PID}")
  child_names+=("ft_pb_t50_rs")
else
  log "skip Route A FT launches (RUN_ROUTE_A=0)"
fi

if [[ "${RUN_ROUTE_B:-1}" == "1" ]]; then
  launch_route_b "${EVAL_GPU:-3}"
  child_pids+=("${LAST_LAUNCHED_PID}")
  child_names+=("route_b")
else
  log "skip Route B eval launches (RUN_ROUTE_B=0)"
fi

overall_rc=0
for i in "${!child_pids[@]}"; do
  pid="${child_pids[$i]}"
  name="${child_names[$i]}"
  if wait "${pid}"; then
    log "job_done name=${name} pid=${pid}"
  else
    rc=$?
    overall_rc="${rc}"
    log "job_failed name=${name} pid=${pid} rc=${rc}"
  fi
done

if [[ "${overall_rc}" -ne 0 ]]; then
  exit "${overall_rc}"
fi

log "geo-teacher posttrain chain completed"
