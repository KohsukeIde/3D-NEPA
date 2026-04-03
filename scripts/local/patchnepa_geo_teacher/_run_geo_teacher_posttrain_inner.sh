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
  "${PARTSEG_SAVE_ROOT:-${ROOT_DIR}/runs/patchpart_itachi}" \
  "${PARTSEG_LOG_ROOT:-${LOG_ROOT}/partseg}" \
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
    status="$(awk -F= '/^status=/{print $2; exit}' <<< "${status_output}")"
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
  if [[ "${FT_DATA_FORMAT:-scan_h5}" == "scan_h5" ]]; then
    log "skip ScanObjectNN variant prep (FT_DATA_FORMAT=scan_h5)"
    return 0
  fi
  if [[ "${RUN_SCANOBJECTNN_PREP:-1}" != "1" ]]; then
    log "skip ScanObjectNN variant prep (RUN_SCANOBJECTNN_PREP=0)"
    return 0
  fi
  local obj_bg="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
  local obj_only="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
  local pb="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
  while true; do
    if variant_ready "${obj_bg}" && variant_ready "${obj_only}" && variant_ready "${pb}"; then
      log "ScanObjectNN variant caches ready"
      return 0
    fi
    local scan_status
    scan_status="$(
      env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
        bash "${ROOT_DIR}/scripts/local/patchnepa_data/status_scanobjectnn_variants_local.sh" || true
    )"
    if [[ "$(awk -F= '/^status=/{print $2; exit}' <<< "${scan_status}")" == "running" ]]; then
      log "waiting for ScanObjectNN variant prep to finish"
      sleep 60
      continue
    fi
    log "building ScanObjectNN variant caches before downstream FT"
    env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
      FOREGROUND=1 \
      bash "${ROOT_DIR}/scripts/local/patchnepa_data/run_scanobjectnn_variants_local.sh" \
      >> "${LOG_FILE}" 2>&1
  done
}

ensure_shapenetpart_data() {
  if [[ "${RUN_SHAPENETPART_FT:-1}" != "1" ]]; then
    return 0
  fi
  local root="${SHAPENETPART_ROOT:-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
  if [[ -f "${root}/synsetoffset2category.txt" ]]; then
    log "ShapeNetPart ready root=${root}"
    return 0
  fi
  log "preparing ShapeNetPart root=${root}"
  env -u LOG_ROOT -u LOG_FILE -u PID_FILE -u TMUX_SESSION -u LAUNCH_MODE \
    DATA_ROOT="${SHAPENETPART_DATA_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data}" \
    TARGET_DIR="${SHAPENETPART_TARGET_DIR:-/mnt/urashima/users/minesawa/3D-NEPA-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}" \
    ZIP_PATH="${SHAPENETPART_ZIP_PATH:-/mnt/urashima/users/minesawa/3D-NEPA-data/downloads/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip}" \
    bash "${ROOT_DIR}/scripts/local/patchnepa_data/prepare_shapenetpart_local.sh" \
    >> "${LOG_FILE}" 2>&1
  if [[ ! -f "${root}/synsetoffset2category.txt" ]]; then
    log "ShapeNetPart setup failed root=${root}"
    exit 1
  fi
}

run_scan_ft_blocking() {
  local variant="$1"
  local visible_gpus="$2"
  local cache_root="$3"
  local scan_h5_root="$4"
  local scan_variant="$5"
  local run_name="${PRETRAIN_RUN_TAG}__ft_${variant}_300ep"
  local job_log="${FT_LOG_ROOT}/${run_name}.log"
  local data_format="${FT_DATA_FORMAT:-scan_h5}"
  local input_root="${cache_root}"
  if [[ "${data_format}" == "scan_h5" ]]; then
    input_root="${scan_h5_root}"
  fi
  log "launch ft variant=${variant} gpus=${visible_gpus} data_format=${data_format} input_root=${input_root}"
  (
    set -euo pipefail
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${visible_gpus}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export CKPT="${CKPT_PATH}"
    export DATA_FORMAT="${data_format}"
    export CACHE_ROOT="${cache_root}"
    export SCAN_H5_ROOT="${scan_h5_root}"
    export SCAN_VARIANT="${scan_variant}"
    export RUN_NAME="${run_name}"
    export SAVE_DIR="${FT_SAVE_ROOT}"
    export EPOCHS="${FT_EPOCHS:-300}"
    export BATCH="${FT_BATCH:-64}"
    export NUM_WORKERS="${FT_NUM_WORKERS:-8}"
    export NPROC_PER_NODE="${FT_NPROC_PER_NODE:-4}"
    export USE_WANDB="${FT_USE_WANDB:-1}"
    export WANDB_PROJECT="${FT_WANDB_PROJECT:-patchnepa-geo-teacher-scanobjectnn}"
    export WANDB_ENTITY="${FT_WANDB_ENTITY:-}"
    export WANDB_GROUP="${FT_WANDB_GROUP:-itachi_geo_teacher_ft}"
    export WANDB_RUN_NAME="${run_name}"
    export WANDB_TAGS="local,itachi,geo_teacher,scanobjectnn,${variant},ddp4"
    export WANDB_MODE="${FT_WANDB_MODE:-online}"
    export VAL_SPLIT_MODE="${FT_VAL_SPLIT_MODE:-pointmae}"
    export AUG_EVAL="${FT_AUG_EVAL:-1}"
    export MC_EVAL_K_TEST="${FT_MC_EVAL_K_TEST:-10}"
    export MODEL_SOURCE="${FT_MODEL_SOURCE:-patchnepa}"
    export POOLING="${FT_POOLING:-cls_max}"
    export HEAD_MODE="${FT_HEAD_MODE:-pointmae_mlp}"
    export PATCHNEPA_CLS_TOKEN_SOURCE="${FT_PATCHNEPA_CLS_TOKEN_SOURCE:-last_q}"
    export PATCHNEPA_FREEZE_PATCH_EMBED="${FT_PATCHNEPA_FREEZE_PATCH_EMBED:-1}"
    bash "${ROOT_DIR}/scripts/finetune/patchnepa_scanobjectnn_finetune.sh"
  ) > "${job_log}" 2>&1
  log "job_done name=ft_${variant}"
}

run_shapenetpart_ft_blocking() {
  local visible_gpus="$1"
  local run_name="${PRETRAIN_RUN_TAG}__ft_shapenetpart_300ep"
  local job_log="${PARTSEG_LOG_ROOT}/${run_name}.log"
  local root="${SHAPENETPART_ROOT:-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
  log "launch partseg gpus=${visible_gpus} root=${root}"
  (
    set -euo pipefail
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${visible_gpus}"
    export PYTHON_BIN="${PYTHON_BIN}"
    export ROOT="${root}"
    export CKPT="${CKPT_PATH}"
    export RUN_NAME="${run_name}"
    export SAVE_DIR="${PARTSEG_SAVE_ROOT}"
    export EPOCHS="${PARTSEG_EPOCHS:-300}"
    export BATCH="${PARTSEG_BATCH:-64}"
    export BATCH_MODE="${PARTSEG_BATCH_MODE:-global}"
    export NUM_WORKERS="${PARTSEG_NUM_WORKERS:-8}"
    export NPROC_PER_NODE="${PARTSEG_NPROC_PER_NODE:-4}"
    export USE_WANDB="${PARTSEG_USE_WANDB:-1}"
    export WANDB_PROJECT="${PARTSEG_WANDB_PROJECT:-patchnepa-geo-teacher-shapenetpart}"
    export WANDB_ENTITY="${PARTSEG_WANDB_ENTITY:-}"
    export WANDB_GROUP="${PARTSEG_WANDB_GROUP:-itachi_geo_teacher_partseg}"
    export WANDB_RUN_NAME="${run_name}"
    export WANDB_TAGS="local,itachi,geo_teacher,shapenetpart,ddp4"
    export WANDB_MODE="${PARTSEG_WANDB_MODE:-online}"
    export PATCHNEPA_FT_MODE="${PARTSEG_PATCHNEPA_FT_MODE:-q_only}"
    export PATCHNEPA_FREEZE_PATCH_EMBED="${PARTSEG_PATCHNEPA_FREEZE_PATCH_EMBED:-1}"
    bash "${ROOT_DIR}/scripts/finetune/patchnepa_shapenetpart_finetune.sh"
  ) > "${job_log}" 2>&1
  log "job_done name=ft_shapenetpart"
}

run_route_b_blocking() {
  local gpu="$1"
  local run_stem="${PRETRAIN_RUN_TAG}__routeb"
  local orchestration_log="${LOG_ROOT}/${run_stem}.log"
  log "launch route-b eval gpu=${gpu}"
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

    if [[ "${RUN_CURVATURE_PROBE:-1}" == "1" ]]; then
      export CKPT="${CKPT_PATH}"
      export CACHE_ROOT="${PROBE_CACHE_ROOT:-data/shapenet_cache_v2_20260401_worldvis}"
      export PROBE_TARGET="curvature"
      export SAVE_DIR="${PROBE_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_probe_itachi}"
      export LOG_ROOT="${PROBE_LOG_ROOT:-${LOG_ROOT}/cqa_probe}"
      export RUN_TAG="${PRETRAIN_RUN_TAG}__curvature_probe"
      export OUT_JSON="${PROBE_RESULT_ROOT:-${ROOT_DIR}/results/cqa_probe_itachi}/${RUN_TAG}.json"
      export TRAIN_SPLIT="${PROBE_TRAIN_SPLIT:-train}"
      export EVAL_SPLIT="${PROBE_EVAL_SPLIT:-test}"
      export MAX_STEPS="${PROBE_MAX_STEPS:-5000}"
      export EVAL_EVERY="${PROBE_EVAL_EVERY:-500}"
      export BATCH="${PROBE_BATCH:-8}"
      export NUM_WORKERS="${PROBE_NUM_WORKERS:-4}"
      export N_CTX="${PROBE_N_CTX:-2048}"
      export N_QRY="${PROBE_N_QRY:-64}"
      export MAX_TRAIN_SAMPLES="${PROBE_MAX_TRAIN_SAMPLES:-0}"
      export MAX_EVAL_SAMPLES="${PROBE_MAX_EVAL_SAMPLES:-128}"
      export CONTROLS="${PROBE_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
      bash "${ROOT_DIR}/scripts/eval/nepa3d_cqa_geo_probe_qg.sh"
    fi
  ) > "${orchestration_log}" 2>&1
  log "job_done name=route_b"
}

log "start geo-teacher posttrain chain run=${PRETRAIN_RUN_TAG}"
wait_for_pretrain
require_ckpt
ensure_scanobjectnn_variants
ensure_shapenetpart_data

if [[ "${RUN_ROUTE_A:-1}" == "1" ]]; then
  IFS=',' read -r -a ft_order <<< "${FT_VARIANT_ORDER:-obj_bg,obj_only,pb_t50_rs}"
  for variant in "${ft_order[@]}"; do
    case "${variant}" in
      obj_bg)
        run_scan_ft_blocking obj_bg "${FT_VISIBLE_GPUS:-0,1,2,3}" \
          "${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}" \
          "${FT_SCAN_H5_ROOT_OBJ_BG:-data/ScanObjectNN/h5_files/main_split}" \
          obj_bg
        ;;
      obj_only)
        run_scan_ft_blocking obj_only "${FT_VISIBLE_GPUS:-0,1,2,3}" \
          "${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}" \
          "${FT_SCAN_H5_ROOT_OBJ_ONLY:-data/ScanObjectNN/h5_files/main_split_nobg}" \
          obj_only
        ;;
      pb_t50_rs)
        run_scan_ft_blocking pb_t50_rs "${FT_VISIBLE_GPUS:-0,1,2,3}" \
          "${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}" \
          "${FT_SCAN_H5_ROOT_PB_T50_RS:-data/ScanObjectNN/h5_files/main_split}" \
          pb_t50_rs
        ;;
      *)
        log "unknown FT variant in FT_VARIANT_ORDER: ${variant}"
        exit 1
        ;;
    esac
  done
else
  log "skip Route A FT runs (RUN_ROUTE_A=0)"
fi

if [[ "${RUN_SHAPENETPART_FT:-1}" == "1" ]]; then
  run_shapenetpart_ft_blocking "${PARTSEG_VISIBLE_GPUS:-0,1,2,3}"
else
  log "skip ShapeNetPart FT (RUN_SHAPENETPART_FT=0)"
fi

if [[ "${RUN_ROUTE_B:-1}" == "1" ]]; then
  run_route_b_blocking "${EVAL_GPU:-0}"
else
  log "skip Route B eval run (RUN_ROUTE_B=0)"
fi

log "geo-teacher posttrain chain completed"
