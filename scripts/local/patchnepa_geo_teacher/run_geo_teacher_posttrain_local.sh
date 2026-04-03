#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

PRETRAIN_RUN_TAG="${PRETRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_post}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${PRETRAIN_RUN_TAG}.posttrain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${PRETRAIN_RUN_TAG}.posttrain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${PRETRAIN_RUN_TAG//[^[:alnum:]_]/_}_post}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PRETRAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
export PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${PRETRAIN_SAVE_ROOT}/${PRETRAIN_RUN_TAG}}"
export CKPT_NAME="${CKPT_NAME:-ckpt_final.pt}"
export CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/${CKPT_NAME}}"
export WAIT_FOR_PRETRAIN="${WAIT_FOR_PRETRAIN:-1}"
export PRETRAIN_POLL_SEC="${PRETRAIN_POLL_SEC:-60}"

export RUN_SCANOBJECTNN_PREP="${RUN_SCANOBJECTNN_PREP:-0}"
export RUN_ROUTE_A="${RUN_ROUTE_A:-1}"
export RUN_ROUTE_B="${RUN_ROUTE_B:-1}"
export RUN_SHAPENETPART_FT="${RUN_SHAPENETPART_FT:-1}"

export FT_VISIBLE_GPUS="${FT_VISIBLE_GPUS:-0,1,2,3}"
export FT_NPROC_PER_NODE="${FT_NPROC_PER_NODE:-4}"
export FT_VARIANT_ORDER="${FT_VARIANT_ORDER:-obj_bg,obj_only,pb_t50_rs}"
export EVAL_GPU="${EVAL_GPU:-0}"
export FT_SAVE_ROOT="${FT_SAVE_ROOT:-${ROOT_DIR}/runs/patchcls_itachi}"
export FT_LOG_ROOT="${FT_LOG_ROOT:-${LOG_ROOT}/finetune}"
export FT_DATA_FORMAT="${FT_DATA_FORMAT:-scan_h5}"
export FT_SCAN_H5_ROOT_OBJ_BG="${FT_SCAN_H5_ROOT_OBJ_BG:-data/ScanObjectNN/h5_files/main_split}"
export FT_SCAN_H5_ROOT_OBJ_ONLY="${FT_SCAN_H5_ROOT_OBJ_ONLY:-data/ScanObjectNN/h5_files/main_split_nobg}"
export FT_SCAN_H5_ROOT_PB_T50_RS="${FT_SCAN_H5_ROOT_PB_T50_RS:-data/ScanObjectNN/h5_files/main_split}"
export OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
export OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
export PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
export FT_EPOCHS="${FT_EPOCHS:-300}"
export FT_BATCH="${FT_BATCH:-64}"
export FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"
export FT_USE_WANDB="${FT_USE_WANDB:-1}"
export FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-patchnepa-geo-teacher-scanobjectnn}"
export FT_WANDB_ENTITY="${FT_WANDB_ENTITY:-}"
export FT_WANDB_GROUP="${FT_WANDB_GROUP:-itachi_geo_teacher_ft}"
export FT_WANDB_MODE="${FT_WANDB_MODE:-online}"
export FT_VAL_SPLIT_MODE="${FT_VAL_SPLIT_MODE:-pointmae}"
export FT_AUG_EVAL="${FT_AUG_EVAL:-1}"
export FT_MC_EVAL_K_TEST="${FT_MC_EVAL_K_TEST:-10}"

export PARTSEG_VISIBLE_GPUS="${PARTSEG_VISIBLE_GPUS:-0,1,2,3}"
export PARTSEG_NPROC_PER_NODE="${PARTSEG_NPROC_PER_NODE:-4}"
export PARTSEG_SAVE_ROOT="${PARTSEG_SAVE_ROOT:-${ROOT_DIR}/runs/patchpart_itachi}"
export PARTSEG_LOG_ROOT="${PARTSEG_LOG_ROOT:-${LOG_ROOT}/partseg}"
export SHAPENETPART_ROOT="${SHAPENETPART_ROOT:-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
export SHAPENETPART_DATA_ROOT="${SHAPENETPART_DATA_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data}"
export SHAPENETPART_TARGET_DIR="${SHAPENETPART_TARGET_DIR:-/mnt/urashima/users/minesawa/3D-NEPA-data/shapenetcore_partanno_segmentation_benchmark_v0_normal}"
export SHAPENETPART_ZIP_PATH="${SHAPENETPART_ZIP_PATH:-/mnt/urashima/users/minesawa/3D-NEPA-data/downloads/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip}"
export PARTSEG_EPOCHS="${PARTSEG_EPOCHS:-300}"
export PARTSEG_BATCH="${PARTSEG_BATCH:-64}"
export PARTSEG_BATCH_MODE="${PARTSEG_BATCH_MODE:-global}"
export PARTSEG_NUM_WORKERS="${PARTSEG_NUM_WORKERS:-8}"
export PARTSEG_USE_WANDB="${PARTSEG_USE_WANDB:-1}"
export PARTSEG_WANDB_PROJECT="${PARTSEG_WANDB_PROJECT:-patchnepa-geo-teacher-shapenetpart}"
export PARTSEG_WANDB_ENTITY="${PARTSEG_WANDB_ENTITY:-}"
export PARTSEG_WANDB_GROUP="${PARTSEG_WANDB_GROUP:-itachi_geo_teacher_partseg}"
export PARTSEG_WANDB_MODE="${PARTSEG_WANDB_MODE:-online}"
export PARTSEG_PATCHNEPA_FT_MODE="${PARTSEG_PATCHNEPA_FT_MODE:-q_only}"
export PARTSEG_PATCHNEPA_FREEZE_PATCH_EMBED="${PARTSEG_PATCHNEPA_FREEZE_PATCH_EMBED:-1}"

export SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_eval_distnorm_unsigned_same_v1.yaml}"
export OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_eval_distnorm_unsigned_offdiag_v1.yaml}"
export SUITE_LOG_ROOT="${SUITE_LOG_ROOT:-${LOG_ROOT}/cqa_multitype}"
export SUITE_RESULT_ROOT="${SUITE_RESULT_ROOT:-${ROOT_DIR}/results/cqa_multitype_itachi}"
export SUITE_MAX_SAMPLES="${SUITE_MAX_SAMPLES:-4995}"
export SUITE_BATCH="${SUITE_BATCH:-16}"
export SUITE_NUM_WORKERS="${SUITE_NUM_WORKERS:-4}"
export SUITE_SPLIT_OVERRIDE="${SUITE_SPLIT_OVERRIDE:-test}"
export SUITE_N_CTX="${SUITE_N_CTX:-2048}"
export SUITE_N_QRY="${SUITE_N_QRY:-64}"
export SUITE_TASKS="${SUITE_TASKS:-udf_distance,mesh_normal_unsigned}"
export SUITE_CONTROLS="${SUITE_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
export SUITE_QUERY_ORDER="${SUITE_QUERY_ORDER:-sampled}"

export COMPLETION_LOG_ROOT="${COMPLETION_LOG_ROOT:-${LOG_ROOT}/cqa_completion}"
export COMPLETION_RESULT_ROOT="${COMPLETION_RESULT_ROOT:-${ROOT_DIR}/results/cqa_completion_itachi}"
export COMPLETION_SPLIT_OVERRIDE="${COMPLETION_SPLIT_OVERRIDE:-test}"
export COMPLETION_MAX_SHAPES="${COMPLETION_MAX_SHAPES:-16}"
export COMPLETION_BATCH="${COMPLETION_BATCH:-4}"
export COMPLETION_N_CTX="${COMPLETION_N_CTX:-2048}"
export COMPLETION_N_QRY="${COMPLETION_N_QRY:-64}"
export COMPLETION_GRID_RES="${COMPLETION_GRID_RES:-12}"
export COMPLETION_CHUNK_N_QUERY="${COMPLETION_CHUNK_N_QUERY:-64}"
export COMPLETION_TAU_LIST="${COMPLETION_TAU_LIST:-0.01,0.02,0.05}"

export RUN_CURVATURE_PROBE="${RUN_CURVATURE_PROBE:-1}"
export PROBE_LOG_ROOT="${PROBE_LOG_ROOT:-${LOG_ROOT}/cqa_probe}"
export PROBE_SAVE_ROOT="${PROBE_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_probe_itachi}"
export PROBE_RESULT_ROOT="${PROBE_RESULT_ROOT:-${ROOT_DIR}/results/cqa_probe_itachi}"
export PROBE_MAX_STEPS="${PROBE_MAX_STEPS:-5000}"
export PROBE_EVAL_EVERY="${PROBE_EVAL_EVERY:-500}"
export PROBE_BATCH="${PROBE_BATCH:-8}"
export PROBE_NUM_WORKERS="${PROBE_NUM_WORKERS:-4}"
export PROBE_TRAIN_SPLIT="${PROBE_TRAIN_SPLIT:-train}"
export PROBE_EVAL_SPLIT="${PROBE_EVAL_SPLIT:-test}"
export PROBE_N_CTX="${PROBE_N_CTX:-2048}"
export PROBE_N_QRY="${PROBE_N_QRY:-64}"
export PROBE_MAX_TRAIN_SAMPLES="${PROBE_MAX_TRAIN_SAMPLES:-0}"
export PROBE_MAX_EVAL_SAMPLES="${PROBE_MAX_EVAL_SAMPLES:-128}"
export PROBE_CONTROLS="${PROBE_CONTROLS:-correct,no_context,wrong_shape_same_synset,wrong_shape_other_synset,shuffled_query}"
export PROBE_CACHE_ROOT="${PROBE_CACHE_ROOT:-data/shapenet_cache_v2_20260401_worldvis}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] geo-teacher posttrain chain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting in foreground"
  echo "[info] log=${LOG_FILE}"
  bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_posttrain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] geo-teacher posttrain chain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      ROOT_DIR PRETRAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE PYTHON_BIN PRETRAIN_SAVE_ROOT \
      PRETRAIN_SAVE_DIR CKPT_NAME CKPT_PATH WAIT_FOR_PRETRAIN PRETRAIN_POLL_SEC \
      RUN_SCANOBJECTNN_PREP RUN_ROUTE_A RUN_ROUTE_B RUN_SHAPENETPART_FT FT_VISIBLE_GPUS FT_NPROC_PER_NODE FT_VARIANT_ORDER EVAL_GPU FT_SAVE_ROOT \
      FT_LOG_ROOT FT_DATA_FORMAT FT_SCAN_H5_ROOT_OBJ_BG FT_SCAN_H5_ROOT_OBJ_ONLY FT_SCAN_H5_ROOT_PB_T50_RS \
      OBJ_BG_CACHE OBJ_ONLY_CACHE PB_T50_RS_CACHE FT_EPOCHS FT_BATCH FT_NUM_WORKERS FT_USE_WANDB FT_WANDB_PROJECT \
      FT_WANDB_ENTITY FT_WANDB_GROUP FT_WANDB_MODE FT_VAL_SPLIT_MODE FT_AUG_EVAL \
      FT_MC_EVAL_K_TEST PARTSEG_VISIBLE_GPUS PARTSEG_NPROC_PER_NODE PARTSEG_SAVE_ROOT PARTSEG_LOG_ROOT \
      SHAPENETPART_ROOT SHAPENETPART_DATA_ROOT SHAPENETPART_TARGET_DIR SHAPENETPART_ZIP_PATH \
      PARTSEG_EPOCHS PARTSEG_BATCH PARTSEG_BATCH_MODE PARTSEG_NUM_WORKERS PARTSEG_USE_WANDB PARTSEG_WANDB_PROJECT \
      PARTSEG_WANDB_ENTITY PARTSEG_WANDB_GROUP PARTSEG_WANDB_MODE PARTSEG_PATCHNEPA_FT_MODE \
      PARTSEG_PATCHNEPA_FREEZE_PATCH_EMBED SAME_MIX_CONFIG OFFDIAG_MIX_CONFIG SUITE_LOG_ROOT \
      SUITE_RESULT_ROOT SUITE_MAX_SAMPLES SUITE_BATCH SUITE_NUM_WORKERS \
      SUITE_SPLIT_OVERRIDE SUITE_N_CTX SUITE_N_QRY SUITE_TASKS SUITE_CONTROLS \
      SUITE_QUERY_ORDER COMPLETION_LOG_ROOT COMPLETION_RESULT_ROOT \
      COMPLETION_SPLIT_OVERRIDE COMPLETION_MAX_SHAPES COMPLETION_BATCH \
      COMPLETION_N_CTX COMPLETION_N_QRY COMPLETION_GRID_RES COMPLETION_CHUNK_N_QUERY \
      COMPLETION_TAU_LIST RUN_CURVATURE_PROBE PROBE_LOG_ROOT PROBE_SAVE_ROOT \
      PROBE_RESULT_ROOT PROBE_MAX_STEPS PROBE_EVAL_EVERY PROBE_BATCH PROBE_NUM_WORKERS \
      PROBE_TRAIN_SPLIT PROBE_EVAL_SPLIT PROBE_N_CTX PROBE_N_QRY \
      PROBE_MAX_TRAIN_SAMPLES PROBE_MAX_EVAL_SAMPLES PROBE_CONTROLS PROBE_CACHE_ROOT
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_posttrain_inner.sh'"
    sleep 1
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached geo-teacher posttrain chain in tmux"
      echo "[info] session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    echo "[error] failed to start tmux session=${TMUX_SESSION}"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_posttrain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached geo-teacher posttrain chain"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached geo-teacher posttrain chain"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
