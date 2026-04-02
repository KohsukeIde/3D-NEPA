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

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PRETRAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
export PRETRAIN_SAVE_DIR="${PRETRAIN_SAVE_DIR:-${PRETRAIN_SAVE_ROOT}/${PRETRAIN_RUN_TAG}}"
export CKPT_NAME="${CKPT_NAME:-ckpt_final.pt}"
export CKPT_PATH="${CKPT_PATH:-${PRETRAIN_SAVE_DIR}/${CKPT_NAME}}"
export WAIT_FOR_PRETRAIN="${WAIT_FOR_PRETRAIN:-1}"
export PRETRAIN_POLL_SEC="${PRETRAIN_POLL_SEC:-60}"

export RUN_SCANOBJECTNN_PREP="${RUN_SCANOBJECTNN_PREP:-1}"
export RUN_ROUTE_A="${RUN_ROUTE_A:-1}"
export RUN_ROUTE_B="${RUN_ROUTE_B:-1}"

export FT_GPU_IDS="${FT_GPU_IDS:-0,1,2}"
export EVAL_GPU="${EVAL_GPU:-3}"
export FT_SAVE_ROOT="${FT_SAVE_ROOT:-${ROOT_DIR}/runs/patchcls_itachi}"
export FT_LOG_ROOT="${FT_LOG_ROOT:-${LOG_ROOT}/finetune}"
export OBJ_BG_CACHE="${OBJ_BG_CACHE:-data/scanobjectnn_obj_bg_v3_nonorm}"
export OBJ_ONLY_CACHE="${OBJ_ONLY_CACHE:-data/scanobjectnn_obj_only_v3_nonorm}"
export PB_T50_RS_CACHE="${PB_T50_RS_CACHE:-data/scanobjectnn_pb_t50_rs_v3_nonorm}"
export FT_EPOCHS="${FT_EPOCHS:-300}"
export FT_BATCH="${FT_BATCH:-64}"
export FT_NUM_WORKERS="${FT_NUM_WORKERS:-8}"
export FT_USE_WANDB="${FT_USE_WANDB:-1}"
export FT_WANDB_PROJECT="${FT_WANDB_PROJECT:-patchnepa-finetune}"
export FT_WANDB_ENTITY="${FT_WANDB_ENTITY:-}"
export FT_WANDB_GROUP="${FT_WANDB_GROUP:-itachi_geo_teacher_ft}"
export FT_WANDB_MODE="${FT_WANDB_MODE:-offline}"
export FT_VAL_SPLIT_MODE="${FT_VAL_SPLIT_MODE:-pointmae}"
export FT_AUG_EVAL="${FT_AUG_EVAL:-1}"
export FT_MC_EVAL_K_TEST="${FT_MC_EVAL_K_TEST:-10}"

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
  exec bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_posttrain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
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
    for name in \
      ROOT_DIR PRETRAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE PYTHON_BIN PRETRAIN_SAVE_ROOT \
      PRETRAIN_SAVE_DIR CKPT_NAME CKPT_PATH WAIT_FOR_PRETRAIN PRETRAIN_POLL_SEC \
      RUN_SCANOBJECTNN_PREP RUN_ROUTE_A RUN_ROUTE_B FT_GPU_IDS EVAL_GPU FT_SAVE_ROOT \
      FT_LOG_ROOT OBJ_BG_CACHE OBJ_ONLY_CACHE PB_T50_RS_CACHE FT_EPOCHS FT_BATCH FT_NUM_WORKERS FT_USE_WANDB FT_WANDB_PROJECT \
      FT_WANDB_ENTITY FT_WANDB_GROUP FT_WANDB_MODE FT_VAL_SPLIT_MODE FT_AUG_EVAL \
      FT_MC_EVAL_K_TEST SAME_MIX_CONFIG OFFDIAG_MIX_CONFIG SUITE_LOG_ROOT \
      SUITE_RESULT_ROOT SUITE_MAX_SAMPLES SUITE_BATCH SUITE_NUM_WORKERS \
      SUITE_SPLIT_OVERRIDE SUITE_N_CTX SUITE_N_QRY SUITE_TASKS SUITE_CONTROLS \
      SUITE_QUERY_ORDER COMPLETION_LOG_ROOT COMPLETION_RESULT_ROOT \
      COMPLETION_SPLIT_OVERRIDE COMPLETION_MAX_SHAPES COMPLETION_BATCH \
      COMPLETION_N_CTX COMPLETION_N_QRY COMPLETION_GRID_RES COMPLETION_CHUNK_N_QUERY \
      COMPLETION_TAU_LIST
    do
      tmux set-environment -g "${name}" "${!name:-}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_posttrain_inner.sh'"
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
