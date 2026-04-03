#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

CHAIN_RUN_TAG="${CHAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_300ep_itachi_main}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher_chain}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${CHAIN_RUN_TAG}.chain.pid}"
TMUX_SESSION="${TMUX_SESSION:-${CHAIN_RUN_TAG//[^[:alnum:]_]/_}_chain}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR CHAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE
export PREV_POSTTRAIN_RUN_TAG="${PREV_POSTTRAIN_RUN_TAG:-geo_teacher_distnorm_unsigned_100ep_itachi_main}"
export WAIT_FOR_PREV_POSTTRAIN="${WAIT_FOR_PREV_POSTTRAIN:-1}"
export PREV_POSTTRAIN_POLL_SEC="${PREV_POSTTRAIN_POLL_SEC:-120}"

export PRETRAIN_SAVE_ROOT="${PRETRAIN_SAVE_ROOT:-${ROOT_DIR}/runs/cqa_itachi}"
export PRETRAIN_LOG_ROOT="${PRETRAIN_LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geo_teacher}"
export PYTHON_BIN="${PYTHON_BIN:-/home/minesawa/anaconda3/bin/python}"
export MIX_CONFIG="${MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distnorm_unsigned_v1.yaml}"
export PRETRAIN_VISIBLE_GPUS="${PRETRAIN_VISIBLE_GPUS:-0,1,2,3}"
export PRETRAIN_NPROC_PER_NODE="${PRETRAIN_NPROC_PER_NODE:-4}"
export PRETRAIN_MASTER_PORT="${PRETRAIN_MASTER_PORT:-29634}"
export PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
export PRETRAIN_BATCH="${PRETRAIN_BATCH:-32}"
export PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-4}"
export PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
export PRETRAIN_WEIGHT_DECAY="${PRETRAIN_WEIGHT_DECAY:-0.05}"
export PRETRAIN_USE_WANDB="${PRETRAIN_USE_WANDB:-1}"
export PRETRAIN_WANDB_PROJECT="${PRETRAIN_WANDB_PROJECT:-patchnepa-cqa-pretrain}"
export PRETRAIN_WANDB_ENTITY="${PRETRAIN_WANDB_ENTITY:-}"
export PRETRAIN_WANDB_GROUP="${PRETRAIN_WANDB_GROUP:-itachi_geo_teacher_full300}"
export PRETRAIN_WANDB_TAGS="${PRETRAIN_WANDB_TAGS:-local,itachi,cqa,geo_teacher,distnorm_unsigned,full300}"
export PRETRAIN_WANDB_MODE="${PRETRAIN_WANDB_MODE:-online}"

export RUN_ROUTE_A="${RUN_ROUTE_A:-1}"
export RUN_SHAPENETPART_FT="${RUN_SHAPENETPART_FT:-1}"
export RUN_ROUTE_B="${RUN_ROUTE_B:-1}"
export FT_VISIBLE_GPUS="${FT_VISIBLE_GPUS:-0,1,2,3}"
export FT_NPROC_PER_NODE="${FT_NPROC_PER_NODE:-4}"
export PARTSEG_VISIBLE_GPUS="${PARTSEG_VISIBLE_GPUS:-0,1,2,3}"
export PARTSEG_NPROC_PER_NODE="${PARTSEG_NPROC_PER_NODE:-4}"
export FT_DATA_FORMAT="${FT_DATA_FORMAT:-scan_h5}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] full300 chain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "[info] starting in foreground"
  echo "[info] log=${LOG_FILE}"
  bash "${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_full300_chain_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    if ! command -v tmux >/dev/null 2>&1; then
      echo "[error] tmux not found"
      exit 1
    fi
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] full300 chain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      ROOT_DIR CHAIN_RUN_TAG LOG_ROOT LOG_FILE PID_FILE PREV_POSTTRAIN_RUN_TAG WAIT_FOR_PREV_POSTTRAIN PREV_POSTTRAIN_POLL_SEC \
      PRETRAIN_SAVE_ROOT PRETRAIN_LOG_ROOT PYTHON_BIN MIX_CONFIG PRETRAIN_VISIBLE_GPUS PRETRAIN_NPROC_PER_NODE PRETRAIN_MASTER_PORT \
      PRETRAIN_EPOCHS PRETRAIN_BATCH PRETRAIN_NUM_WORKERS PRETRAIN_LR PRETRAIN_WEIGHT_DECAY PRETRAIN_USE_WANDB PRETRAIN_WANDB_PROJECT \
      PRETRAIN_WANDB_ENTITY PRETRAIN_WANDB_GROUP PRETRAIN_WANDB_TAGS PRETRAIN_WANDB_MODE RUN_ROUTE_A RUN_SHAPENETPART_FT RUN_ROUTE_B \
      FT_VISIBLE_GPUS FT_NPROC_PER_NODE PARTSEG_VISIBLE_GPUS PARTSEG_NPROC_PER_NODE FT_DATA_FORMAT
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_full300_chain_inner.sh'"
    sleep 1
    if tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] started detached full300 chain in tmux"
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
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geo_teacher/_run_geo_teacher_full300_chain_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    if kill -0 "${child_pid}" 2>/dev/null; then
      echo "[info] started detached full300 chain"
      echo "[info] pid=${child_pid}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    rm -f "${PID_FILE}"
    echo "[error] failed to start detached full300 chain"
    echo "[error] inspect log=${LOG_FILE}"
    exit 1
    ;;
  *)
    echo "[error] unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    exit 1
    ;;
esac
