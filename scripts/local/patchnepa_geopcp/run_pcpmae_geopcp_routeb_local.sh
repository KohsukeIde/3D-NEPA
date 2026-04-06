#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_pcpmae_geopcp_common.sh"

ARM_TAG="${ARM_TAG:-pcp_worldvis_base_100ep}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:?set PRETRAIN_CKPT to a PCP-MAE / Geo-PCP pretrain checkpoint}"
LOG_ROOT="${LOG_ROOT:-/mnt/urashima/users/minesawa/3D-NEPA-data/logs/patchnepa_geopcp/routeb}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${ARM_TAG}.routeb.log}"
PID_FILE="${PID_FILE:-${LOG_ROOT}/${ARM_TAG}.routeb.pid}"
TMUX_SESSION="${TMUX_SESSION:-${ARM_TAG//[^[:alnum:]_]/_}_routeb}"
LAUNCH_MODE="${LAUNCH_MODE:-tmux}"
ENV_FILE="${ENV_FILE:-${LOG_ROOT}/.${TMUX_SESSION}.env}"

mkdir -p "${LOG_ROOT}"

export ROOT_DIR PCP_ROOT PYTHON_BIN CONDA_PREFIX CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST MAX_JOBS PYTHONPATH PATH LD_LIBRARY_PATH
export ARM_TAG PRETRAIN_CKPT LOG_ROOT LOG_FILE PID_FILE
export ROUTEB_TRAIN_RUN_TAG="${ROUTEB_TRAIN_RUN_TAG:-${ARM_TAG}__external_pointmae_udfdist}"
export ROUTEB_SAVE_DIR="${ROUTEB_SAVE_DIR:-${ROOT_DIR}/runs/cqa_itachi/${ROUTEB_TRAIN_RUN_TAG}}"
export ROUTEB_TRAIN_MIX_CONFIG="${ROUTEB_TRAIN_MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_packed_distance_only_v1.yaml}"
export SAME_MIX_CONFIG="${SAME_MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_eval_distnorm_unsigned_same_v1.yaml}"
export OFFDIAG_MIX_CONFIG="${OFFDIAG_MIX_CONFIG:-nepa3d/tracks/patch_nepa/cqa/configs/shapenet_geo_teacher_eval_distnorm_unsigned_offdiag_v1.yaml}"
export ROUTEB_MAX_STEPS="${ROUTEB_MAX_STEPS:-10000}"
export ROUTEB_EPOCHS="${ROUTEB_EPOCHS:-100}"
export ROUTEB_BATCH="${ROUTEB_BATCH:-32}"
export ROUTEB_NUM_WORKERS="${ROUTEB_NUM_WORKERS:-8}"
export ROUTEB_SEED="${ROUTEB_SEED:-0}"
export ROUTEB_DEVICE="${ROUTEB_DEVICE:-cuda}"
export ROUTEB_GPU="${ROUTEB_GPU:-0}"
export ROUTEB_N_CTX="${ROUTEB_N_CTX:-2048}"
export ROUTEB_N_QRY="${ROUTEB_N_QRY:-64}"
export ROUTEB_MAX_SAMPLES="${ROUTEB_MAX_SAMPLES:-512}"
export ROUTEB_COMPLETION_MAX_SHAPES="${ROUTEB_COMPLETION_MAX_SHAPES:-16}"
export ROUTEB_PROBE_MAX_STEPS="${ROUTEB_PROBE_MAX_STEPS:-5000}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "[info] geopcp Route-B chain already running (pid=${existing_pid})"
    echo "[info] log=${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  bash "${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_routeb_inner.sh" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

case "${LAUNCH_MODE}" in
  tmux)
    command -v tmux >/dev/null 2>&1 || geopcp_die "tmux not found"
    if env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null; then
      echo "[info] geopcp Route-B chain already running in tmux session=${TMUX_SESSION}"
      echo "[info] log=${LOG_FILE}"
      exit 0
    fi
    : > "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
    for name in \
      ROOT_DIR PCP_ROOT PYTHON_BIN CONDA_PREFIX CUDA_HOME CC CXX TORCH_CUDA_ARCH_LIST MAX_JOBS PYTHONPATH PATH LD_LIBRARY_PATH \
      ARM_TAG PRETRAIN_CKPT LOG_ROOT LOG_FILE PID_FILE ROUTEB_TRAIN_RUN_TAG ROUTEB_SAVE_DIR ROUTEB_TRAIN_MIX_CONFIG \
      SAME_MIX_CONFIG OFFDIAG_MIX_CONFIG ROUTEB_MAX_STEPS ROUTEB_EPOCHS ROUTEB_BATCH ROUTEB_NUM_WORKERS ROUTEB_SEED \
      ROUTEB_DEVICE ROUTEB_GPU ROUTEB_N_CTX ROUTEB_N_QRY ROUTEB_MAX_SAMPLES ROUTEB_COMPLETION_MAX_SHAPES ROUTEB_PROBE_MAX_STEPS
    do
      printf 'export %s=%q\n' "${name}" "${!name:-}" >> "${ENV_FILE}"
    done
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux new-session -d -s "${TMUX_SESSION}" \
      "cd '${ROOT_DIR}' && source '${ENV_FILE}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_routeb_inner.sh'"
    sleep 1
    env -u LD_LIBRARY_PATH -u PYTHONPATH tmux has-session -t "=${TMUX_SESSION}" 2>/dev/null || geopcp_die "failed to start tmux session=${TMUX_SESSION}"
    echo "[info] started detached Geo-PCP Route-B chain in tmux"
    echo "[info] session=${TMUX_SESSION}"
    echo "[info] log=${LOG_FILE}"
    ;;
  nohup)
    nohup bash -lc \
      "cd '${ROOT_DIR}' && bash '${ROOT_DIR}/scripts/local/patchnepa_geopcp/_run_pcpmae_geopcp_routeb_inner.sh'" \
      >> "${LOG_FILE}" 2>&1 < /dev/null &
    child_pid=$!
    echo "${child_pid}" > "${PID_FILE}"
    sleep 1
    kill -0 "${child_pid}" 2>/dev/null || geopcp_die "failed to start detached Geo-PCP Route-B chain"
    echo "[info] started detached Geo-PCP Route-B chain"
    echo "[info] pid=${child_pid}"
    echo "[info] log=${LOG_FILE}"
    ;;
  *)
    geopcp_die "unsupported LAUNCH_MODE=${LAUNCH_MODE}"
    ;;
esac
