#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QUEUE_FILE="${QUEUE_FILE:-${ROOT_DIR}/scripts/local/patchnepa_local_queue.tsv}"
QUEUE_NAME="${QUEUE_NAME:-patchnepa_local}"
GPU_IDS="${GPU_IDS:-0}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${ROOT_DIR}/logs/local_queue/${QUEUE_NAME}}"
STATE_FILE="${RUNTIME_ROOT}/state.tsv"
QUEUE_PID_FILE="${RUNTIME_ROOT}/queue.pid"

if [[ ! -f "${QUEUE_FILE}" ]]; then
  echo "[error] queue file not found: ${QUEUE_FILE}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"

init_state_file() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    printf "timestamp\tid\tgpu\tstatus\trc\tlog_path\tartifact_path\n" > "${STATE_FILE}"
  fi
}

append_state() {
  local id="$1"
  local gpu="$2"
  local status="$3"
  local rc="$4"
  local log_path="$5"
  local artifact_path="$6"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date -Is)" "${id}" "${gpu}" "${status}" "${rc}" "${log_path}" "${artifact_path}" >> "${STATE_FILE}"
}

latest_status() {
  local id="$1"
  awk -F'\t' -v qid="${id}" 'NR > 1 && $2 == qid {s = $4} END {if (s != "") print s}' "${STATE_FILE}"
}

normalize_field() {
  local val="${1:-}"
  if [[ "${val}" == "-" ]]; then
    printf ""
  else
    printf "%s" "${val}"
  fi
}

abs_path() {
  local raw
  raw="$(normalize_field "${1:-}")"
  if [[ -z "${raw}" ]]; then
    printf ""
  elif [[ "${raw}" = /* ]]; then
    printf "%s" "${raw}"
  else
    printf "%s/%s" "${ROOT_DIR}" "${raw}"
  fi
}

log_runner() {
  local gpu="$1"
  shift
  local msg="$*"
  local runner_log="${RUNTIME_ROOT}/runner_gpu${gpu}.log"
  printf "[%s][gpu%s] %s\n" "$(date '+%F %T')" "${gpu}" "${msg}" | tee -a "${runner_log}"
}

run_one() {
  local gpu="$1"
  local id="$2"
  local enabled="$3"
  local priority="$4"
  local gpu_group="$5"
  local kind="$6"
  local launch_cmd="$7"
  local workdir="$8"
  local done_check="$9"
  local log_path="${10}"
  local artifact_path="${11}"
  local backlog_id="${12}"
  local canon_doc="${13}"
  local decision_rule="${14}"

  if [[ "${enabled}" != "1" ]]; then
    return 0
  fi
  if [[ -n "${gpu_group}" && "${gpu_group}" != "any" && "${gpu_group}" != "${gpu}" ]]; then
    return 0
  fi

  local current_status
  current_status="$(latest_status "${id}")"
  if [[ "${current_status}" == "done" || "${current_status}" == "failed" || "${current_status}" == "skipped" ]]; then
    return 0
  fi

  local done_check_abs log_abs artifact_abs workdir_abs
  done_check_abs="$(abs_path "${done_check}")"
  log_abs="$(abs_path "${log_path}")"
  artifact_abs="$(abs_path "${artifact_path}")"
  workdir_abs="$(abs_path "${workdir}")"
  if [[ -z "${workdir_abs}" ]]; then
    workdir_abs="${ROOT_DIR}"
  fi

  if [[ -n "${done_check_abs}" && -e "${done_check_abs}" ]]; then
    append_state "${id}" "${gpu}" "skipped" "0" "${log_abs}" "${artifact_abs}"
    log_runner "${gpu}" "skip ${id} priority=${priority} kind=${kind} (done_check exists)"
    return 0
  fi

  mkdir -p "$(dirname "${log_abs}")"
  if [[ -n "${artifact_abs}" ]]; then
    mkdir -p "$(dirname "${artifact_abs}")"
  fi

  append_state "${id}" "${gpu}" "running" "0" "${log_abs}" "${artifact_abs}"
  log_runner "${gpu}" "start ${id} priority=${priority} kind=${kind} backlog=${backlog_id}"
  log_runner "${gpu}" "canon_doc=${canon_doc}"
  log_runner "${gpu}" "decision_rule=${decision_rule}"

  (
    cd "${workdir_abs}"
    bash -lc "${launch_cmd}"
  ) > "${log_abs}" 2>&1
  local rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    if [[ -n "${done_check_abs}" && ! -e "${done_check_abs}" ]]; then
      append_state "${id}" "${gpu}" "failed" "86" "${log_abs}" "${artifact_abs}"
      log_runner "${gpu}" "fail ${id} rc=86 (command succeeded but done_check missing)"
      return 86
    fi
    append_state "${id}" "${gpu}" "done" "0" "${log_abs}" "${artifact_abs}"
    log_runner "${gpu}" "done ${id}"
    return 0
  fi

  append_state "${id}" "${gpu}" "failed" "${rc}" "${log_abs}" "${artifact_abs}"
  log_runner "${gpu}" "fail ${id} rc=${rc}"
  return "${rc}"
}

worker() {
  local gpu="$1"
  local worker_index="$2"
  local worker_count="$3"
  local runner_log="${RUNTIME_ROOT}/runner_gpu${gpu}.log"
  : > "${runner_log}"
  echo $$ > "${RUNTIME_ROOT}/worker_gpu${gpu}.pid"

  local sorted_manifest
  sorted_manifest="$(mktemp)"
  awk -F'\t' 'NR > 1 && $0 !~ /^#/ {print}' "${QUEUE_FILE}" | sort -t $'\t' -k3,3n -k1,1 > "${sorted_manifest}"

  local row_index=0
  while IFS=$'\t' read -r id enabled priority gpu_group kind launch_cmd workdir done_check log_path artifact_path backlog_id canon_doc decision_rule; do
    if (( row_index % worker_count != worker_index )); then
      row_index=$((row_index + 1))
      continue
    fi
    run_one "${gpu}" "${id}" "${enabled}" "${priority}" "${gpu_group}" "${kind}" "${launch_cmd}" "${workdir}" "${done_check}" "${log_path}" "${artifact_path}" "${backlog_id}" "${canon_doc}" "${decision_rule}" || true
    row_index=$((row_index + 1))
  done < "${sorted_manifest}"

  rm -f "${sorted_manifest}"
}

if [[ -f "${QUEUE_PID_FILE}" ]]; then
  existing_pid="$(cat "${QUEUE_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && ps -p "${existing_pid}" >/dev/null 2>&1; then
    echo "[attach] queue ${QUEUE_NAME} already running pid=${existing_pid}"
    exit 0
  fi
fi

init_state_file
echo $$ > "${QUEUE_PID_FILE}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
worker_count="${#GPU_ARRAY[@]}"
if [[ "${worker_count}" -lt 1 ]]; then
  echo "[error] no GPUs configured in GPU_IDS=${GPU_IDS}" >&2
  rm -f "${QUEUE_PID_FILE}"
  exit 1
fi

cleanup() {
  rm -f "${QUEUE_PID_FILE}"
  for gpu in "${GPU_ARRAY[@]}"; do
    rm -f "${RUNTIME_ROOT}/worker_gpu${gpu}.pid"
  done
}
trap cleanup EXIT

pids=()
for idx in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$idx]}"
  worker "${gpu}" "${idx}" "${worker_count}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "${pid}" || rc=$?
done

exit "${rc}"
