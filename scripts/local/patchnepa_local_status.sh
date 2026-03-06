#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

QUEUE_FILE="${QUEUE_FILE:-${ROOT_DIR}/scripts/local/patchnepa_local_queue.tsv}"
QUEUE_NAME="${QUEUE_NAME:-patchnepa_local}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${ROOT_DIR}/logs/local_queue/${QUEUE_NAME}}"
STATE_FILE="${RUNTIME_ROOT}/state.tsv"

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

latest_state_line() {
  local id="$1"
  if [[ ! -f "${STATE_FILE}" ]]; then
    return 0
  fi
  awk -F'\t' -v qid="${id}" 'NR > 1 && $2 == qid {line = $0} END {if (line != "") print line}' "${STATE_FILE}"
}

if [[ ! -f "${QUEUE_FILE}" ]]; then
  echo "[error] queue file not found: ${QUEUE_FILE}" >&2
  exit 1
fi

printf "id\tpriority\tenabled\tstatus\tgpu\tkind\tbacklog_id\tdone_check_present\tartifact_path\n"
while IFS=$'\t' read -r id enabled priority gpu_group kind launch_cmd workdir done_check log_path artifact_path backlog_id canon_doc decision_rule; do
  if [[ "${id}" == "id" || "${id}" =~ ^# ]]; then
    continue
  fi
  latest="$(latest_state_line "${id}")"
  status="queued"
  gpu="-"
  if [[ -n "${latest}" ]]; then
    status="$(printf "%s" "${latest}" | awk -F'\t' '{print $4}')"
    gpu="$(printf "%s" "${latest}" | awk -F'\t' '{print $3}')"
  elif [[ "${enabled}" != "1" ]]; then
    status="disabled"
  fi

  done_abs="$(abs_path "${done_check}")"
  done_present="no"
  if [[ -n "${done_abs}" && -e "${done_abs}" ]]; then
    done_present="yes"
  fi
  artifact_abs="$(abs_path "${artifact_path}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${id}" "${priority}" "${enabled}" "${status}" "${gpu}" "${kind}" "${backlog_id}" "${done_present}" "${artifact_abs:-"-"}"
done < "${QUEUE_FILE}" | if command -v column >/dev/null 2>&1; then column -t -s $'\t'; else cat; fi
