#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

SPLIT_JOB_ID="${SPLIT_JOB_ID:?set SPLIT_JOB_ID (e.g. 101730.qjcm)}"
MAT_JOB_ID="${MAT_JOB_ID:?set MAT_JOB_ID (e.g. 101731.qjcm)}"
OUT_ROOT="${OUT_ROOT:-data/shapenet_unpaired_cache_v2_20260303}"
OUT_JSON="${OUT_JSON:-data/shapenet_unpaired_splits_v2_20260303.json}"
RUNLOG="${RUNLOG:-nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md}"
POLL_SEC="${POLL_SEC:-300}"
TIMEOUT_SEC="${TIMEOUT_SEC:-604800}" # 7 days
STATE_LOG="${STATE_LOG:-logs/preprocess/shapenet_unpaired/watch_${SPLIT_JOB_ID%.*}_${MAT_JOB_ID%.*}.log}"

mkdir -p "$(dirname "${STATE_LOG}")"

job_field() {
  local jid="$1"
  local key_re="$2"
  qstat -xf "${jid}" 2>/dev/null | awk -F' = ' -v k="${key_re}" '$1 ~ k {print $2; exit}'
}

job_state() {
  local jid="$1"
  local st
  st="$(job_field "${jid}" "job_state" || true)"
  if [[ -z "${st}" ]]; then
    echo "UNK"
  else
    echo "${st}"
  fi
}

job_exit() {
  local jid="$1"
  local ex
  ex="$(job_field "${jid}" "Exit_status|exit_status" || true)"
  if [[ -z "${ex}" ]]; then
    echo "NA"
  else
    echo "${ex}"
  fi
}

ts() {
  date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(ts)] watch start: split=${SPLIT_JOB_ID} mat=${MAT_JOB_ID}" | tee -a "${STATE_LOG}"
start_epoch="$(date +%s)"
complete=0
complete_reason=""

while true; do
  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  if (( elapsed > TIMEOUT_SEC )); then
    echo "[$(ts)] timeout after ${elapsed}s" | tee -a "${STATE_LOG}"
    break
  fi

  s_split="$(job_state "${SPLIT_JOB_ID}")"
  s_mat="$(job_state "${MAT_JOB_ID}")"
  e_split="$(job_exit "${SPLIT_JOB_ID}")"
  e_mat="$(job_exit "${MAT_JOB_ID}")"
  echo "[$(ts)] split(state=${s_split},exit=${e_split}) mat(state=${s_mat},exit=${e_mat})" | tee -a "${STATE_LOG}"

  if [[ "${s_mat}" == "F" || "${s_mat}" == "E" ]]; then
    complete=1
    complete_reason="mat_job_finished:${s_mat}"
    break
  fi
  if [[ "${s_mat}" == "UNK" && -f "${OUT_ROOT}/_meta/split_source.json" ]]; then
    # job history may be purged; materialize metadata indicates completion.
    complete=1
    complete_reason="mat_job_purged_meta_present"
    break
  fi
  sleep "${POLL_SEC}"
done

if [[ "${complete}" != "1" ]]; then
  echo "[$(ts)] watch ended without completion; skip runlog append" | tee -a "${STATE_LOG}"
  exit 0
fi

s_split="$(job_state "${SPLIT_JOB_ID}")"
s_mat="$(job_state "${MAT_JOB_ID}")"
e_split="$(job_exit "${SPLIT_JOB_ID}")"
e_mat="$(job_exit "${MAT_JOB_ID}")"

count_npz() {
  local d="$1"
  if [[ -d "${d}" ]]; then
    find "${d}" -type f -name '*.npz' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

n_mesh="$(count_npz "${OUT_ROOT}/train_mesh")"
n_pc="$(count_npz "${OUT_ROOT}/train_pc")"
n_udf="$(count_npz "${OUT_ROOT}/train_udf")"
n_eval="$(count_npz "${OUT_ROOT}/eval")"

meta_line="meta_missing"
if [[ -f "${OUT_ROOT}/_meta/split_source.json" ]]; then
  meta_line="$(tr '\n' ' ' < "${OUT_ROOT}/_meta/split_source.json" | sed -E 's/[[:space:]]+/ /g' | cut -c1-500)"
fi

{
  echo ""
  echo "## 118. Auto watch result for unpaired split/materialize (${SPLIT_JOB_ID}, ${MAT_JOB_ID}) ($(date +%Y-%m-%d))"
  echo ""
  echo "Watcher log:"
  echo ""
  echo "- \`${STATE_LOG}\`"
  echo ""
  echo "Final states:"
  echo ""
  echo "- \`${SPLIT_JOB_ID}\`: state=\`${s_split}\`, exit=\`${e_split}\`"
  echo "- \`${MAT_JOB_ID}\`: state=\`${s_mat}\`, exit=\`${e_mat}\`"
  echo ""
  echo "Outputs:"
  echo ""
  echo "- split json: \`${OUT_JSON}\`"
  echo "- materialized root: \`${OUT_ROOT}\`"
  echo "- counts: train_mesh=\`${n_mesh}\`, train_pc=\`${n_pc}\`, train_udf=\`${n_udf}\`, eval=\`${n_eval}\`"
  echo ""
  echo "Materialize meta snapshot:"
  echo ""
  echo "- \`${meta_line}\`"
} >> "${RUNLOG}"

echo "[$(ts)] completion_reason=${complete_reason}" | tee -a "${STATE_LOG}"
echo "[$(ts)] appended summary to ${RUNLOG}" | tee -a "${STATE_LOG}"
