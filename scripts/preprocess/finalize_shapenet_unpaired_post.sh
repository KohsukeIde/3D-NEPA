#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${ROOT_DIR}"

SPLIT_JOB_ID="${SPLIT_JOB_ID:?set SPLIT_JOB_ID}"
MAT_JOB_ID="${MAT_JOB_ID:?set MAT_JOB_ID}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
OUT_JSON="${OUT_JSON:?set OUT_JSON}"
RUNLOG="${RUNLOG:-nepa3d/docs/patch_nepa/runlog_patch_nepa_202602.md}"
FINAL_LOG="${FINAL_LOG:-logs/preprocess/shapenet_unpaired/finalize_${SPLIT_JOB_ID%.*}_${MAT_JOB_ID%.*}.log}"

mkdir -p "$(dirname "${FINAL_LOG}")"

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

count_npz() {
  local d="$1"
  if [[ -d "${d}" ]]; then
    find "${d}" -type f -name '*.npz' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

split_state="$(job_state "${SPLIT_JOB_ID}")"
split_exit="$(job_exit "${SPLIT_JOB_ID}")"
mat_state="$(job_state "${MAT_JOB_ID}")"
mat_exit="$(job_exit "${MAT_JOB_ID}")"

n_mesh="$(count_npz "${OUT_ROOT}/train_mesh")"
n_pc="$(count_npz "${OUT_ROOT}/train_pc")"
n_udf="$(count_npz "${OUT_ROOT}/train_udf")"
n_eval="$(count_npz "${OUT_ROOT}/eval")"

meta_line="meta_missing"
if [[ -f "${OUT_ROOT}/_meta/split_source.json" ]]; then
  meta_line="$(tr '\n' ' ' < "${OUT_ROOT}/_meta/split_source.json" | sed -E 's/[[:space:]]+/ /g' | cut -c1-500)"
fi

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finalize start"
  echo "split=${SPLIT_JOB_ID} state=${split_state} exit=${split_exit}"
  echo "mat=${MAT_JOB_ID} state=${mat_state} exit=${mat_exit}"
  echo "out_json=${OUT_JSON}"
  echo "out_root=${OUT_ROOT}"
  echo "counts train_mesh=${n_mesh} train_pc=${n_pc} train_udf=${n_udf} eval=${n_eval}"
  echo "meta=${meta_line}"
} | tee -a "${FINAL_LOG}"

{
  echo ""
  echo "## 118. Unpaired split/materialize completed (${SPLIT_JOB_ID}, ${MAT_JOB_ID}) ($(date +%Y-%m-%d))"
  echo ""
  echo "Final states:"
  echo ""
  echo "- \`${SPLIT_JOB_ID}\`: state=\`${split_state}\`, exit=\`${split_exit}\`"
  echo "- \`${MAT_JOB_ID}\`: state=\`${mat_state}\`, exit=\`${mat_exit}\`"
  echo ""
  echo "Outputs:"
  echo ""
  echo "- split json: \`${OUT_JSON}\`"
  echo "- materialized root: \`${OUT_ROOT}\`"
  echo "- counts: train_mesh=\`${n_mesh}\`, train_pc=\`${n_pc}\`, train_udf=\`${n_udf}\`, eval=\`${n_eval}\`"
  echo "- meta snapshot: \`${meta_line}\`"
  echo "- finalize log: \`${FINAL_LOG}\`"
} >> "${RUNLOG}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] runlog appended: ${RUNLOG}" | tee -a "${FINAL_LOG}"

